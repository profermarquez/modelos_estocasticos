import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Definir el símbolo de la acción y el período de tiempo
ticker = 'YPFD.BA'  # Acción de YPF en la Bolsa de Buenos Aires
start_date = '2023-01-01'
end_date = '2024-01-01'

# Descargar los datos de la acción
data = yf.download(ticker, start=start_date, end=end_date)

# Verificar si se descargaron datos correctamente
if data.empty:
    print(f"No se encontraron datos para {ticker}. Prueba otro ticker.")
    exit()

# Seleccionar la columna de precios correctamente
if 'Adj Close' in data.columns:
    data = data[['Adj Close']].rename(columns={'Adj Close': 'Precio'})
elif 'Close' in data.columns:
    data = data[['Close']].rename(columns={'Close': 'Precio'})
else:
    raise ValueError("No se encontró la columna 'Adj Close' o 'Close' en los datos descargados.")

# Asegurar que los datos tengan la misma cantidad de registros que la simulación
N = len(data)  # Definir N dinámicamente según los datos disponibles

# Calcular los rendimientos logarítmicos diarios
data['Log_Returns'] = np.log(data['Precio'] / data['Precio'].shift(1))

# Calcular mu y sigma
mu = data['Log_Returns'].mean()
sigma = data['Log_Returns'].std()

print(f"Media (mu): {mu}")
print(f"Desviación estándar (sigma): {sigma}")

# Simulación del Movimiento Browniano Geométrico
T = 1.0  # Tiempo en años
dt = T / N  # Ajustar N dinámicamente según los datos reales
S0 = data['Precio'].iloc[-1].item()  # ✅ Corregido para evitar FutureWarning

# Simulación de 10 trayectorias
num_simulations = 10
simulations = np.zeros((N, num_simulations))

for i in range(num_simulations):
    prices = np.zeros(N)  # Asegurar que el array tiene tamaño fijo
    prices[0] = S0  # Primer valor igual al último precio real
    for t in range(1, N):
        random_shock = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        prices[t] = prices[t-1] * np.exp(random_shock)
    simulations[:, i] = prices  # Asignar valores a la simulación

# Convertir a DataFrame con el mismo índice de los datos reales
simulations_df = pd.DataFrame(simulations, index=data.index)

# 🔹 **Solucionar error al calcular el error absoluto acumulado**
error_total = []
precio_real = data['Precio'].to_numpy()  # Convertir a NumPy array para operaciones eficientes

for i in range(num_simulations):
    simulacion_actual = simulations_df[i].to_numpy()  # Convertir simulación a NumPy array
    error = np.sum(np.abs(simulacion_actual - precio_real))  # Calcular error absoluto acumulado
    error_total.append((i, error))  # ✅ Asegurar que es una lista de tuplas

# Ordenar simulaciones por error
error_total = sorted(error_total, key=lambda x: x[1])  # ✅ Corrección del método sort()

# Seleccionar la mejor y la peor simulación
best_simulation = simulations_df[error_total[0][0]]  # Menor error
worst_simulation = simulations_df[error_total[-1][0]]  # Mayor error

# 🔹 **Graficar solo la mejor y peor simulación + datos reales**
plt.figure(figsize=(10, 5))  # Reducir el tamaño del gráfico

# Graficar los datos reales con una línea gruesa y color negro
plt.plot(data.index, data['Precio'], label='Datos Reales', color='black', linewidth=2)

# Graficar la mejor simulación en color azul
plt.plot(data.index, best_simulation, label='Mejor Simulación', color='blue', linestyle='dashed', linewidth=1.5)

# Graficar la peor simulación en color rojo
plt.plot(data.index, worst_simulation, label='Peor Simulación', color='red', linestyle='dashed', linewidth=1.5)

# Mejoras visuales
plt.title(f'Simulación del Movimiento Browniano Geométrico para {ticker}')
plt.xlabel('Días')
plt.ylabel('Precio Simulado')
plt.legend(loc='upper left', fontsize=10)  # Mostrar leyenda
plt.grid(True, linestyle='--', alpha=0.6)  # Agregar rejilla

# Mostrar gráfico
plt.show()
