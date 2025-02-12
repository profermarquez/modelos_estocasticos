import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Definir el símbolo de la acción y el período de tiempo
ticker = 'YPFD.BA'
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

# Calcular rendimientos logarítmicos diarios
data['Log_Returns'] = np.log(data['Precio'] / data['Precio'].shift(1))

# Parámetros del Modelo de Heston
kappa = 2.0  # Velocidad de reversion a la media de la volatilidad
theta = data['Log_Returns'].var()  # Nivel de volatilidad de largo plazo
xi = 0.1  # Volatilidad de la volatilidad
rho = -0.5  # Correlación entre el proceso de volatilidad y el precio

# Inicializar la volatilidad
data['Variance'] = data['Log_Returns'].rolling(window=30).var()
data.dropna(inplace=True)  # Eliminar valores NaN

# Parámetros de simulación
N = len(data)
T = 1.0
dt = T / N
S0 = data['Precio'].iloc[-1].item()
V0 = data['Variance'].iloc[-1]

# Convertir `mu` a array
mu_values = data['Log_Returns'].mean()

# Simulación con modelo de Heston
num_simulations = 10
simulations = np.zeros((N, num_simulations))

for i in range(num_simulations):
    prices = np.zeros(N)
    volatilities = np.zeros(N)
    prices[0] = S0
    volatilities[0] = V0
    
    for t in range(1, N):
        # Generar ruido gaussiano para precio y volatilidad
        Z1 = np.random.normal(0, 1)
        Z2 = np.random.normal(0, 1)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2  # Correlación entre ruido de precio y volatilidad

        # Modelo de Heston para la volatilidad
        volatilities[t] = np.abs(volatilities[t-1] + kappa * (theta - volatilities[t-1]) * dt + xi * np.sqrt(volatilities[t-1]) * np.sqrt(dt) * Z2)

        # Movimiento del precio con volatilidad estocástica
        prices[t] = prices[t-1] * np.exp((mu_values - 0.5 * volatilities[t]) * dt + np.sqrt(volatilities[t]) * np.sqrt(dt) * Z1)

    simulations[:, i] = prices

# Convertir a DataFrame
simulations_df = pd.DataFrame(simulations, index=data.index)

# Calcular el error absoluto acumulado entre cada simulación y los datos reales
error_total = []
precio_real = data['Precio'].to_numpy()

for i in range(num_simulations):
    simulacion_actual = simulations_df[i].to_numpy()
    error = np.sum(np.abs(simulacion_actual - precio_real))
    error_total.append((i, error))

# Ordenar simulaciones por error
error_total = sorted(error_total, key=lambda x: x[1])

# Seleccionar la mejor y la peor simulación
best_simulation = simulations_df[error_total[0][0]]
worst_simulation = simulations_df[error_total[-1][0]]

# Graficar
plt.figure(figsize=(10, 5))

plt.plot(data.index, data['Precio'], label='Datos Reales', color='black', linewidth=2)
plt.plot(data.index, best_simulation, label='Mejor Simulación', color='blue', linestyle='dashed', linewidth=1.5)
plt.plot(data.index, worst_simulation, label='Peor Simulación', color='red', linestyle='dashed', linewidth=1.5)

plt.title(f'Simulación con Modelo de Heston para {ticker}')
plt.xlabel('Días')
plt.ylabel('Precio Simulado')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
