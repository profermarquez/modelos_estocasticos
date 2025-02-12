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

# Drift (Media exponencial de los retornos)
data['mu'] = data['Log_Returns'].ewm(span=30).mean()

# Volatilidad (Ventana móvil)
data['sigma'] = data['Log_Returns'].rolling(window=30).std()

# Parámetros para el modelo de Merton (Jump-Diffusion)
jump_lambda = 0.1  # Probabilidad de un salto por día
jump_mu = 0.02     # Media de los saltos
jump_sigma = 0.05  # Volatilidad de los saltos

# Remover valores NaN
data.dropna(inplace=True)

# Ajustar N al número de registros reales después de limpiar NaN
N = len(data)

# Parámetros de simulación
T = 1.0
dt = T / N
S0 = data['Precio'].iloc[-1].item()

# Convertir `mu` y `sigma` a arrays para evitar problemas de indexación
mu_values = data['mu'].values
sigma_values = data['sigma'].values

# Simulación con drift, volatilidad adaptativa y saltos
num_simulations = 10
simulations = np.zeros((N, num_simulations))

for i in range(num_simulations):
    prices = np.zeros(N)
    prices[0] = S0
    for t in range(1, N):
        mu_t = mu_values[t]
        sigma_t = sigma_values[t]
        
        # Proceso de Wiener (Movimiento Browniano normal)
        normal_shock = np.random.normal(loc=mu_t * dt, scale=sigma_t * np.sqrt(dt))
        
        # Proceso de salto (Poisson)
        jump_occurred = np.random.rand() < jump_lambda
        jump_shock = jump_occurred * np.random.normal(loc=jump_mu, scale=jump_sigma)
        
        # Aplicar el cambio de precio con el proceso combinado
        prices[t] = prices[t-1] * np.exp(normal_shock + jump_shock)
    
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

plt.title(f'Simulación con Jump-Diffusion para {ticker}')
plt.xlabel('Días')
plt.ylabel('Precio Simulado')
plt.legend(loc='upper left', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

