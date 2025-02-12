# ejemplo de cómo obtener datos de Yahoo Finance con yfinance
import yfinance as yf

ticker = 'YPFD.BA'  # Ticker de YPF en la Bolsa de Buenos Aires
data = yf.download(ticker, start='2023-01-01', end='2024-01-01')

if data.empty:
    print(f"No se encontraron datos para {ticker}. Prueba otro ticker.")
else:
    print(data.head())  # Mostrar los primeros datos
