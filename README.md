# contexto
Trataremos de realizar una simulación del movimiento browniano geométrico (GBM) para modelar la evolución histórica de acciones argentinas utilizando datos reales en Python. 

Probamos tres modelos estocásticos y no se aproximan a los datos reales.

¿Por qué falla?
Los modelos estocásticos clásicos asumen una distribución normal o log-normal de los retornos, lo cual no siempre es cierto en mercados emergentes o acciones con alta volatilidad.
No consideran la memoria de los datos históricos, por lo que las simulaciones no siguen bien las tendencias recientes.
No capturan bien la estacionalidad y los patrones de mercado, que suelen estar presentes en los precios de las acciones.

# Instalacion y activacion del entorno virtual
/virtualenv env         /env/Scripts/activate.bat

# requerimientos
pip install pandas numpy matplotlib yfinance

pip install pandas_datareader

pip install investpy


# ejecutar
py mov_browniano_geo.py

py modelo_merton_jump_diff.py

py modelo_heston.py

