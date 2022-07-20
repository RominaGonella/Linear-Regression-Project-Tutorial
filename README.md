# Resumen

1. Primero se trabajó en archivo explore.ipynb para realizar el EDA y probar distintos modelos.
2. Se creó una variable nueva que agrupa la cantidad de hijos en tres categorías: 0, 1 y 2 o más.
3. Del EDA surge que la correlación entre las variables es baja, por lo que en principio se consideran todas las variables para el modelo.
4. Se estimó un primer modelo de regresión lineal con todas las variables disponibles y parámetros por defecto.
5. En el paso 4 no se realizó búsqueda de mejores hiperparámetros, ya que la funcion no ofrece muchas opciones, en su lugar se realizó selección de variables, identificando 4 variables como las más importantes: age, bmi, smoker y children_gr. Esta selección es consistente con los hallazgos del EDA.
6. Se guardó el modelo elegido en la carpeta models.
7. Se creó un pipeline en app.py cargando solamente el código imprescindible para crear y guardar el modelo elegido.