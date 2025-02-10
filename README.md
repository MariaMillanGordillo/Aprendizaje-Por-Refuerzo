# Aprendizaje por Refuerzo

## Descripción General
Este proyecto implementa un escenario para un robot manipulador en un entorno de Aprendizaje por Refuerzo (RL). Se han desarrollado distintos enfoques para resolver el problema de Pick&Place, utilizando algoritmos como Monte-Carlo con Exploring Starts, Q-Learning, Monte Carlo Tree Search (MCTS) y técnicas de aprendizaje profundo con Stable Baselines 3 (SB3).

## Implementación

### Ejercicio 1: Definición del Escenario
Se ha modelado un escenario Pick&Place en el que un robot manipulador cuenta con las siguientes acciones discretas:
- Movimientos en los ejes X, Y, Z.
- Apertura y cierre de la pinza.

El objetivo de la tarea es:
1. Cerrar la pinza en la pieza.
2. Transportar la pieza hasta un punto final (goal).
3. Abrir la pinza para soltar la pieza en el objetivo.

Se ha definido el estado del sistema y diseñado funciones de recompensa adecuadas para guiar el aprendizaje del agente.

---

### Ejercicio 2: Métodos de Aprendizaje por Refuerzo Clásicos
Se han implementado y analizado dos algoritmos:
1. **Monte-Carlo con Exploring Starts**
2. **Q-Learning**

Para cada uno:
- Se ha probado en un problema simple de Gymnasium (Frozen Lake) antes de aplicarlo al escenario del robot manipulador.
- Se ha analizado el desempeño usando distintas funciones de recompensa.
- Se ha descrito cada componente del algoritmo y se han analizado las funciones de valor obtenidas.
- Se han generado visualizaciones de los resultados.

---

### Ejercicio 3: Algoritmo Monte Carlo Tree Search (MCTS)
Se ha implementado el algoritmo **MCTS** y evaluado su rendimiento en:
1. Un problema simple de Gymnasium (Frozen Lake).
2. El escenario del robot manipulador, probando distintas funciones de recompensa y políticas de simulación:
   - Una aleatoria.
   - Una basada en minimizar la distancia al objetivo.

Se ha realizado un estudio de sensibilidad respecto a los parámetros clave, como:
- Constante de exploración.
- Número de iteraciones.

Se han incluido gráficas para validar el funcionamiento del algoritmo.

---

### Ejercicio 4: Entrenamiento con Stable Baselines 3 (SB3)
Se ha entrenado al agente utilizando **DQL (Deep Q-Learning)** y **PPO (Proximal Policy Optimization)** con SB3.

Pasos realizados:
1. **Verificación del entorno:** Se ha confirmado que el escenario cumple con la guía de diseño de Gymnasium (`check_env()`).
2. **Entrenamiento del agente** con DQL y PPO.

Recursos útiles:
- Documentación de SB3: https://stable-baselines3.readthedocs.io/en/master/

