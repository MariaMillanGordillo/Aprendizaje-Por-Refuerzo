# Reorganized Code

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

import matplotlib.pyplot as plt

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

import os
from copy import deepcopy
from math import sqrt, log
from anytree import AnyNode

# ===========================================================================
# Reward Functions
# ===========================================================================

def calculate_reward_simple(env):
    """
    Calcula la recompensa basada únicamente en objetivos alcanzados y penalizaciones claras.
    Args:
        env (Environment): El entorno del robot (posiciones y estados).
    Returns:
        float: Recompensa acumulada según el estado actual.
    """
    reward = 0

    # Penalización por intentar recoger la pieza en una posición incorrecta
    if env.has_piece == 0 and env.action_last == 6 and (
        env.robot_x_position != env.piece_x_position or
        env.robot_y_position != env.piece_y_position or
        env.robot_z_position != env.piece_z_position
    ):
        reward -= 50

    # Recompensa por recoger la pieza (solo una vez)
    if env.has_piece == 1 and env.piece_x_position == -1:
        reward += 100

    # Recompensa por llevar la pieza al objetivo
    if env.has_piece == 1:
        aligned_axes = sum([
            env.robot_x_position == env.goal_x_position,
            env.robot_y_position == env.goal_y_position,
            env.robot_z_position == env.goal_z_position
        ])
        reward += 50 * aligned_axes

    # Finalización del objetivo
    if env.has_piece == 0 and (
        env.piece_x_position == env.goal_x_position and
        env.piece_y_position == env.goal_y_position and
        env.piece_z_position == env.goal_z_position
    ):
        reward += 5000

    return reward

def calculate_reward_distance(env):
    """
    Calcula la recompensa únicamente en función de las distancias al objetivo y la pieza.

    Args:
        env (Environment): El entorno que contiene el estado actual del robot. 

    Returns:
        float: La recompensa calculada para el estado actual del entorno.
    """
    reward = 0

    # Calcular la distancia al objeto
    distance_to_piece = ((env.robot_x_position - env.piece_x_position) ** 2 +
                         (env.robot_y_position - env.piece_y_position) ** 2 +
                         (env.robot_z_position - env.piece_z_position) ** 2) ** 0.5

    # Calcular la distancia al objetivo
    distance_to_goal = ((env.robot_x_position - env.goal_x_position) ** 2 +
                        (env.robot_y_position - env.goal_y_position) ** 2 +
                        (env.robot_z_position - env.goal_z_position) ** 2) ** 0.5

    if env.has_piece == 0:
        # Recompensa basada en la cercanía al objeto
        reward += max(0, 100 - distance_to_piece * 10)
    else:
        # Recompensa basada en la cercanía al objetivo
        reward += max(0, 500 - distance_to_goal * 10)

        # Recompensa adicional por reducir la distancia al objetivo
        if hasattr(env, 'previous_distance_to_goal'):
            if distance_to_goal < env.previous_distance_to_goal:
                reward += 50
        env.previous_distance_to_goal = distance_to_goal

    piece_distance_to_goal = ((env.piece_x_position - env.goal_x_position) ** 2 +
                              (env.piece_y_position - env.goal_y_position) ** 2 +
                              (env.piece_z_position - env.goal_z_position) ** 2) ** 0.5

    # Recompensa por llevar la pieza al objetivo
    if piece_distance_to_goal == 0 and env.has_piece == 0:
        reward += 100000

    return reward

def calculate_reward_mixed(env):
    """
    Calcula la recompensa mixta basada en objetivos alcanzados y distancia al objetivo. Añade penalizaciones por movimientos redundantes y bucles.

    Args:
        env (Environment): El entorno que contiene el estado actual del robot.

    Returns:
        float: La recompensa calculada para el estado actual del entorno.
    """
    reward = 0 
    env.previous_distance_to_goal = ((env.robot_x_position - env.goal_x_position) ** 2 +
                            (env.robot_y_position - env.goal_y_position) ** 2 +
                            (env.robot_z_position - env.goal_z_position) ** 2) ** 0.5

    if env.has_piece == 0:
        # Recompensa proporcional a la cercanía al objeto.
        distance_to_piece = ((env.robot_x_position - env.piece_x_position) ** 2 +
                             (env.robot_y_position - env.piece_y_position) ** 2 +
                             (env.robot_z_position - env.piece_z_position) ** 2) ** 0.5
        reward += max(0, 50 - distance_to_piece * 2)
    else:
        # Recompensa proporcional a la cercanía al objetivo.
        distance_to_goal = ((env.robot_x_position - env.goal_x_position) ** 2 +
                            (env.robot_y_position - env.goal_y_position) ** 2 +
                            (env.robot_z_position - env.goal_z_position) ** 2) ** 0.5
        reward += max(0, 200 - distance_to_goal * 2)

        # Penalización y recompensa acumulativa basada en la distancia al objetivo.
        if hasattr(env, 'previous_distance_to_goal'):
            if distance_to_goal > env.previous_distance_to_goal:
                reward -= 5000
            else:
                reward += 10000
        env.previous_distance_to_goal = distance_to_goal

    # Recompensa por llegar a la pieza.
    if env.piece_x_position == env.robot_x_position and env.piece_y_position == env.robot_y_position and env.piece_z_position == env.robot_z_position:
        reward += 5000

    # Recompensa por recoger la pieza (solo una vez).
    if env.has_piece == 1 and not hasattr(env, 'piece_collected'):
        reward += 100
        setattr(env, 'piece_collected', True)

    # Recompensa constante por mantener la pieza.
    if env.has_piece == 1:
        reward += 10000

    # Recompensa por llegar al objetivo con la pieza.
    if env.piece_x_position == env.robot_x_position == env.goal_x_position and env.piece_y_position == env.robot_y_position == env.goal_y_position and env.piece_z_position == env.robot_z_position == env.goal_z_position:
        reward += 500000

    # Recompensa por alcanzar el objetivo.
    if (env.piece_x_position == env.goal_x_position and
        env.piece_y_position == env.goal_y_position and
        env.piece_z_position == env.goal_z_position and
        not hasattr(env, 'goal_reached')):
        reward += 1000000
        setattr(env, 'goal_reached', True)

    # Penalización por soltar la pieza en una posición incorrecta.
    if env.has_piece == 0 and not hasattr(env, 'piece_placed') and (
        env.piece_x_position != env.goal_x_position or
        env.piece_y_position != env.goal_y_position or
        env.piece_z_position != env.goal_z_position
    ):
        reward -= 10000
        setattr(env, 'piece_placed', True)

    # Penalización por movimientos redundantes.
    if hasattr(env, 'visited_positions'):
        current_position = (env.robot_x_position, env.robot_y_position, env.robot_z_position)
        if current_position in env.visited_positions:
            reward -= 50
        else:
            env.visited_positions.add(current_position)
    else:
        env.visited_positions = set([(env.robot_x_position, env.robot_y_position, env.robot_z_position)])

    # Penalización por entrar en bucles.
    if hasattr(env, 'recent_positions'):
        current_position = (env.robot_x_position, env.robot_y_position, env.robot_z_position)
        env.recent_positions.append(current_position)
        # Limitar el historial a las últimas 10 posiciones.
        if len(env.recent_positions) > 10:
            env.recent_positions.pop(0)
        # Penaliar bucles.
        if len(env.recent_positions) > len(set(env.recent_positions)):
            reward -= 20000
    else:
        env.recent_positions = [(env.robot_x_position, env.robot_y_position, env.robot_z_position)]

    return reward

# ===========================================================================
# Custom Environment Definition
# ===========================================================================
class RobotPickAndPlaceEnv(gym.Env):
    """
    Representa el entorno del robot para resolver el problema de pick & place.

    Este entorno simula un espacio discreto en 3D donde un robot debe recoger una
    pieza en un punto inicial y transportarla a un objetivo. El robot puede moverse
    en las direcciones X, Y y Z, además de abrir o cerrar su pinza para coger y soltar la pieza.

    Attributes:
        robot_position: Posición actual del robot en (x, y, z).
        piece_position: Posición actual de la pieza en (x, y, z). (-1, -1, -1 si está siendo transportada).
        goal_position: Posición del objetivo en (x, y, z).
        has_piece (bool): Indica si el robot tiene la pieza agarrada (1 si tiene la pieza, 0 en caso contrario).
        step_limit (int): Número máximo de pasos permitidos.
        action_space (gymnasium.spaces.Discrete): Espacio de acciones (0-6).
        observation_space (gymnasium.spaces.Box): Espacio de observaciones del entorno.
        reward_function: Función de recompensa utilizada para evaluar los pasos del robot.
    """
    def __init__(self, reward_function=calculate_reward_simple):
        super(RobotPickAndPlaceEnv, self).__init__()
        # Definir espacio de acciones
        self.action_space = spaces.Discrete(7)

        # Definir espacio de observaciones como un Box
        # [robot_x, robot_y, robot_z, has_piece, piece_x, piece_y, piece_z]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -1, -1, -1]),
            high=np.array([9, 9, 9, 1, 9, 9, 9]),
            dtype=np.int32
        )

        # Establecer función de recompensa
        self.reward_function = reward_function

        # Inicializar estado
        self.reset()

        # Inicializar figura, eje y gráfica de matplotlib
        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Condiciones iniciales
        self.robot_x_position = 0
        self.robot_y_position = 0
        self.robot_z_position = 0
        self.has_piece = 0
        self.piece_x_position = 5
        self.piece_y_position = 5
        self.piece_z_position = 5
        self.goal_x_position = 9
        self.goal_y_position = 7
        self.goal_z_position = 9
        self.steps = 0

        # Atributos para recompensas
        self.action_last = None
        self.visited_positions = set()
        self.previous_distance_to_goal = None
        self.recent_positions = []

        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        assert self.action_space.contains(action), "Acción inválida"
        done = False
        truncated = False
        self.action_last = action  # Registrar la última acción para el cálculo de recompensa

        # Actualizar la posición del robot con lógica determinista
        if action == 0:  # Mover izquierda
            self.robot_x_position = max(0, self.robot_x_position - 1)
        elif action == 1:  # Mover derecha
            self.robot_x_position = min(9, self.robot_x_position + 1)
        elif action == 2:  # Mover abajo
            self.robot_y_position = max(0, self.robot_y_position - 1)
        elif action == 3:  # Mover arriba
            self.robot_y_position = min(9, self.robot_y_position + 1)
        elif action == 4:  # Mover atrás
            self.robot_z_position = max(0, self.robot_z_position - 1)
        elif action == 5:  # Mover adelante
            self.robot_z_position = min(9, self.robot_z_position + 1)
        elif action == 6:  # Abrir/cerrar pinza
            if self.has_piece == 0 and (
                self.robot_x_position == self.piece_x_position and
                self.robot_y_position == self.piece_y_position and
                self.robot_z_position == self.piece_z_position
            ):
                self.has_piece = 1
                self.piece_x_position = -1
                self.piece_y_position = -1
                self.piece_z_position = -1
            elif self.has_piece == 1:
                self.has_piece = 0
                self.piece_x_position = self.robot_x_position
                self.piece_y_position = self.robot_y_position
                self.piece_z_position = self.robot_z_position

        # Verificar si el objetivo se ha alcanzado
        if (
            self.piece_x_position == self.goal_x_position and
            self.piece_y_position == self.goal_y_position and
            self.piece_z_position == self.goal_z_position
        ):
            done = True

        # Calcular recompensa
        reward = self.reward_function(self)
        self.steps += 1

        # Verificar límite de pasos
        if self.steps >= 1000:
            truncated = True

        return self._get_observation(), reward, done, truncated, {}

    def render(self):
        """
        Renderiza el estado actual del entorno en una gráfica 3D.
        """
        # Inicializar figura y ejes si no existen
        if self.fig is None or self.ax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection="3d")

            # Inicializar gráficos del robot, la pieza y el objetivo
            self.robot_plot, = self.ax.plot([], [], [], "go", label="Robot", markersize=7)
            self.goal_plot, = self.ax.plot([], [], [], "ro", label="Objetivo", markersize=8)
            self.piece_plot, = self.ax.plot([], [], [], "bo", label="Pieza", markersize=6)

            # Configuración de los límites del eje
            self.ax.set_xlim([0, 10])
            self.ax.set_ylim([0, 10])
            self.ax.set_zlim([0, 10])

            # Etiquetas y leyenda
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
            self.ax.legend()

        # Actualizar posición del robot
        self.robot_plot.set_data([self.robot_x_position], [self.robot_y_position])
        self.robot_plot.set_3d_properties([self.robot_z_position])

        # Actualizar posición de la pieza
        if self.has_piece:
            self.piece_plot.set_data([self.robot_x_position], [self.robot_y_position])
            self.piece_plot.set_3d_properties([self.robot_z_position])
        else:
            self.piece_plot.set_data([self.piece_x_position], [self.piece_y_position])
            self.piece_plot.set_3d_properties([self.piece_z_position])

        # Actualizar posición del objetivo
        self.goal_plot.set_data([self.goal_x_position], [self.goal_y_position])
        self.goal_plot.set_3d_properties([self.goal_z_position])

        # Actualizar el título con el número de pasos
        self.ax.set_title(f"Paso: {self.steps} | Posición Robot: (" +
                          f"{self.robot_x_position}, {self.robot_y_position}, {self.robot_z_position}) " +
                          f"| Posición Pieza: ({self.piece_x_position}, {self.piece_y_position}, {self.piece_z_position})")

        # Dibujar y pausar para visualización en tiempo real
        plt.draw()
        plt.pause(0.01)

    def _get_observation(self):
        return np.array([
            self.robot_x_position,
            self.robot_y_position,
            self.robot_z_position,
            self.has_piece,
            self.piece_x_position,
            self.piece_y_position,
            self.piece_z_position
        ], dtype=np.int32)

    def close(self):
        """
        Cierra la visualización del entorno.
        """
        plt.close()

# ===========================================================================
# Configuración de Callbacks para DQN y PPO
# ===========================================================================

def setup_callbacks(log_dir, best_model_dir, eval_freq=10000, n_eval_episodes=5):
    """Configura el callback de evaluación para ambos algoritmos."""
    # Crear entorno de evaluación separado
    eval_env = Monitor(RobotPickAndPlaceEnv(reward_function=calculate_reward_mixed))
    
    return EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        verbose=1,
        deterministic=True
    )

def plot_evaluations(log_dir, title):
    """
    Grafica los resultados de las evaluaciones almacenadas en evaluations.npz
    """
    eval_data = np.load(os.path.join(log_dir, "evaluations.npz"))
    
    # Calcular la media de recompensas por evaluación
    timesteps = eval_data["timesteps"]
    mean_rewards = eval_data["results"].mean(axis=1).flatten()
    
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, mean_rewards, marker="o", linestyle="-", color="#2c3e50")
    plt.title(f"Evaluaciones - {title}", fontsize=14, fontweight="bold")
    plt.xlabel("Pasos", fontsize=12)
    plt.ylabel("Recompensa Media", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "eval_results.png"))
    plt.close()


# ===========================================================================
# Entrenamiento Integrado con Callbacks
# ===========================================================================

if __name__ == "__main__":
    # Crear entorno principal
    env = RobotPickAndPlaceEnv(reward_function=calculate_reward_mixed)
    check_env(env)
    
    # Configurar callbacks para ambos algoritmos
    dqn_callback = setup_callbacks(
        "Ejercicio 4/dqn_eval_logs/",
        "Ejercicio 4/dqn_best_model/",
        eval_freq=5000
    )
    
    ppo_callback = setup_callbacks(
        "Ejercicio 4/ppo_eval_logs/",
        "Ejercicio 4/ppo_best_model/",
        eval_freq=10000
    )

    # Entrenamiento DQN con callback
    dqn_model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=100000,
        batch_size=256,
        gamma=0.999,
        tensorboard_log="Ejercicio 4/dqn_tensorboard/"
    )
    dqn_model.learn(total_timesteps=100000, callback=dqn_callback)
    dqn_model.save("Ejercicio 4/dqn_final_model")

    # Entrenamiento PPO con callback
    ppo_model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="Ejercicio 4/ppo_tensorboard/"
    )
    ppo_model.learn(total_timesteps=200000, callback=ppo_callback)
    ppo_model.save("Ejercicio 4/ppo_final_model")

    # Visualización de resultados
    for model in [dqn_model, ppo_model]:
        print(f"\nEvaluación {type(model).__name__}:")
        obs, _ = env.reset()
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            env.render()
            if done or truncated:
                obs, _ = env.reset()

    # Cierre correcto
    env.close()
    plt.close('all')

    # Uso:
    plot_evaluations("Ejercicio 4/dqn_eval_logs/", "Desempeño DQN")
    plot_evaluations("Ejercicio 4/ppo_eval_logs/", "Desempeño PPO")

# Obtenemos resultados muy pobres. Con DQN obtenemos valores de recompensa media más altos, mientras que con PPO son más constantes.
# Sin embargo, en ambos casos, el robot no logra completar el objetivo de llevar la pieza al objetivo. Esto puede deberse a la complejidad del entorno y la dificultad de encontrar una política óptima.
# Por falta de tiempo, no he podido probar con otros hiperparámetros o configuraciones del modelo que mejoren considerablemente el rendimiento.