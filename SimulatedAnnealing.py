import numpy as np
import matplotlib.pyplot as plt


# Función de Rosenbrock
def rosenbrock(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x**2) ** 2


# Función para generar una nueva solución vecina (Simulated Annealing)
def neighbor_solution(point, step_size):
    return point + np.random.uniform(-step_size, step_size, point.shape)


# Criterio de Boltzmann para aceptar una nueva solución
def accept_solution(delta, temperature):
    if delta < 0:
        return True
    else:
        return np.random.rand() < np.exp(-delta / temperature)


# Parámetros iniciales
np.random.seed(0)
initial_point = np.random.rand(2) * 2 - 1  # Punto inicial aleatorio (en [-1, 1])
initial_temperature = 1.0  # Temperatura inicial
cooling_rate = 0.99  # Tasa de enfriamiento
step_size = 0.05  # Tamaño de cada paso para generar vecinos
max_iterations = 10000  # Número máximo de iteraciones
min_temperature = 1e-3  # Temperatura mínima para detener el algoritmo

# Simulated Annealing
current_point = initial_point
current_value = rosenbrock(current_point[0], current_point[1])
temperature = initial_temperature
points_history = [current_point]

for iteration in range(max_iterations):
    new_point = neighbor_solution(current_point, step_size)
    new_value = rosenbrock(new_point[0], new_point[1])

    # Se acepta la nueva solución si es mejor
    if accept_solution(new_value - current_value, temperature):
        current_point, current_value = new_point, new_value
        points_history.append(current_point)

    # Se reduce la temperatura
    temperature *= cooling_rate

    # Condición de finalización por temperatura mínima
    if temperature < min_temperature:
        break

# Grafica función de Rosenbrock y el punto mínimo
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=np.logspace(-0.5, 3.5, 20), cmap="viridis")
plt.plot(
    *zip(*points_history),
    marker="o",
    color="r",
    markersize=3,
    linestyle="-",
    linewidth=1,
    label="Camino al mínimo",
)
plt.plot(
    current_point[0],
    current_point[1],
    "bo",
    label=f"Punto mínimo: ({current_point[0]:.4f}, {current_point[1]:.4f})",
)
plt.title("Camino al mínimo de la función de Rosenbrock")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Resultados
print("Punto final:", current_point)
print("Valor en el punto final:", current_value)
