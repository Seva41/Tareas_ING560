import numpy as np
import matplotlib.pyplot as plt


# Definición de la función de Rosenbrock
def rosenbrock(x, y, a=1, b=100):
    return (a - x) ** 2 + b * (y - x**2) ** 2


# Gradiente de la función de Rosenbrock
def rosenbrock_gradient(x, y, a=1, b=100):
    grad_x = -2 * (a - x) - 4 * b * x * (y - x**2)
    grad_y = 2 * b * (y - x**2)
    return np.array([grad_x, grad_y])


# Definición de los parámetros iniciales
np.random.seed(0)
initial_point = np.random.rand(2) * 2 - 1  # Punto inicial aleatorio en [-1, 1]
learning_rate = 0.002
iterations = 1000000
threshold = 1e-6

# Bucle de optimización
point = initial_point
points_history = [point]  # Historial de puntos para graficar
for i in range(iterations):
    grad = rosenbrock_gradient(point[0], point[1])
    new_point = point - learning_rate * grad
    points_history.append(new_point)
    if np.linalg.norm(new_point - point) < threshold:
        break
    point = new_point

# Graficar la función de Rosenbrock y el punto mínimo encontrado
x = np.linspace(-1, 2, 400)
y = np.linspace(-1, 2, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=np.logspace(-0.5, 3.5, 20), cmap="viridis")
plt.plot(
    *zip(*points_history), marker="o", color="r", markersize=5, label="Camino al mínimo"
)
final_point = points_history[-1]
plt.plot(
    points_history[-1][0],
    points_history[-1][1],
    "bo",
    label=f"Punto mínimo: ({final_point[0]:.4f}, {final_point[1]:.4f})",
)  # Punto final en azul
plt.title("Camino al mínimo de la función de Rosenbrock")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Muestra de resultados
print("Punto final:", point)
print("Valor en el punto final:", rosenbrock(point[0], point[1]))
