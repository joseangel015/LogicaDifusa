import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# plt.close('all')
# Definir coordenadas de los datos
x1 = np.array([0.36, 0.85])
x2 = np.array([0.65, 0.89])
x3 = np.array([0.62, 0.55])
x4 = np.array([0.50, 0.75])
x5 = np.array([0.35, 1.00])
x6 = np.array([0.90, 0.35])
x7 = np.array([1.00, 0.24])
x8 = np.array([0.99, 0.55])
x9 = np.array([0.83, 0.36])
x10 = np.array([0.88, 0.43])

# Definir el tamaño de la cuadrícula
grid_size = 0.2

# Definir el parámetro alfa
alfa = 5.25

# Definir el número de clusters
n = 4
n = n - 1

# Definir el parámetro beta
beta = 1.5

# Crea matriz con los puntos
X = np.array([[x1[0], x2[0], x3[0], x4[0], x5[0], x6[0], x7[0], x8[0], x9[0], x10[0]],
              [x1[1], x2[1], x3[1], x4[1], x5[1], x6[1], x7[1], x8[1], x9[1], x10[1]]])

# Crear la cuadrícula
x_grid = np.arange(0, 1 + grid_size, grid_size)
y_grid = np.arange(0, 1 + grid_size, grid_size)

# Crear una matriz para almacenar los puntos de la cuadrícula
grid_points = []

# Generar todos los puntos en la cuadrícula
for x in x_grid:
    for y in y_grid:
        grid_points.append([x, y])

# Convertir la lista de puntos de la cuadrícula en un array de NumPy
grid_points = np.array(grid_points)

# Crear una lista para almacenar las pertenencias de la cuadrícula
grid_memberships = []

# Calcular la pertenencia para cada punto de la cuadrícula
for grid_point in grid_points:
    membership = 0.0
    for i in range(X.shape[1]):
        data_point = X[:, i]
        distance = np.linalg.norm(grid_point - data_point)
        membership += np.exp(-alfa * distance)
    grid_memberships.append(membership)

# Convertir la lista de pertenencias en un array de NumPy
grid_memberships = np.array(grid_memberships)
print(grid_memberships)

# Obtener el valor máximo del arreglo
max_membership = np.max(grid_memberships)
print("\nValor máximo del arreglo:", max_membership)

# Obtener el índice del valor máximo en grid_memberships
max_index = np.argmax(grid_memberships)

# Obtener la coordenada del punto de la cuadrícula correspondiente al índice máximo
centroid1 = grid_points[max_index]
print("Primer centroide:", centroid1)

# Calcular grid_size
grid_size = int(np.sqrt(len(grid_memberships)))

# Crear una figura y un eje 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Crear las coordenadas de la cuadrícula en los ejes x y y
grid_x, grid_y = np.meshgrid(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size))

# Graficar la montaña en 3D
ax.plot_surface(grid_x, grid_y, grid_memberships.reshape((grid_size, grid_size)), cmap='viridis')

# Etiquetar los ejes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Membership')

# Mostrar el gráfico
plt.show()

# Crear una lista para almacenar los centroides
centroids = [centroid1]

# Calcular los centroides adicionales
for _ in range(n):
    # Encuentra el índice del punto con la membresía máxima
    max_index = np.argmax(grid_memberships)

    # Obtiene las coordenadas del centroide actual
    centroid_current = grid_points[max_index]

    # Calcula las distancias a las coordenadas del centroide actual
    distances = np.linalg.norm(grid_points - centroid_current, axis=1)

    # Calcula las nuevas membresías utilizando la fórmula modificada
    new_memberships = np.maximum(grid_memberships - np.max(grid_memberships) * np.exp(-beta * distances), 0)

    # Encuentra el índice del punto con la nueva membresía máxima
    max_index = np.argmax(new_memberships)

    # Obtiene las coordenadas del centroide siguiente
    centroid_next = grid_points[max_index]

    # Agrega el centroide siguiente a la lista de centroides
    centroids.append(centroid_next)

    # Actualiza los valores de grid_memberships para el siguiente cálculo de centroide
    grid_memberships = new_memberships

    # Crear una figura y un subplot 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Graficar la montaña en 3D
    ax.plot_surface(grid_x, grid_y, new_memberships.reshape((grid_size, grid_size)), cmap='viridis')

    # Etiquetar los ejes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Membership')

    # Mostrar la figura
    plt.show()

    print(f"Centroide {_ + 2}:", centroid_next)

print("\nCentroides:")
for i, centroid in enumerate(centroids):
    print(f"Centroide {i + 1}: {centroid}")

# Convertir la lista de centroides en un array de NumPy
centroids = np.array(centroids)

plt.plot(X[0], X[1], "o", markersize=16, markeredgecolor="blue", markerfacecolor="lime", label="Puntos de entrada")
# Graficar los centroides finales en la gráfica inicial
plt.plot(centroids[:, 0], centroids[:, 1], "x", markersize=16, markeredgecolor="red", markeredgewidth=2, label="Centroides")

plt.grid()
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.legend()

# Mostrar la gráfica
plt.show()