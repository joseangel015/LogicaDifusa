import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import random

plt.close('all')

#Generación de puntos aleatorios entre 0 y 10
xi = np.zeros((2,30))
cont = 0
for i in range(0,2,1):
    for j in range(0,30,1):
        if i == 0:
            if cont < 10:
                xi[i][j] = random.randint(0,30)
            elif cont >= 10 and cont < 20:
                xi[i][j] = random.randint(50,80)
            else:
                xi[i][j] = random.randint(100,130)
        elif i == 1:
            if cont < 10 or cont > 20 :
                xi[i][j] = random.randint(80,100)
            else:
                xi[i][j] = random.randint(0,15)
        cont += 1
    cont = 0

l1 = xi[0] # x
l2 = xi[1] # y

#Grilla
xk = [l1,l2]
limites = [[min(l1), min(l2)],
           [max(l1), max(l2)]]

#Resolución de la grilla
resolucion = 10
num_clusters = resolucion**2

x = np.linspace(limites[0][0],limites[1][0],resolucion)
y = np.linspace(limites[0][1],limites[1][1],resolucion)
grid =  np.meshgrid(x,y)
X_grid, Y_grid = grid



# Función de montaña
M = np.zeros((resolucion,resolucion))
alpha = 0.1
beta = 0.1

# Condición de paro
delta = 2


# Obtención de la primera montaña
for i in range(len(X_grid)):
    for j in range(len(Y_grid)):
        for k in range(len(xk[0])):
            M[i][j] += math.e ** (-alpha*math.sqrt( (X_grid[0][i]-xk[0][k])**2 + (Y_grid[j][0]-xk[1][k])**2 ))

# Coordenadas del valor máximo (N1): plt
indices_maximo = np.argmax(M)
fila_maximo, columna_maximo = np.unravel_index(indices_maximo, M.shape)

# Valor máximo (pico de la montaña)
M1 = M[fila_maximo][columna_maximo]

# Valores máximos de los picos de montaña
valores_maximos = []
valores_maximos = np.append(valores_maximos, M1)

# Coordenadas del valor máximo dentro de la matriz M
coordenadas_maximos = np.zeros((1,2))
coordenada_maximo = [fila_maximo,columna_maximo]
coordenadas_maximos[0] = coordenada_maximo 

# Plot 3D de la primera montaña
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_zlabel('Z')
surf = ax.plot_surface(X_grid, Y_grid, M, cmap=cm.coolwarm,linewidth=0, antialiased=False)

# Cálculo de las siguientes montañas
cont = 0

while valores_maximos[0]/valores_maximos[cont] < delta: # Condición de paro
    
    if cont == 0:
        delta = 1.3

    for i in range(len(X_grid)):
        for j in range(len(Y_grid)):
            acumulable = 0
            for k in range(len(coordenadas_maximos)):
                coordenadaX = int(coordenadas_maximos[k][0])
                coordenadaY = int(coordenadas_maximos[k][1])
                # Distancia euclidiana
                d = math.sqrt((X_grid[0][coordenadaX]-X_grid[0][i])**2 + (Y_grid[coordenadaY][0]-Y_grid[j][0])**2 )
                acumulable += math.exp( -beta*d)
            M[i][j] = M[i][j] - valores_maximos[cont]*acumulable

    # Coordenadas del valor máximo (N1)
    indices_maximo = np.argmax(M)    
    fila_maximo, columna_maximo = np.unravel_index(indices_maximo, M.shape)
    
    # Valor máximo (pico de la montaña)
    M1 = M[fila_maximo][columna_maximo]
    valores_maximos = np.append(valores_maximos, M1)
    coordenada_maximo = [fila_maximo,columna_maximo]
    coordenadas_maximos = np.append(coordenadas_maximos,[coordenada_maximo],axis=0)
            

    # Plot 3D de la montaña
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    surf = ax.plot_surface(X_grid, Y_grid, M, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    
    cont += 1
    


# Plotear la grilla y puntos de datos
plt.figure(7)
plt.scatter(X_grid, Y_grid, s = 10 ,color='grey',label = "Nodos")
plt.scatter(l1,l2,color = 'green',label = "Datos")
colores = ["#F796E1","#FF0000","#8E44AD","#BCF558","#0000FF","#17A589"]
for i in range(len(coordenadas_maximos)):
    plt.scatter(X_grid[0][int(coordenadas_maximos[i][0])],Y_grid[int(coordenadas_maximos[i][1])][0], 
                s = 100, c = colores[i], label = "Centro de cluster {}".format(i+1))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Grilla')
plt.grid(True)
plt.legend()
plt.show()

# Coeficiente delta
print("Valores máximos = {}".format(valores_maximos),\
      "Coeficientes Delta = {}".format(valores_maximos[0]/valores_maximos[:]))


