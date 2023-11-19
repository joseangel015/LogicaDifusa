import numpy as np 
import matplotlib.pyplot as plt
import math


plt.close('all')

# Puntos a graficar
xk = np.loadtxt('IrisDataBase.txt',usecols=(0,1,2)) # Definiendo variables 
# Reordenamiento del arreglo
xk = xk.reshape(3,len(xk))

l1 = xk[0] # x
l2 = xk[1] # y
l3 = xk[2] # z

#Grilla
limites = [[min(l1), min(l2),min(l3)],
            [max(l1), max(l2),max(l3)]]

#Resolución de la grilla
resolucion = 10
num_clusters = resolucion**3

x = np.linspace(limites[0][0],limites[1][0],resolucion)
y = np.linspace(limites[0][1],limites[1][1],resolucion)
z = np.linspace(limites[0][2],limites[1][2],resolucion)

grid =  np.meshgrid(x,y,z)
X_grid, Y_grid, Z_grid = grid

# Función de montaña
M = np.zeros((resolucion,resolucion,resolucion))
alpha = 0.379
beta = 0.5

# Condicion de paro
delta = 2


N = range(0,2) # Número de clústers

# Obtención de la primera montaña
for i in range(len(X_grid)):
    for j in range(len(Y_grid)):
        for k in range(len(Z_grid)):
            for l in range(len(xk[0])):
                M[i][j][k] += math.e ** \
                    (-alpha*math.sqrt( (X_grid[0][i][0]-xk[0][l])**2 \
                                      + (Y_grid[j][0][0]-xk[1][l])**2  \
                                      + (Z_grid[0][0][k]-xk[2][l])**2))


# Coordenadas del valor máximo (N1)
indices_maximo = np.argmax(M)
x_maximo, y_maximo, z_maximo = np.unravel_index(indices_maximo, M.shape)

# Valor máximo (pico de la montaña)
M1 = M[x_maximo][y_maximo][z_maximo]

# Valores máximos de los picos de montaña
valores_maximos = []
valores_maximos = np.append(valores_maximos, M1)

# Coordenadas del valor máximo dentro de la matriz M[10][10]
coordenadas_maximos = np.zeros((1,3))
coordenada_maximo = [x_maximo,y_maximo,z_maximo]
coordenadas_maximos[0] = coordenada_maximo 


# Cálculo de las siguientes montañas
cont = 0

while valores_maximos[0]/valores_maximos[cont] < delta: # Condición de paro
    
    if cont == 0:
        delta = 1.5
        
    for i in range(len(X_grid)):
        for j in range(len(Y_grid)):        
            for k in range(len(Z_grid)):
                acumulable = 0
                for l in range(len(coordenadas_maximos)):
                    coordenadaX = int(coordenadas_maximos[l][0])
                    coordenadaY = int(coordenadas_maximos[l][1])
                    coordenadaZ = int(coordenadas_maximos[l][2])
                    d = math.sqrt((X_grid[0][coordenadaX][0]-X_grid[0][i][0])**2 +\
                                  (Y_grid[coordenadaY][0][0]-Y_grid[j][0][0])**2 +\
                                  (Z_grid[0][0][coordenadaZ]-Z_grid[0][0][k])**2)
                    acumulable += math.exp( -beta*d)
                M[i][j][k] = M[i][j][k] - valores_maximos[cont]*acumulable

    indices_maximo = np.argmax(M)
    # Coordenadas del valor máximo (N1)
    x_maximo, y_maximo, z_maximo = np.unravel_index(indices_maximo, M.shape)
    
    # Valor máximo (pico de la montaña)
    M1 = M[x_maximo][y_maximo][z_maximo]
    valores_maximos = np.append(valores_maximos, M1)
    coordenada_maximo = [x_maximo,y_maximo,z_maximo]
    coordenadas_maximos = np.append(coordenadas_maximos,[coordenada_maximo],axis=0)
            
    cont += 1


# # Creamos la figura
fig = plt.figure()
# # Creamos el plano 3D
ax1 = fig.add_subplot(111, projection='3d')

colores = ["#F796E1","#FF0000","#8E44AD","#BCF558","#0000FF","#17A589"]
marcadores = ['d','+','P']

# Puntos de datos
ax1.scatter(xk[0], xk[1],xk[2], c=colores[2], marker=marcadores[1],label = 'Datos')

# Centros de cluster
coordenadas_maximos = coordenadas_maximos.T
ax1.scatter(coordenadas_maximos[0],coordenadas_maximos[1],coordenadas_maximos[2],s = 100, c = colores[1], marker = marcadores[0],label='Centros de cluster')
plt.grid(True)
plt.legend()
plt.show()

# Coeficiente delta
print("Valores máximos = {}".format(valores_maximos),\
      "Coeficientes Delta = {}".format(valores_maximos[0]/valores_maximos[:]))

