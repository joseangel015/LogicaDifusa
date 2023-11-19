import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import cm

plt.close('all')

# Lectura de la imagen
imagen=cv2.imread('img3.jpg')
# cv2.imshow('Imagen a color',imagen)
height, width, channels = imagen.shape

# Cambio de tamaño (Resize)
resolucion_qvga = (320,240)
imagen_qvga = cv2.resize(imagen,resolucion_qvga)
# cv2.imshow('Imagen con tamaño modificado',imagen_qvga)

# Escala de grises
imagen_grises = cv2.cvtColor(imagen_qvga,cv2.COLOR_BGR2GRAY)
cv2.imshow('Imagen en escala de grises',imagen_grises)

# Binarización de la imagen
thresh = 250# Umbral
_,img_binaria_bn = cv2.threshold(imagen_grises, thresh,255,cv2.THRESH_BINARY) # Imagen con fondo blanco
img_binaria = cv2.bitwise_not(img_binaria_bn) # Imagen con fondo negro
cv2.imshow('Imagen invertida',img_binaria)


# Obtención de los pixeles de la imagen
canales = cv2.split(img_binaria) # Resolución de 240 filas (pixeles) X 320 columnas (pixeles)
pixeles = canales[0]
 
coordenadas_puntos = np.where(pixeles == 255)
plt.figure(1)
plt.scatter(coordenadas_puntos[0], coordenadas_puntos[1], s = 10 ,color='black',label = "Pixeles")
plt.xlabel('Ancho')
plt.ylabel('Alto')
plt.title('Pixeles de los objetos')
plt.legend()
plt.show()


# Utilización de la función de montaña para obtener el número de clusters (objetos)

l1 = coordenadas_puntos[0] # x
l2 = coordenadas_puntos[1] # y

#Grilla
xk = [l1,l2]
limites = [[min(l1), min(l2)],
           [max(l1), max(l2)]]

#Resolución de la grilla
resolucion = 10
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
        delta = 1.17

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
plt.figure(100)
plt.scatter(X_grid, Y_grid, s = 10 ,color='grey',label = "Nodos")
plt.scatter(l1,l2,color = 'green',label = "Datos")
colores = ["#F796E1","#FF0000","#8E44AD","#BCF558","#0000FF","#17A589","#B8860B","#556B2F","#2F4F4F","#FFD700","#90EE90"]
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
    
# Número de clusters 
print("Número de objetos sobre la mesa = {} objetos".format(len(valores_maximos)))

# Clusterización mediante K - means