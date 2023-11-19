import cv2
import math
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random

plt.close('all')

# Lectura de la imagen
imagen=cv2.imread('img2.png')
height, width, channels = imagen.shape

# Cambio de tamaño (Resize)
resolucion_qvga = (200,120)
imagen_qvga = cv2.resize(imagen,resolucion_qvga)
# cv2.imshow('Imagen con tamaño modificado',imagen_qvga)


# Escala de grises
imagen_grises = cv2.cvtColor(imagen_qvga,cv2.COLOR_BGR2GRAY)
cv2.imshow('Imagen en escala de grises',imagen_grises)

# Binarización de la imagen
thresh = 250
_,img_binaria_bn = cv2.threshold(imagen_grises, thresh,255,cv2.THRESH_BINARY) # Imagen con fondo blanco
img_binaria = cv2.bitwise_not(img_binaria_bn) # Imagen con fondo negro
cv2.imshow('Imagen binarizada',img_binaria)


# Creating the data
canales = cv2.split(img_binaria) 
pixeles = canales[0]
coordenadas_puntos = np.where(pixeles == 255)
numero_puntos = len(coordenadas_puntos[0])

x1 = coordenadas_puntos[0]
x2 = coordenadas_puntos[1]


# Método del codo

X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

distorciones = []
K = range(1, 30)

for k in K:
	
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distorciones.append(sum(np.min(cdist(X, kmeans.cluster_centers_,
										'euclidean'), axis=1)) / X.shape[0])

# Calcular la variación relativa de las distancias
variaciones = []
for i in range(1, len(distorciones)):
    variacion = (distorciones[i-1] - distorciones[i]) / distorciones[i-1]
    variaciones.append(variacion)

# Encontrar el punto de inflexión
umbral = 0.1  # Umbral de variación relativa
elbow_index = np.where(np.array(variaciones) < umbral)[0][0]
k_ideal = K[elbow_index]
print(k_ideal)

# Método sustractivo

# Radios
ra = 12
rb = 12

# Densidad
D = np.zeros(numero_puntos)

xi = np.transpose(X)

for i in range(0,numero_puntos):
    for j in range(0,numero_puntos):
        distancia = math.sqrt( (xi[0][i] - xi[0][j])**2 + \
                              (xi[1][i] - xi[1][j])**2 )
        D[i] += math.exp( -distancia/(ra/2)**2 )
        
indice_maximo = np.argmax(D) 

# Maximos
MD = []
indices_maximos = []
MD.append(D[indice_maximo])
indices_maximos.append(indice_maximo)

# Restructuración de densidades
cont = 0
for k in range(1,k_ideal):
    
    for i in range(0,numero_puntos):
        distancia = math.sqrt( (xi[0][i] - xi[0][indice_maximo])**2 + \
                                  (xi[1][i] - xi[1][indice_maximo])**2 )
        D[i] = D[i] - MD[cont]*math.exp( -distancia/(rb/2)**2 )
    
    indice_maximo = np.argmax(D)
    MD.append(D[indice_maximo])
    indices_maximos.append(indice_maximo)
    cont += 1  

plt.figure(1)
plt.scatter(coordenadas_puntos[0], coordenadas_puntos[1], s = 10 ,color='blue',label = "Pixeles")
colores = ["#F796E1","#FF0000","#8E44AD","#BCF558","#17A589","#B8860B","#556B2F","#2F4F4F","#FFD700","#90EE90"]
for i in range(len(MD)):
    plt.scatter(xi[0][indices_maximos[i]],xi[1][indices_maximos[i]],s=500,color = colores[i],label=f'Objeto {i+1}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Grilla')
plt.grid(True)
plt.legend()
plt.show()

print("Valores máximos = {}".format(MD),\
      "Coeficientes Delta = {}".format(MD[0]/MD[:]))



