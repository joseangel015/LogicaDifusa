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
resolucion_qvga = (320,240)
imagen_qvga = cv2.resize(imagen,resolucion_qvga)


# Escala de grises
imagen_grises = cv2.cvtColor(imagen_qvga,cv2.COLOR_BGR2GRAY)

# Binarización de la imagen
thresh = 250
_,img_binaria_bn = cv2.threshold(imagen_grises, thresh,255,cv2.THRESH_BINARY) # Imagen con fondo blanco
img_binaria = cv2.bitwise_not(img_binaria_bn) # Imagen con fondo negro


# Creating the data
canales = cv2.split(img_binaria) 
pixeles = canales[0]
coordenadas_puntos = np.where(pixeles == 255)
numero_puntos = len(coordenadas_puntos[0])

x1 = coordenadas_puntos[0]
x2 = coordenadas_puntos[1]


# Método del codo

X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

distortions = []
K = range(1, 30)

for k in K:
	
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeans.cluster_centers_,
										'euclidean'), axis=1)) / X.shape[0])

# Calcular la variación relativa de las distancias
variations = []
for i in range(1, len(distortions)):
    variation = (distortions[i-1] - distortions[i]) / distortions[i-1]
    variations.append(variation)

# Encontrar el punto de inflexión
threshold = 0.1  # Umbral de variación relativa
elbow_index = np.where(np.array(variations) < threshold)[0][0]
ideal_k = K[elbow_index]


# Clusterización por Kmeans

U = np.zeros((ideal_k,numero_puntos))
Um1 = np.zeros((ideal_k,numero_puntos))

for i in range(0,numero_puntos,1):
    aux = random.randint(0,ideal_k-1)
    U[aux][i] = 1
    Um1[aux][i] = 1  

cont = 0

while(True):
    
    #Cálculo de los centroides
    centrosxy = np.zeros((ideal_k,2))
    numx = 0
    denx = 0
    numy = 0
    deny = 0    

    for i in range(0,ideal_k,1):
        for j in range(0,numero_puntos,1):
            numx += U[i][j]*x1[j]
            denx += U[i][j]
            numy += U[i][j]*x2[j]
            deny += U[i][j]     
        centrosxy[i][0] = numx/denx
        centrosxy[i][1] = numy/deny
        numx = 0
        denx = 0
        numy = 0
        deny = 0

    # #Distancias entre los centroides y los datos
    distancias = np.zeros((ideal_k,numero_puntos))
    
    for j in range(0,ideal_k,1):
        for i in range(0,numero_puntos,1):
            distancias[j][i] = math.sqrt((x1[i]-centrosxy[j][0])**2 + (x2[i]-centrosxy[j][1])**2)
    
    indices_min = np.argmin(distancias, axis=0)
    
    #Actualización de U
    for i in range(0,numero_puntos,1):
        indice = indices_min[i]
        for j in range(0,ideal_k,1):
            if j == indice:
                Um1[j][i] = 1   
            else:
                Um1[j][i] = 0
    cont += 1
    if np.array_equal(U,Um1):
        break
    U = Um1
    
plt.figure(1)
plt.scatter(coordenadas_puntos[0], coordenadas_puntos[1], s = 10 ,color='black',label = "Pixeles")
colores = ["#F796E1","#FF0000","#8E44AD","#BCF558","#0000FF","#17A589","#B8860B","#556B2F","#2F4F4F","#FFD700","#90EE90"]
for i in range(ideal_k):
    plt.scatter(centrosxy[i][0],centrosxy[i][1], 
                s = 100, c = 'black', label = "Centro de cluster {}".format(i+1))
plt.xlabel('Ancho')
plt.ylabel('Alto')
plt.title('Pixeles de los objetos')
plt.legend()
plt.show()

print(f"Número de elementos identificados = {ideal_k}")