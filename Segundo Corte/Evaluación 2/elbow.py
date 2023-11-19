import cv2
import math
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

plt.close('all')

# Lectura de la imagen
imagen=cv2.imread('img.png')
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

x1 = coordenadas_puntos[0]
x2 = coordenadas_puntos[1]
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

# Visualizing the data
plt.figure(0)
plt.plot()
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()

distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

for k in K:
	# Building and fitting the model
	kmeanModel = KMeans(n_clusters=k).fit(X)

	distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
										'euclidean'), axis=1)) / X.shape[0])
	inertias.append(kmeanModel.inertia_)

	mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
								'euclidean'), axis=1)) / X.shape[0]
	mapping2[k] = kmeanModel.inertia_
    
for key, val in mapping1.items():
	print(f'{key} : {val}')

plt.figure(1)
plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

for key, val in mapping2.items():
	print(f'{key} : {val}')

plt.figure(2)
plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.show()

# Calcular la variación relativa de las distancias
variations = []
for i in range(1, len(distortions)):
    variation = (distortions[i-1] - distortions[i]) / distortions[i-1]
    variations.append(variation)

# Encontrar el punto de inflexión
threshold = 0.1  # Umbral de variación relativa
elbow_index = np.where(np.array(variations) < threshold)[0][0]
ideal_k = K[elbow_index]
