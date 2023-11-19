import cv2
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import random

plt.close('all')

#Apertura de la imagen
imagen=cv2.imread("EXAMPLE2.bmp")
height, width, channels = imagen.shape
print(height,width,channels)

numeroPixeles  = height*width

cont = 0
contAux = 0

R = np.zeros(numeroPixeles)
G = np.zeros(numeroPixeles)
B = np.zeros(numeroPixeles)

for i in range(0,height,1):
    for j in range(0,width,1):
        R[contAux] = imagen[i][j][0]
        G[contAux] = imagen[i][j][1]
        B[contAux] = imagen[i][j][2]
        contAux += 1

#Número de clusters
K =  range(1,8)

distortions = []
inertias = []

for k in K:
    
    U = np.zeros((k,numeroPixeles))
    Um1 = np.zeros((k,numeroPixeles))

    for i in range(0,numeroPixeles,1):
        aux = random.randint(0,k-1)
        U[aux][i] = 1
        Um1[aux][i] = 1   
    
    cont = 0

    while(True):
        
        #Cálculo de los centroides
        centrosxyz = np.zeros((k,3))
        numx = 0
        denx = 0
        numy = 0
        deny = 0
        numz = 0
        denz = 0
        
        for i in range(0,k,1):
            for j in range(0,numeroPixeles,1):
                numx += U[i][j]*R[j]
                denx += U[i][j]
                numy += U[i][j]*G[j]
                deny += U[i][j]   
                numz += U[i][j]*B[j]
                denz += U[i][j]
            centrosxyz[i][0] = numx/denx
            centrosxyz[i][1] = numy/deny
            centrosxyz[i][2] = numz/denz
            
            numx = 0
            denx = 0
            numy = 0
            deny = 0
            numz = 0
            denz = 0
        
        # #Distancias entre los centroides y los datos
        
        distancias = np.zeros((k,numeroPixeles))

        for j in range(0,k,1):
            for i in range(0,numeroPixeles,1):
                distancias[j][i] = math.sqrt((R[i]-centrosxyz[j][0])**2 + (G[i]-centrosxyz[j][1])**2 + (B[i]-centrosxyz[j][2])**2)
        
        indices_min = np.argmin(distancias, axis=0)
        #Actualización de U
        for i in range(0,numeroPixeles,1):
            indice = indices_min[i]
            for j in range(0,k,1):
                if j == indice:
                    Um1[j][i] = 1   
                else:
                    Um1[j][i] = 0
        cont += 1
        if np.array_equal(U,Um1):
            break
        U = Um1
        #Final del while#
    distancia = 0
    pertenenciaU = np.argmax(U,axis=0)
    #Continuación del ciclo for k in K
    for i in range(0,numeroPixeles,1):
        indice = pertenenciaU[i]
        distancia += (R[i]-centrosxyz[indice][0])**2 + (G[i]-centrosxyz[indice][1])**2 + (B[i]-centrosxyz[indice][2])**2
    
    distancia = distancia/numeroPixeles   
    distortions.append(distancia)   




plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()


