import numpy as np 
import matplotlib.pyplot as plt
import math
import random

plt.close('all')

# Puntos de datos
xi = np.zeros((2,30))
cont = 0

l1 = [0.36, 0.65, 0.62, 0.5, 0.35, 0.9, 1, 0.99, 0.83, 0.88] # x
l2 = [0.85, 0.89, 0.55, 0.75, 1, 0.35, 0.24, 0.55, 0.36, 0.43] # y

xi = [l1,l2]
numPuntos = len(xi[0])

#Número de clusters
K =  range(1,8)

distortions = []
inertias = []

for k in K:
    
    U = np.zeros((k,numPuntos))
    Um1 = np.zeros((k,numPuntos))

    for i in range(0,numPuntos,1):
        aux = random.randint(0,k-1)
        U[aux][i] = 1
        Um1[aux][i] = 1   
    
    cont = 0

    while(True):
        
        #Cálculo de los centroides
        centrosxy = np.zeros((k,2))
        numx = 0
        denx = 0
        numy = 0
        deny = 0
        
        for i in range(0,k,1):
            for j in range(0,numPuntos,1):
                numx += U[i][j]*xi[0][j]
                denx += U[i][j]
                numy += U[i][j]*xi[1][j]
                deny += U[i][j]     
            centrosxy[i][0] = numx/denx
            centrosxy[i][1] = numy/deny
            numx = 0
            denx = 0
            numy = 0
            deny = 0
        
        # #Distancias entre los centroides y los datos
        
        distancias = np.zeros((k,numPuntos))

        for j in range(0,k,1):
            for i in range(0,numPuntos,1):
                distancias[j][i] = math.sqrt((xi[0][i]-centrosxy[j][0])**2 + (xi[1][i]-centrosxy[j][1])**2)
        
        indices_min = np.argmin(distancias, axis=0)
        #Actualización de U
        for i in range(0,numPuntos,1):
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
    for i in range(0,numPuntos,1):
        indice = pertenenciaU[i]
        distancia += (xi[0][i]-centrosxy[indice][0])**2 + (xi[1][i]-centrosxy[indice][1])**2
    
    inertias.append(distancia)
    distancia = distancia/numPuntos    
    distortions.append(distancia)   



plt.figure(1)
plt.plot(K, distortions, 'bx-')
plt.xlabel('Valores de K')
plt.ylabel('Distorsión')
plt.title('Método del codo usando Distorción')
plt.show()

plt.figure(2)
plt.plot(K, inertias, 'bx-')
plt.xlabel('Valores de K')
plt.ylabel('Inercia')
plt.title('Método del codo usando Inercia')
plt.show()
