import numpy as np 
import matplotlib.pyplot as plt
import math
import random

plt.close('all')

# Puntos a graficar
xi = np.loadtxt('IrisDataBase.txt',usecols=(0,1,2)) # Definiendo variables 
# Reordenamiento del arreglo
xi = xi.reshape(3,len(xi))

l1 = xi[0] # x
l2 = xi[1] # y
l3 = xi[2] # z

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
        centroxyz = np.zeros((k,3))
        numx = 0
        denx = 0
        numy = 0
        deny = 0
        numz = 0
        denz = 0
        
        for i in range(0,k,1):
            for j in range(0,numPuntos,1):
                numx += U[i][j]*xi[0][j]
                denx += U[i][j]
                numy += U[i][j]*xi[1][j]
                deny += U[i][j] 
                numz += U[i][j]*xi[2][j]
                denz += U[i][j]
            centroxyz[i][0] = numx/denx
            centroxyz[i][1] = numy/deny
            centroxyz[i][2] = numz/denz
            numx = 0
            denx = 0
            numy = 0
            deny = 0
            numz = 0
            denz = 0
        
        # #Distancias entre los centroides y los datos
        
        distancias = np.zeros((k,numPuntos))

        for j in range(0,k,1):
            for i in range(0,numPuntos,1):
                distancias[j][i] = math.sqrt((xi[0][i]-centroxyz[j][0])**2 + (xi[1][i]-centroxyz[j][1])**2 \
                                             + (xi[2][i]-centroxyz[j][2])**2 )
        
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
        distancia += (xi[0][i]-centroxyz[indice][0])**2 + (xi[1][i]-centroxyz[indice][1])**2 \
            + (xi[2][i]-centroxyz[indice][2])**2
    
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
