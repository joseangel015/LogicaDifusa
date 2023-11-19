%matplotlib qt

#Práctica 1.2b Red perceptron Ejercicio 1
import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import random
import math

plt.close("all")


#Matriz de pertenencia
U = np.zeros((3,150))
Um1 = np.zeros((3,150))

for i in range(0,150,1):
    aux = random.randint(0,2)
    U[aux][i] = 1
    Um1[aux][i] = 1

cont = 0


v1o = [0,0,0]
v2o = [0,0,0]
v3o = [0,0,0]

#Puntos a graficar
P=np.loadtxt('IrisDataBase.txt',usecols=(0,1,2)) # Definiendo variables 

while(True):
    
    v11 = 0
    v12 = 0
    v13 = 0
    v21 = 0
    v22 = 0
    v23 = 0
    v31 = 0
    v32 = 0
    v33 = 0
    
    #Cálculo de los centroides 
    
    #Cluster 1
    numx = 0
    denx = 0
    numy = 0
    deny = 0
    numz = 0
    denz = 0
    
    
    for i in range(0,150,1):
        numx += U[0][i]*P[i][0]
        denx += U[0][i]
        numy += U[0][i]*P[i][1]
        deny += U[0][i]
        numz += U[0][i]*P[i][2]
        denz += U[0][i]
    
    v11 = numx/denx
    v12 = numy/deny
    v13 = numz/denz
    
    v1 = [v11,v12,v13]
    
    
    #Cluster 2
    numx = 0
    denx = 0
    numy = 0
    deny = 0
    numz = 0
    denz = 0
    
    
    for i in range(0,150,1):
        numx += U[1][i]*P[i][0]
        denx += U[1][i]
        numy += U[1][i]*P[i][1]
        deny += U[1][i]
        numz += U[1][i]*P[i][2]
        denz += U[1][i]
    
    v21 = numx/denx
    v22 = numy/deny
    v23 = numz/denz
    
    v2 = [v21,v22,v23]
    
    
    #Cluster 3
    numx = 0
    denx = 0
    numy = 0
    deny = 0
    numz = 0
    denz = 0
    
    for i in range(0,150,1):
        numx += U[2][i]*P[i][0]
        denx += U[2][i]
        numy += U[2][i]*P[i][1]
        deny += U[2][i]
        numz += U[2][i]*P[i][2]
        denz += U[2][i]
    
    v31 = numx/denx
    v32 = numy/deny
    v33 = numz/denz
    
    v3 = [v31,v32,v33]
    
    if cont == 0:
        v1o = v1
        v2o = v2
        v3o = v3
    
    #Distancias entre los centroides y los datos
    
    d1 = np.zeros(150)
    d2 = np.zeros(150)
    d3 = np.zeros(150)
    
    for i in range(0,150,1):
        #Cluster 1
        d1[i] = math.sqrt((P[i][0]-v1[0])**2 + (P[i][1]-v1[1])**2 + (P[i][2]-v1[2])**2)
        #Cluster 2
        d2[i] = math.sqrt((P[i][0]-v2[0])**2 + (P[i][1]-v2[1])**2 + (P[i][2]-v2[2])**2) 
        #Cluster 3
        d3[i] = math.sqrt((P[i][0]-v3[0])**2 + (P[i][1]-v3[1])**2 + (P[i][2]-v3[2])**2) 
       
    #Actualización de U
    for i in range(0,150,1):
        aux = [d1[i],d2[i],d3[i]]
        valmin = np.min(aux)
        for j in range(0,3,1):
            if j == aux.index(valmin):
                Um1[j][i] = 1   
            else:
                Um1[j][i] = 0
        
    cont += 1

    if np.array_equal(U,Um1):
        break
    U = Um1


# # Creamos la figura
fig = plt.figure()
# # Creamos el plano 3D
ax1 = fig.add_subplot(111, projection='3d')

colores = ["#17A589","#D68910","#8E44AD"]
marcadores = ['d','+','P']

for i in range(0,150,1):
    aux = [U[0][i],U[1][i],U[2][i]]
    indice = aux.index(1)
    ax1.scatter(P[i][0], P[i][1],P[i][2], c=colores[indice], marker=marcadores[indice])
    
ax1.scatter(v1[0], v1[1],v1[2], c='r', marker=marcadores[0],label='Centroide C1 final')
ax1.scatter(v2[0], v2[1],v2[2], c='#FF00FF', marker=marcadores[1],label='Centroide C2 final')
ax1.scatter(v3[0], v3[1],v3[2], c='g', marker=marcadores[2],label='Centroide C3 final')

ax1.scatter(v1o[0], v1o[1],v1o[2], c='#000000', marker='d',label='Centroide C1 original')
ax1.scatter(v2o[0], v2o[1],v2o[2], c='#000000', marker='+',label='Centroide C2 original')
ax1.scatter(v3o[0], v3o[1],v3o[2], c='#000000', marker='P',label='Centroide C3 original')
plt.legend()
plt.show()
