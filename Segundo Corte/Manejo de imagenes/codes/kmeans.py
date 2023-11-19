import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import math

plt.close('all')

# #Cambio de tamaño (Resize)
# imresize=cv2.resize(imagen,(0,0),fx=0.5,fy=0.5) #Dividir imagen
# cv2.imshow('Imagen con tamaño modificado',imresize)
# # imresize2=cv2.resize(imagen,(550,350))
# # cv2.imshow('Imagen con tamaño modificado',imresize2)

# #Escala de grises
# GrisImagen=cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
# cv2.imshow('Imagen en escala de grises',GrisImagen)

# #Imagen binarizada
# thresh =150
# img_binaria = cv2.threshold(GrisImagen, thresh,255,cv2.THRESH_BINARY)[1]
# invertida = cv2.bitwise_not(img_binaria)
# cv2.imshow('Imagen binarizada',img_binaria)
# cv2.imshow('Imagen invertida',invertida)


imagen=cv2.imread("pintura.jpg")
height, width, channels = imagen.shape
print(height,width,channels)

numeroPixeles  = height*width

cont = 0
contAux = 0
#Imagen = channels numero de matrices de height de alto por width de ancho

#Matriz de pertenencia
U = np.zeros((3,numeroPixeles))
Um1 = np.zeros((3,numeroPixeles))

R = np.zeros(numeroPixeles)
G = np.zeros(numeroPixeles)
B = np.zeros(numeroPixeles)

for i in range(0,height,1):
    for j in range(0,width,1):
        R[contAux] = imagen[i][j][0]
        G[contAux] = imagen[i][j][1]
        B[contAux] = imagen[i][j][2]
        contAux += 1
        
    
for i in range(0,numeroPixeles,1):
    aux = random.randint(0,2)
    U[aux][i] = 1
    Um1[aux][i] = 1



#Centros de clúster originales
v1o = [0,0,0]
v2o = [0,0,0]
v3o = [0,0,0]

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
    
    
    for i in range(0,numeroPixeles,1):
        numx += U[0][i]*R[i]
        denx += U[0][i]
        numy += U[0][i]*G[i]
        deny += U[0][i]
        numz += U[0][i]*B[i]
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
    
    
    for i in range(0,numeroPixeles,1):
        numx += U[1][i]*R[i]
        denx += U[1][i]
        numy += U[1][i]*G[i]
        deny += U[1][i]
        numz += U[1][i]*B[i]
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
    
    for i in range(0,numeroPixeles,1):
        numx += U[2][i]*R[i]
        denx += U[2][i]
        numy += U[2][i]*G[i]
        deny += U[2][i]
        numz += U[2][i]*B[i]
        denz += U[2][i]

    v31 = numx/denx
    v32 = numy/deny
    v33 = numz/denz
    
    v3 = [v31,v32,v33]
    
    
    #Distancias entre los centroides y los datos
    
    d1 = np.zeros(numeroPixeles)
    d2 = np.zeros(numeroPixeles)
    d3 = np.zeros(numeroPixeles)
    
    for i in range(0,numeroPixeles,1):
        #Cluster 1
        d1[i] = math.sqrt((R[i]-v1[0])**2 + (G[i]-v1[1])**2 + (B[i]-v1[2])**2)
        #Cluster 2
        d2[i] = math.sqrt((R[i]-v2[0])**2 + (G[i]-v2[1])**2 + (B[i]-v2[2])**2) 
        #Cluster 3
        d3[i] = math.sqrt((R[i]-v3[0])**2 + (G[i]-v3[1])**2 + (B[i]-v3[2])**2)
            
        
    #Actualización de U
    for i in range(0,numeroPixeles,1):
        aux = [d1[i],d2[i],d3[i]]
        valmin = np.min(aux)
        for j in range(0,3,1):
            if j == aux.index(valmin):
                Um1[j][i] = 1   
            else:
                Um1[j][i] = 0
    
    if cont == 0:
        v1o = v1
        v2o = v2
        v3o = v3
    cont += 1

    if np.array_equal(U,Um1):
        break
    U = Um1

imagenRGB =  np.zeros((height,width,channels))

v1 = [round(v1[0]),round(v1[1]),round(v1[2])]
v2 = [round(v2[0]),round(v2[1]),round(v2[2])]
v3 = [round(v3[0]),round(v3[1]),round(v3[2])]

contAux = 0

for i in range(0,height,1):
    for j in range(0,width,1):
        if U[0][contAux] == 1:
            imagenRGB[i][j][0] = v1[0]
            imagenRGB[i][j][1] = v1[1]
            imagenRGB[i][j][2] = v1[2]
        elif U[1][contAux] == 1:
            imagenRGB[i][j][0] = v2[0]
            imagenRGB[i][j][1] = v2[1]
            imagenRGB[i][j][2] = v2[2]
        elif U[2][contAux] == 1:
            imagenRGB[i][j][0] = v3[0]
            imagenRGB[i][j][1] = v3[1]
            imagenRGB[i][j][2] = v3[2]
        contAux += 1

cv2.imwrite("imagenRGB.jpg",imagenRGB)

# Creamos la figura
fig = plt.figure()
# Creamos el plano 3D
ax1 = fig.add_subplot(111, projection='3d')

coloresRGB = ["#FF0000","#00FF00","#0000FF","#EB7384","#0E7C1D","#56E5FF"]
marcadores = ['d','+','P']


for i in range(0,numeroPixeles,1):
    aux = [U[0][i],U[1][i],U[2][i]]
    indice = aux.index(1)
    ax1.scatter(R[i],G[i],B[i], c=coloresRGB[indice], marker=marcadores[indice]) 
        
ax1.scatter(v1[0], v1[1],v1[2], s=500, c=coloresRGB[3], marker='d', label='Centroide C1 final')
ax1.scatter(v2[0], v2[1],v2[2], s=500, c=coloresRGB[4], marker='+', label='Centroide C2 final')
ax1.scatter(v3[0], v3[1],v3[2], s=500, c=coloresRGB[5], marker='P', label='Centroide C3 final')

ax1.scatter(v1o[0], v1o[1],v1o[2], s=100, c='#000000', marker='d', label='Centroide C1 original')
ax1.scatter(v2o[0], v2o[1],v2o[2], s=100, c='#000000', marker='+', label='Centroide C2 original')
ax1.scatter(v3o[0], v3o[1],v3o[2], s=100, c='#000000', marker='P', label='Centroide C3 original')
plt.show()
