import numpy as np 
import matplotlib.pyplot as plt
import math

U = [[0,0,1,1],[1,1,0,0]]
Um1 = [[0,0,1,1],[1,1,0,0]]
xi = [[1,4,4,5.5],[1,1,2,1]]
cont = 0
v1o = [0,0]
v2o = [0,0]
    
while(True):
    
    v11 = 0
    v12 = 0
    v21 = 0
    v22 = 0
    
    #Cálculo de los centroides 
    
    #Cluster 1
    numx = 0
    denx = 0
    numy = 0
    deny = 0
    
    
    for i in range(0,4,1):
        numx += U[0][i]*xi[0][i]
        denx += U[0][i]
        numy += U[0][i]*xi[1][i]
        deny += U[0][i]
    
    v11 = numx/denx
    v12 = numy/deny
    
    v1 = [v11,v12]
    
    #print(str(v11),str(v12))
    
    #Cluster 2
    numx = 0
    denx = 0
    numy = 0
    deny = 0
    
    
    for i in range(0,4,1):
        numx += U[1][i]*xi[0][i]
        denx += U[1][i]
        numy += U[1][i]*xi[1][i]
        deny += U[1][i]
        
        
    v21 = numx/denx
    v22 = numy/deny
    
    v2 = [v21,v22]
    
    
    #Distancias entre los centroides y los datos
    
    d1 = np.zeros(4)
    d2 = np.zeros(4)
    
    for i in range(0,4,1):
        #Cluster 1
        d1[i] = math.sqrt((xi[0][i]-v1[0])**2 + (xi[1][i]-v1[1])**2)
        #Cluster 2
        d2[i] = math.sqrt((xi[0][i]-v2[0])**2 + (xi[1][i]-v2[1])**2) 
        
    d = [d1,d2]   
    pertenencia_ik = 0
    m = 0.1
    #Actualización de U (normalización de las distancias)
    for i in range(0,2,1):
        for k in range(0,4,1):
            for j in range(0,2,1):
                pertenencia_ik += (d[i][k]/d[j][k])**(2/(m-1))
            Um1[i][k] = 1/pertenencia_ik  
            pertenencia_ik = 0
    print(Um1) 
    if cont == 0:
        v1o = v1
        v2o = v2
    cont += 1

    if Um1 == U:
        break
    U = Um1


#Graficación
plt.figure(1)
plt.plot(xi[0],xi[1],"o",label = "Muestras a agrupar")
plt.plot(v1o[0],v1o[1],"o",label = "Centroide C1 original")
plt.plot(v2o[0],v2o[1],"o",label = "Centroide C2 original")
plt.plot(v1[0],v1[1],"o",label = "Centroide C1 final")
plt.plot(v2[0],v2[1],"o",label = "Centroide C2 final")
plt.legend()
