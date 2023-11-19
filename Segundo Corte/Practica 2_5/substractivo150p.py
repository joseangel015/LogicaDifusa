import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import random

plt.close('all')

#Generación de puntos aleatorios entre 0 y 10

xi = np.loadtxt('IrisDataBase.txt',usecols=(0,1,2)) # Definiendo variables 
# Reordenamiento del arreglo
xi = xi.reshape(3,len(xi))
numero_puntos = len(xi[0])

# Radios
ra = 10
rb = 5

# Densidad
D = np.zeros(numero_puntos)

for i in range(0,numero_puntos):
    for j in range(0,numero_puntos):
        distancia = math.sqrt( (xi[0][i] - xi[0][j])**2 + \
                              (xi[1][i] - xi[1][j])**2 + \
                                  (xi[2][i] - xi[2][j])**2 )
        D[i] += math.exp( -distancia/(ra/2)**2 )
       
indice_maximo = np.argmax(D) 

# Maximos
MD = []
indices_maximos = []
MD.append(D[indice_maximo])
indices_maximos.append(indice_maximo)


# Restructuración de densidades
cont = 0
umbral = 3
while MD[0]/MD[cont] < umbral:
    
    for i in range(0,numero_puntos):
        distancia = math.sqrt( (xi[0][i] - xi[0][indice_maximo])**2 + \
                                  (xi[1][i] - xi[1][indice_maximo])**2 + \
                                      (xi[2][i] - xi[2][indice_maximo])**2 )
        D[i] = D[i] - MD[cont]*math.exp( -distancia/(rb/2)**2 )
    
    indice_maximo = np.argmax(D)
    MD.append(D[indice_maximo])
    indices_maximos.append(indice_maximo)
    cont += 1       


fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')

colores = ["#F796E1","#FF0000","#8E44AD","#BCF558","#0000FF","#17A589"]
plt.figure(1)
ax1.scatter(xi[0], xi[1],xi[2], s=10,c=colores[5],label = 'Datos')

for i in range(len(MD)):
    ax1.scatter(xi[0][indices_maximos[i]],xi[1][indices_maximos[i]],xi[2][indices_maximos[i]],s=100,color = colores[i],label=f'Cluster {i+1}')

plt.title('Grilla')
plt.legend()
plt.show()

print("Valores máximos = {}".format(MD),\
      "Coeficientes Delta = {}".format(MD[0]/MD[:]))
