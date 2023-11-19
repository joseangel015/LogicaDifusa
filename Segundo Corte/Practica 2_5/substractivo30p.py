import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import random

plt.close('all')

#Generación de puntos aleatorios entre 0 y 10
numero_puntos = 30
xi = np.zeros((2,numero_puntos))
cont = 0
for i in range(0,2,1):
    for j in range(0,numero_puntos,1):
        if i == 0:
            if cont < 10:
                xi[i][j] = random.randint(0,30)
            elif cont >= 10 and cont < 20:
                xi[i][j] = random.randint(50,80)
            else:
                xi[i][j] = random.randint(100,130)
        elif i == 1:
            if cont < 10 or cont > 20 :
                xi[i][j] = random.randint(80,100)
            else:
                xi[i][j] = random.randint(0,15)
        cont += 1
    cont = 0


# Radios
ra = 10
rb = 5

# Densidad
D = np.zeros(numero_puntos)

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

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Grilla')
plt.grid(True)
plt.legend()
plt.show()

print("Valores máximos = {}".format(MD),\
      "Coeficientes Delta = {}".format(MD[0]/MD[:]))
# Restructuración de densidades
cont = 0
umbral = 1.15
while MD[0]/MD[cont] < umbral:
    
    for i in range(0,numero_puntos):
        distancia = math.sqrt( (xi[0][i] - xi[0][indice_maximo])**2 + \
                                  (xi[1][i] - xi[1][indice_maximo])**2 )
        D[i] = D[i] - MD[cont]*math.exp( -distancia/(rb/2)**2 )
    
    indice_maximo = np.argmax(D)
    MD.append(D[indice_maximo])
    indices_maximos.append(indice_maximo)
    cont += 1       

colores = ["#F796E1","#FF0000","#8E44AD","#BCF558","#0000FF","#17A589"]
plt.figure(1)
plt.scatter(xi[0],xi[1],s=10,color='black',label='Puntos')
for i in range(len(MD)):
    plt.scatter(xi[0][indices_maximos[i]],xi[1][indices_maximos[i]],s=100,color = colores[i],label=f'Cluster {i+1}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Grilla')
plt.grid(True)
plt.legend()
plt.show()

print("Valores máximos = {}".format(MD),\
      "Coeficientes Delta = {}".format(MD[0]/MD[:]))
