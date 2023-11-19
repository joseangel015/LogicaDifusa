import numpy as np
from math import exp 
import matplotlib.pyplot as plt 
from matplotlib import cm


x = np.arange(0, 100, 0.1) 
y = np.arange(0, 100, 0.1)      
z = np.arange(0, 100, 0.1)                  
X, Y = np.meshgrid(x, y)

A1 = np.zeros(len(x))
A2 = np.zeros(len(x))
B1 = np.zeros(len(y))
B2 = np.zeros(len(y))
C1 = np.zeros(len(z))
C2 = np.zeros(len(z))


#Obtención de los grados de pertenencia a los conjuntos de los universos de entrada
def difusificacion(): 
  valx = int(input("Ingrese el valor de x: "))
  memA = [A1[valx*10],A2[valx*10]]
  valy = int(input("Ingrese el valor de y: "))
  memB = [B1[valy*10],B2[valy*10]]
  composicion(memA,memB)

#Composición max - min (Evaluación de las reglas "si-entonces")
def composicion(memA,memB):
  
#Tabla de inferencia (intersección - min)
  tabla = [[np.minimum(memA[0],memB[0]),np.minimum(memA[1],memB[0])],
           [np.minimum(memA[0],memB[1]),np.minimum(memA[1],memB[1])]] 
  dedifusificacion(tabla)
  

def dedifusificacion(tabla):
  
  z_d = 0
  wi = [tabla[0][0],tabla[1][0],tabla[0][1],tabla[1][1]]
  zi = np.zeros(4)
  numerador = 0
  denominador = 0
  
  zi[0] = 50 + np.log(1/wi[0]-1)/0.3
  zi[1] = 50 + np.log(1/wi[1]-1)/0.3
  zi[2] = 50 - np.log(1/wi[2]-1)/0.3
  zi[3] = 50 - np.log(1/wi[3]-1)/0.3
  
  print(wi)
  print(zi)
  for i in [0,1,2,3]:   
    numerador += wi[i]*zi[i]
    denominador += wi[i]
  
  z_d = numerador/denominador
  print("z* = ", str(z_d))


#Generación de los conjuntos difusos
for i in range(len(x)):
  A1[i] = 1 / ( 1 + exp(0.3*(x[i]-50)))
  A2[i] = 1 / ( 1 + exp(-0.3*(x[i]-50)))
  B1[i] = exp(-1/2*((y[i]-25)/20)**2)
  B2[i] = exp(-1/2*((y[i]-75)/20)**2)
  C1[i] = 1 / ( 1 + exp(0.3*(z[i]-50)))
  C2[i] = 1 / ( 1 + exp(-0.3*(z[i]-50)))


plt.figure(1)
plt.plot(x,A1,x,A2)
plt.xlabel("Universo X")
plt.ylabel("Membresía")

plt.figure(2)
plt.plot(y,B1,y,B2)
plt.xlabel("Universo Y")
plt.ylabel("Membresía")

plt.figure(3)
plt.plot(z,C1,z,C2)
plt.xlabel("Universo Z")
plt.ylabel("Membresía")

difusificacion()
