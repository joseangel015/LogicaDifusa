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
CZ = np.zeros((len(x),len(y)))

for i in range(len(x)):
  A1[i] = 1 / ( 1 + exp(0.3*(x[i]-50)))
  A2[i] = 1 / ( 1 + exp(-0.3*(x[i]-50)))
  B1[i] = exp(-1/2*((y[i]-25)/20)**2)
  B2[i] = exp(-1/2*((y[i]-75)/20)**2)
  C1[i] = 1 / ( 1 + exp(0.3*(z[i]-50)))
  C2[i] = 1 / ( 1 + exp(-0.3*(z[i]-50)))


def difusificacion(valx,valy):
  memA = [A1[valx],A2[valx]]
  memB = [B1[valy],B2[valy]]
  return composicion(memA,memB)

def composicion(memA,memB): 
  
  tabla = [[np.minimum(memA[0],memB[0]),np.minimum(memA[1],memB[0])],
           [np.minimum(memA[0],memB[1]),np.minimum(memA[1],memB[1])]]

  return dedifusificacion(tabla)
  

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
   
   for i in [0,1,2,3]:   
     numerador += wi[i]*zi[i]
     denominador += wi[i]
   
   z_d = numerador/denominador
   
   return z_d

def malla():
  for i in range(len(x)):
    for j in range(len(y)):
      CZ[i,j] = difusificacion(i,j) #Superficie de control generada
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  ax.set_xlabel('Y')
  ax.set_ylabel('X')
  ax.set_zlabel('Z')
  surf = ax.plot_surface(X, Y, CZ, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
  

malla()
