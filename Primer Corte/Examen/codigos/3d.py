import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm


x = np.arange(0, 50, 0.1) 
y = np.arange(0, 100, 0.2)      
z1 = np.arange(15, 30, 0.1)
z2 = np.arange(0, 3, 0.01)
X, Y = np.meshgrid(x, y)

#Temperatura seca
A1 = np.zeros(len(x)) #Frio
A2 = np.zeros(len(x)) #Templado
A3 = np.zeros(len(x)) #Caliente
A4 = np.zeros(len(x)) #Muy caliente

#Humedad relativa
B1 = np.zeros(len(y)) #Seco
B2 = np.zeros(len(y)) #Zona de comfort
B3 = np.zeros(len(y)) #Húmedo


#Temperatura del aire
C1 = np.zeros(len(z1)) #Frio
C2 = np.zeros(len(z1)) #Templado
C3 = np.zeros(len(z1)) #Caliente

#Velocidad del viento
D1 = np.zeros(len(z2)) #Calma
D2 = np.zeros(len(z2)) #Aire ligero
D3 = np.zeros(len(z2)) #Brisa ligera
D4 = np.zeros(len(z2)) #Brisa suave


def funcionCampana(a,b,c,C,r):

  a = a*10 #Ancho de la campana
  b = b*10 #Razon de cambio
  c = c*10 #Centro de la campana

  for i in range(len(r)):
    C[i] = 1/(1 + abs((i-c)/a)**(2*b))
  
  return C

def funcionTrapezoidal(a,b,c,d,C,r):
  a = a*10
  b = b*10
  c = c*10
  d = d*10
  for i in range(len(r)):
    if i <= a:
      C[i] = 0
    elif a < i and i <= b:
      C[i] = (i - a)/(b - a)
    elif b < i and i <= c:
      C[i] = 1
    elif c < i and i <= d:
      C[i] = (d - i)/(d - c)
    elif d < i:
      C[i] = 0

  return C

#Generación de los conjuntos difusos

#Temperatura seca
A1 = funcionCampana(10,0.25,0,A1,x)
A2 = funcionCampana(10,0.25,15,A2,x)
A3 = funcionCampana(10,0.25,35,A3,x)
A4 = funcionCampana(10,0.25,50,A4,x)

#Humedad relativa
B1 = funcionTrapezoidal(-1,-1,20,30,B1,y)
B2 = funcionTrapezoidal(20,30,70,80,B2,y)
B3 = funcionTrapezoidal(70,80,100,100,B3,y)

# Temperatura del viento
C1 = funcionCampana(5,0.25,0,C1,z1)
C2 = funcionCampana(5,0.25,7.5,C2,z1)
C3 = funcionCampana(5,0.25,15,C3,z1)

#Velocidad del viento
D1 = funcionTrapezoidal(-1*10,-1*10,0.37*10,0.62*10,D1,z2)
D2 = funcionTrapezoidal(0.37*10,0.62*10,1.37*10,1.62*10,D2,z2)
D3 = funcionTrapezoidal(1.37*10,1.62*10,2.37*10,2.62*10,D3,z2)
D4 = funcionTrapezoidal(2.37*10,2.62*10,3*10,3*10,D4,z2)

CZ1 = np.zeros((len(x),len(y)))
CZ2 = np.zeros((len(x),len(y)))

#Obtención de los grados de pertenencia a los conjuntos de los universos de entrada

def difusificacion(valx,valy):

  memA = [A1[valx],A2[valx],A3[valx],A4[valx]]
  memB = [B1[valy],B2[valy],B3[valy]]

  #Composición max - min (Evaluación de las reglas "si-entonces")
  
  #Tablas de inferencia (intersección - min)
  tabla = [[np.minimum(memA[0],memB[0]),np.minimum(memA[1],
            memB[0]),np.minimum(memA[2],memB[0]),np.minimum(memA[3],memB[0])],
            [np.minimum(memA[0],memB[1]),np.minimum(memA[1],
            memB[1]),np.minimum(memA[2],memB[1]),np.minimum(memA[3],memB[1])],
            [np.minimum(memA[0],memB[2]),np.minimum(memA[1],
            memB[2]),np.minimum(memA[2],memB[2]),np.minimum(memA[3],memB[2])]]


  #Agregación - max hacia Temperatura del viento
  agregacionC1 = max(tabla[0][2],max(tabla[0][3],
                 max(tabla[1][3],max(tabla[2][2],tabla[2][3])))) #Frio
  agregacionC2 = max(tabla[0][1],max(tabla[1][0],
                 max(tabla[1][1],max(tabla[1][2],tabla[2][1])))) #Templado
  agregacionC3 = max(tabla[0][0],tabla[2][0]) #Caliente

  #Agregación - max hacia Velocidad del viento
  agregacionD1 = tabla[1][1] #Calma
  agregacionD2 = max(tabla[0][0],max(tabla[0][1],
                 max(tabla[1][0],max(tabla[2][0],tabla[2][1])))) #Aire ligero
  agregacionD3 = max(tabla[0][2],max(tabla[1][2],tabla[2][2])) #Brisa ligera
  agregacionD4 = max(tabla[0][3],max(tabla[1][3],tabla[2][3])) #Brisa suave

  aux1 = agregacionC1*np.ones(len(C1))
  aux2 = agregacionC2*np.ones(len(C2))
  aux3 = agregacionC3*np.ones(len(C3))

  aux4 = agregacionD1*np.ones(len(D1))
  aux5 = agregacionD2*np.ones(len(D2))
  aux6 = agregacionD3*np.ones(len(D3))
  aux7 = agregacionD4*np.ones(len(D4))
    
  aux1 = agregacionC1*np.ones(len(C1))
  aux2 = agregacionC2*np.ones(len(C2))
  aux3 = agregacionC3*np.ones(len(C3))

  aux4 = agregacionD1*np.ones(len(D1))
  aux5 = agregacionD2*np.ones(len(D2))
  aux6 = agregacionD3*np.ones(len(D3))
  
  C1_r = np.minimum(C1,aux1)
  C2_r = np.minimum(C2,aux2)
  C3_r = np.minimum(C3,aux3)

  D1_r = np.minimum(D1,aux4)
  D2_r = np.minimum(D2,aux5)
  D3_r = np.minimum(D3,aux6)
  D4_r = np.minimum(D4,aux7)
  
  z_d = np.zeros((1,2))
  nume = 0
  deno = 0
  CT1 = np.maximum(C1_r,np.maximum(C2_r,C3_r))
  CT2 = np.maximum(D1_r,np.maximum(D2_r,np.maximum(D3_r,D4_r)))

  for i in range(len(z1)):
    nume += CT1[i]*(i/10+15)
    deno += CT1[i]

  z_d[0,0] = nume/deno
  
  nume = 0
  deno = 0
  
  for i in range(len(z2)):
    nume += CT2[i]*i/100
    deno += CT2[i]

  z_d[0,1] = nume/deno
  
  return z_d

for i in range(len(x)):
   for j in range(len(y)):
     CZ1[i,j] = difusificacion(i,j)[0][0]
     CZ2[i,j] = difusificacion(i,j)[0][1]
     

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_zlabel('Z')
surf = ax.plot_surface(X, Y, CZ1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
ax2.set_xlabel('Y')
ax2.set_ylabel('X')
ax2.set_zlabel('Z')
surf = ax2.plot_surface(X, Y, CZ2, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
