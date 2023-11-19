import numpy as np
import matplotlib.pyplot as plt 

x = np.arange(0, 359, 1) 
y = np.arange(0, 359, 1)      
z1 = np.arange(0, 600, 1)
z2 = np.arange(0, 2, 0.01)


#Posición del generador
A1 = np.zeros(len(x)) #Norte EG
A2 = np.zeros(len(x)) #Este G
A3 = np.zeros(len(x)) #Sur G
A4 = np.zeros(len(x)) #Oeste G
A5 = np.zeros(len(x)) #Norte OG
#Posición de la veleta
B1 = np.zeros(len(y)) #Norte EG
B2 = np.zeros(len(y)) #Este G
B3 = np.zeros(len(y)) #Sur G
B4 = np.zeros(len(y)) #Oeste G
B5 = np.zeros(len(y)) #Norte OG

#Velocidad de giro
C1 = np.zeros(len(z1)) #Baja
C2 = np.zeros(len(z1)) #Media
C3 = np.zeros(len(z1)) #Alta

#Sentido de giro
D1 = np.zeros(len(z2)) #0
D2 = np.zeros(len(z2)) #1
D3 = np.zeros(len(z2)) #2

def funcionTrapezoidal(a,b,c,d,C,r):

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

#Posición del generador
A1 = funcionTrapezoidal(-1,-1,30,60,A1,x)
A2 = funcionTrapezoidal(30,60,120,150,A2,x)
A3 = funcionTrapezoidal(120,150,210,240,A3,x)
A4 = funcionTrapezoidal(210,240,300,330,A4,x)
A5 = funcionTrapezoidal(300,330,359,359,A5,x)

plt.figure(1)
plt.plot(x,A1,label = "Norte EG")
plt.plot(x,A2,label = "Este G")
plt.plot(x,A3,label = "Sur EG")
plt.plot(x,A4,label = "Oeste G")
plt.plot(x,A5,label = "Norte OG")
plt.legend()
plt.xlabel("Posición del Generador")
plt.ylabel("Membresía")

#Posición de la veleta
B1 = funcionTrapezoidal(-1,-1,30,60,B1,y)
B2 = funcionTrapezoidal(30,60,120,150,B2,y)
B3 = funcionTrapezoidal(120,150,210,240,B3,y)
B4 = funcionTrapezoidal(210,240,300,330,B4,y)
B5 = funcionTrapezoidal(300,330,359,359,B5,y)

plt.figure(2)
plt.plot(y,B1,label = "Norte EG")
plt.plot(y,B2,label = "Este G")
plt.plot(y,B3,label = "Sur EG")
plt.plot(y,B4,label = "Oeste G")
plt.plot(y,B5,label = "Norte OG")
plt.legend()
plt.xlabel("Posición de la Veleta")
plt.ylabel("Membresía")

#Velocidad de giro
C1 = funcionTrapezoidal(-1,-1,100,200,C1,z1)
C2 = funcionTrapezoidal(100,200,400,500,C2,z1)
C3 = funcionTrapezoidal(400,500,600,600,C3,z1)

plt.figure(3)
plt.plot(z1,C1,label = "Baja")
plt.plot(z1,C2,label = "Media")
plt.plot(z1,C3,label = "Alta")
plt.legend()
plt.xlabel("Velocidad [Hz]")
plt.ylabel("Membresía")

#Sentido de giro
D1 = funcionTrapezoidal(-1*100,-1*100,0.5*100,0.9*100,D1,z2)
D2 = funcionTrapezoidal(0.5*100,0.9*100,1.1*100,1.5*100,D2,z2)
D3 = funcionTrapezoidal(1.1*100,1.5*100,2*100,2*100,D3,z2)

plt.figure(4)
plt.plot(z2,D1,label = "Antihorario")
plt.plot(z2,D2,label = "No mover")
plt.plot(z2,D3,label = "Horario")
plt.legend()
plt.xlabel("Sentido de Giro")
plt.ylabel("Membresía")


#Obtención de los grados de pertenencia a los conjuntos de los universos de entrada

valx = int(input("Ingrese el valor de la posición del generador(0-359): "))
memA = [A1[valx],A2[valx],A3[valx],A4[valx],A5[valx]]
valy = int(input("Ingrese el valor de la veleta (0-359): "))
memB = [B1[valy],B2[valy],B3[valy],B4[valy],B5[valy]]

#Composición max - min (Evaluación de las reglas "si-entonces")

#Tablas de inferencia (intersección - min)
tabla = [[np.minimum(memA[0],memB[0]),
          np.minimum(memA[1],memB[0]),
         np.minimum(memA[2],memB[0]),
         np.minimum(memA[3],memB[0]),
         np.minimum(memA[4],memB[0])],
         [np.minimum(memA[0],memB[1]),
          np.minimum(memA[1],memB[1]),
         np.minimum(memA[2],memB[1]),
         np.minimum(memA[3],memB[1]),
         np.minimum(memA[4],memB[1])],
         [np.minimum(memA[0],memB[2]),
          np.minimum(memA[1],memB[2]),
         np.minimum(memA[2],memB[2]),
         np.minimum(memA[3],memB[2]),
         np.minimum(memA[4],memB[2])],
         [np.minimum(memA[0],memB[3]),
          np.minimum(memA[1],memB[3]),
         np.minimum(memA[2],memB[3]),
         np.minimum(memA[3],memB[3]),
         np.minimum(memA[4],memB[3])],
         [np.minimum(memA[0],memB[4]),
          np.minimum(memA[1],memB[4]),
         np.minimum(memA[2],memB[4]),
         np.minimum(memA[3],memB[4])
         ,np.minimum(memA[4],memB[4])]] 

print("Tabla de inferencia: ",str(tabla))

#Agregación - max hacia Velocidad de giro
agregacionC1 = max(tabla[0][0],max(tabla[1][1],
                                   max(tabla[2][2],
               max(tabla[3][3],max(tabla[4][4],
                                   max(tabla[4][0],tabla[0][4])))))) #Baja
agregacionC2 = max(tabla[0][1],max(tabla[0][3],
                                   max(tabla[1][0],
               max(tabla[1][2],max(tabla[1][4],
                                   max(tabla[2][1],
               max(tabla[2][3],max(tabla[3][0],
                                   max(tabla[3][2],
               max(tabla[3][4],max(tabla[4][1],
                                   tabla[4][3]))))))))))) #Media
agregacionC3 = max(tabla[0][2],max(tabla[1][3],
                                   max(tabla[2][0],
               max(tabla[2][4],max(tabla[3][1],
                                   tabla[4][2]))))) #Alta              

#Agregación - max hacia Sentido de giro
agregacionD1 = max(tabla[0][0],max(tabla[1][1],
        max(tabla[2][2],max(tabla[3][3],max(tabla[4][4],
        max(tabla[4][0],tabla[0][4])))))) #Baja
agregacionD2 = max(tabla[0][1],max(tabla[0][2],
            max(tabla[1][2],max(tabla[2][3],max(tabla[2][4],
            max(tabla[3][0],max(tabla[3][1],
               max(tabla[3][4],tabla[4][1])))))))) #Media
agregacionD3 = max(tabla[0][3],max(tabla[1][0],
            max(tabla[1][3],max(tabla[1][4],max(tabla[2][0],
            max(tabla[2][1],max(tabla[3][2],
               max(tabla[4][2],tabla[4][3])))))))) #Alta

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
 
plt.figure(5)
plt.plot(z1,C1,z1,C2,z1,C3)
plt.plot(z1,C1_r,label = "C1 recortado")
plt.plot(z1,C2_r,label = "C2 recortado")
plt.plot(z1,C3_r,label = "C3 recortado")
plt.xlabel("Velocidad [Hz]")
plt.ylabel("Membresía")
plt.legend()
 
plt.figure(6)
plt.plot(z2,D1,z2,D2,z2,D3)
plt.plot(z2,D1_r,label = "D1 recortado")
plt.plot(z2,D2_r,label = "D2 recortado")
plt.plot(z2,D3_r,label = "D3 recortado")
plt.xlabel("Sentido de giro")
plt.ylabel("Membresía") 
plt.legend()

#Dedifusificación

#Velocidad de Giro
z_d = 0
nume = 0
deno = 0

CT1 = np.maximum(C1_r,np.maximum(C2_r,C3_r))
plt.figure(7)
plt.plot(z1,CT1)
plt.xlabel("Velocidad [Hz]")
plt.ylabel("Membresía")

for i in range(len(z1)):
  nume += CT1[i]*i
  deno += CT1[i]

z_d = nume/deno
print("z* = ", str(z_d), " [Hz]")

#Sentido de giro

z_d = 0
nume = 0
deno = 0

CT2 = np.maximum(D1_r,np.maximum(D2_r,D3_r))
plt.figure(8)
plt.plot(z2,CT2)
plt.xlabel("Sentido de giro")
plt.ylabel("Membresía")

for i in range(len(z2)):
  nume += CT2[i]*i/100
  deno += CT2[i]

z_d = nume/deno
print("z* = ", str(z_d))

