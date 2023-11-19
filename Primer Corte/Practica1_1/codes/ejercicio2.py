# Para la ejecución de este programa es necesario contar con las variables 
# declaradas en el ejercicio 1

def difusificacion(valx,valy):
  memA = [A1[valx],A2[valx]]
  memB = [B1[valy],B2[valy]]
  return composicion(memA,memB)

def composicion(memA,memB): 
  
  tabla = [[np.minimum(memA[0],memB[0]),np.minimum(memA[1],memB[0])],
           [np.minimum(memA[0],memB[1]),np.minimum(memA[1],memB[1])]]
  agregacionC1 = max(tabla[0][0],tabla[1][0])
  agregacionC2 = max(tabla[0][1],tabla[1][1])
  return agregacion(agregacionC1,agregacionC2)
  

def agregacion(agregacionC1,agregacionC2):
  aux1 = agregacionC1*np.ones(len(C1))
  aux2 = agregacionC2*np.ones(len(C2))
  
  C1_r = np.minimum(C1,aux1)
  C2_r = np.minimum(C2,aux2)
 
  return dedifusificacion(C1_r,C2_r)

def dedifusificacion(C1_r,C2_r):
  z_d = 0
  aux1 = 0
  aux2 = 0
  CT = np.maximum(C1_r,C2_r)

  for i in range(len(z)):
    aux1 += CT[i]*i/10
    aux2 += CT[i]
  
  z_d = aux1/aux2
  return z_d

def malla():
  for i in range(len(x)):
    for j in range(len(y)):
      CM[i,j] = difusificacion(i,j) #Superficie de control generada
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  ax.set_xlabel('Y')
  ax.set_ylabel('X')
  ax.set_zlabel('Z')
  surf = ax.plot_surface(X, Y, CM, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
  

malla()
