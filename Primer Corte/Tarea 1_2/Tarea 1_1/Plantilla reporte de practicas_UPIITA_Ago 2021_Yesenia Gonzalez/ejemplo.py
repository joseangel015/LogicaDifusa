#Operaciones con conjuntos certeros
import numpy as np 
import matplotlib.pyplot as plt 

# Variables 
x = np.arange(0, 10, 0.1)                         # Rango del universo
a1=20
b1=40
a2=30
b2=50
A=np.zeros(len(x))
B=np.zeros(len(x))
union=np.zeros(len(x))
intersec=np.zeros(len(x))
for i in range(len(x)):
  if i < a1:
    A[i]=0
  if i >= a1 and i < b1:
    A[i]=1
  else:
    A[i]=0;
  if i < a2:
    B[i]=0 
  if i >= a2 and i < b2:
    B[i]=1
  else:
    B[i]=0
plt.plot(x,A,x,B)