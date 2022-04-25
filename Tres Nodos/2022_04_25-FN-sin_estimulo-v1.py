#Intento integrar el modelo de Feedback Negativo de tres nodos

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk
import integra_FN as fn
plt.ion()

#%%
#Integramos
#Primero con los parámetros que propone el paper
C1 = 4.5
dX1 = 0.01
dYX1 = 7
KYX1 = 0.04
Ty1 = 0.35
dy1 = 0.4
TY1 = 0.67
dY1 = 0.25

tiempo, variables = fn.integra_FN(C1, dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1, tiempo_max=100000)

X, y, Y = variables

plt.figure()
plt.plot(tiempo, X, label='X')
plt.plot(tiempo, y, label='y')
plt.plot(tiempo, Y, label='Y')
plt.grid()
plt.title('Con los Valores del Paper')
plt.legend()
plt.show()

#Ahora cambio los parámetros. Elijo otros del mismo orden de magnitud que los del paper, pero bastante distintos
C1 = 2
dX1 = 0.01 
dYX1 = 10
KYX1 = 0.04
Ty1 = 0.3
dy1 = 0.4
TY1 = 0.6
dY1 = 0.2

tiempo, variables = fn.integra_FN(C1, dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1, tiempo_max=100000)

X, y, Y = variables

plt.figure()
plt.plot(tiempo, X, label='X')
plt.plot(tiempo, y, label='y')
plt.plot(tiempo, Y, label='Y')
plt.grid()
plt.title('Cambiando los Valores del Paper')
plt.legend()
plt.show()
