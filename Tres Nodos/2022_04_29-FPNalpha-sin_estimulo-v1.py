#Intento integrar el modelo de Feedback Positivo Negativo de tres nodos

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk
import integra_FPNalpha as fpn
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
C2 = 0.015
dYX2 = 0.14
dX2 = 0.1
Ty2 = 0.21
dy2 = 0.1
TY2 = 0.3
dY2 = 0.1
alpha = 0.5

tiempo, variables = fpn.integra_FPN(C1, dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1, C2, dYX2, dX2, Ty2, dy2, TY2, dY2, alpha, tiempo_max=100000)

X, y, Y = variables

plt.figure()
plt.plot(tiempo, X, label='X')
plt.plot(tiempo, y, label='y')
plt.plot(tiempo, Y, label='Y')
plt.grid()
plt.title('Con los Valores del Paper')
plt.legend()
plt.show()

#Me fijo que pasa al variar el alpha
alphas = np.linspace(0, 1, 6)
fig, axs = plt.subplots(2, 3, sharex=False, sharey=False)
for i, ax in enumerate(axs.flatten()):
    tiempo, variables = fpn.integra_FPN(C1, dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1, C2, dYX2, dX2, Ty2, dy2, TY2, dY2, alpha=alphas[i], tiempo_max=100000)
    
    X, y, Y = variables
    
    ax.plot(tiempo, X, label='X')
    ax.plot(tiempo, y, label='y')
    ax.plot(tiempo, Y, label='Y')
    ax.grid()
    ax.legend()
    # ax.set_xlabel('Tiempo')
    ax.set_title(f"alpha={alphas[i]}")
plt.show()
