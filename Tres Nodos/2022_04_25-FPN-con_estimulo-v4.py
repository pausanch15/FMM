#Ahora integro el modelo con el C2 desacoplado del feedback positivo y estímulo. Agrego también el parámetro f que regula la intensidad del feedback positivo de X.

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk
import integra_FPN_desacoplado_parametro_feedback as fpndespf
from scipy import interpolate
import runge_kuta_estimulo as rks
plt.ion()

#Cosas de matplotlib para hacer los gráficos
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('ggplot')
plt.rc("text", usetex=True)
plt.rc('font', family='serif')
plt.ion()

#%%
#Integramos
#Primero con los parámetros que propone el paper
dYX2 = 0.14
dX2 = 0.1
Ty2 = 0.21
dy2 = 0.1
TY2 = 0.3
dY2 = 0.1
f = 1

tiempo, variables, tiempo_estimulo, estimulo = fpndespf.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, f, tiempo_max=2000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=0.04, tau=200, ts=30, tb=530, condiciones_iniciales=[0.01, 0.01, 0.01])

X, y, Y = variables
epsilon = 0.01
X, y, Y = variables*epsilon

plt.figure()
plt.plot(tiempo, X, label='$X$')
plt.plot(tiempo, y, label='$y$')
plt.plot(tiempo, Y, label='$Y$')
plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
plt.grid(1)
plt.yticks(fontsize=15, color='black')
plt.xticks(fontsize=15, color='black')
plt.title('FPN', fontsize=15)
plt.legend(fontsize=15)
plt.xlabel('Tiempo', fontsize=15, color='black')
plt.tight_layout()
plt.savefig('Figuras/FPNvalpaper.pdf', dpi=300)

#%%
#Trato de ver las reververaciones que encontramos con Ale
dYX2 = 0.18
altura_escalon = 0.1

tiempo, variables, tiempo_estimulo, estimulo = fpndespf.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, f, tiempo_max=2000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=altura_escalon, tau=200, ts=30, tb=530, condiciones_iniciales=[0.01, 0.01, 0.01])

X, y, Y = variables
epsilon = 0.01
X, y, Y = variables*epsilon

plt.figure()
plt.plot(tiempo, X, label='$X$')
plt.plot(tiempo, y, label='$y$')
plt.plot(tiempo, Y, label='$Y$')
plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
plt.grid(1)
plt.yticks(fontsize=15, color='black')
plt.xticks(fontsize=15, color='black')
plt.title('FPN con Reverberaciones', fontsize=15)
plt.legend(fontsize=15)
plt.xlabel('Tiempo', fontsize=15, color='black')
plt.tight_layout()
plt.savefig('Figuras/revFPN.pdf', dpi=300)
