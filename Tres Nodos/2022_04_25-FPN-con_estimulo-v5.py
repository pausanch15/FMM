#Para casos en donde no vemos reververaciones, hacemos un barrido en los parámetros de feedback positivo F y negativo dYX2.
#Aplicamos también el contadorde memoria que propone el paper para estos casos.

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk
import integra_FPN_desacoplado_parametro_feedback as fpndesf
from scipy import interpolate
import runge_kuta_estimulo as rks
from scipy.signal import find_peaks
plt.ion()

#%%
#Elegimos los valores en los que vamos a barrer
f_s, dYX2_s = np.linspace(0, 1, 15), np.linspace(0, 0.7, 15)

#Barro en f
dYX2 = 0.14
dX2 = 0.1
Ty2 = 0.21
dy2 = 0.1
TY2 = 0.3
dY2 = 0.1
epsilon = 0.01

# for f in f_s:
    # tiempo, variables, tiempo_estimulo, estimulo = fpndesf.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, f, tiempo_max=2000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=0.04, tau=200, ts=30, tb=530, condiciones_iniciales=[0.01, 0.01, 0.01])
# 
    # X, y, Y = variables*epsilon
# 
    # plt.figure()
    # plt.plot(tiempo, X, label='X')
    # plt.plot(tiempo, y, label='y')
    # plt.plot(tiempo, Y, label='Y')
    # plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
    # plt.grid()
    # plt.title(f'{f=:.2}')
    # plt.legend()
    # plt.show()
    # plt.savefig(f'2022_05_26-barrido_f_{f}.png')
    # plt.close()

#Barro en dYX2
f = 1
dX2 = 0.1
Ty2 = 0.21
dy2 = 0.1
TY2 = 0.3
dY2 = 0.1
epsilon = 0.01

# for dYX2 in dYX2_s:
    # tiempo, variables, tiempo_estimulo, estimulo = fpndesf.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, f, tiempo_max=2000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=0.04, tau=200, ts=30, tb=530, condiciones_iniciales=[0.01, 0.01, 0.01])
# 
    # X, y, Y = variables*epsilon
# 
    # plt.figure()
    # plt.plot(tiempo, X, label='X')
    # plt.plot(tiempo, y, label='y')
    # plt.plot(tiempo, Y, label='Y')
    # plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
    # plt.grid()
    # plt.title(f'{dYX2=:.2}')
    # plt.legend()
    # plt.show()
    # plt.savefig(f'2022_05_26-barrido_dYX2_{dYX2}.png')
    # plt.close()

#Para el barrido en f, cuento los picos que hay una vez que empieza a decaer el escalón.
dYX2 = 0.14
dX2 = 0.1
Ty2 = 0.21
dy2 = 0.1
TY2 = 0.3
dY2 = 0.1
epsilon = 0.01
tb = 530

memoria_f = []

for f in f_s:
    tiempo, variables, tiempo_estimulo, estimulo = fpndesf.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, f, tiempo_max=2000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=0.04, tau=200, ts=30, tb=530, condiciones_iniciales=[0.01, 0.01, 0.01])

    X, y, Y = variables*epsilon
    i_tb = np.where(tiempo>tb)[0][0]
    picos, caracteristicas = find_peaks(X[i_tb:], height=X[i_tb])
    altura_picos = caracteristicas['peak_heights']

    #Memoria: la defino como el número de picos que se cuentan una vez que el estímulo aplicado empieza a decrecer
    memoria_f.append(len(picos))

    #Plots
    # plt.figure()
    # plt.plot(tiempo, X, label='X')
    # plt.plot(tiempo[i_tb:][picos], altura_picos, 'o', label='Memoria')
    # plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
    # plt.grid()
    # plt.title(f'{f=:.2}')
    # plt.legend()
    # plt.show()
    # plt.savefig(f'2022_05_26-memoria-barrido_f_{f}.png')
    # plt.close()

#Para el barrido en dYX2, cuento los picos que hay una vez que empieza a decaer el escalón.
f = 1
dX2 = 0.1
Ty2 = 0.21
dy2 = 0.1
TY2 = 0.3
dY2 = 0.1
epsilon = 0.01
tb = 530

memoria_dYX2 = []

for dYX2 in dYX2_s:
    tiempo, variables, tiempo_estimulo, estimulo = fpndesf.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, f, tiempo_max=2000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=0.04, tau=200, ts=30, tb=530, condiciones_iniciales=[0.01, 0.01, 0.01])

    X, y, Y = variables*epsilon
    i_tb = np.where(tiempo>tb)[0][0]
    picos, caracteristicas = find_peaks(X[i_tb:], height=X[i_tb])
    altura_picos = caracteristicas['peak_heights']

    #Memoria: la defino como el número de picos que se cuentan una vez que el estímulo aplicado empieza a decrecer
    memoria_dYX2.append(len(picos))

