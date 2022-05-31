#Para loscasos en los que sí hay reververaciones, intento calcular lascantidades que propone el paper de Mitra

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk
import no_pushear_integra_FPN_desacoplado_parametro_feedback as fpndesf
from scipy import interpolate
import runge_kuta_estimulo as rks
from scipy.signal import find_peaks
plt.ion()

#%%
#Encontrar los picos de las reververacionesdebería ser exctamente igual que antes
#Lo que cambia es que hay que pedir que solo se tengan en cuenta los mayores que los que tiene el sistema durante el escalón.
dYX2 = 0.18 
dX2 = 0.1 
Ty2 = 0.21 
dy2 = 0.1 
TY2 = 0.3 
dY2 = 0.1 
altura_escalon = 0.1
f = 1
epsilon = 0.01

tiempo, variables, tiempo_estimulo, estimulo = fpndesf.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, f, tiempo_max=2000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=altura_escalon, tau=200, ts=30, tb=530, condiciones_iniciales=[0.01, 0.01, 0.01])

X, y, Y = variables*epsilon

#Todos los picos

#Picos de las reververaciones
tb = 530
umbral_memoria = 0.001 #Diferencia entre la amplitud de los picos durante el escalón y los que consideramos reververaciones
i_tb = np.where(tiempo>tb)[0][0]
picos, caracteristicas = find_peaks(X[i_tb:], height=X[i_tb]+umbral_memoria)
altura_picos = caracteristicas['peak_heights']

#Plots
plt.figure()
plt.plot(tiempo, X, label='X')
plt.plot(tiempo[i_tb:][picos], altura_picos, 'o', label='Memoria')
plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
plt.grid()
plt.legend()
plt.show()

#%%
#Empiezo a calcular las cantidades que propone el paper
t_primer_pico_rev = tiempo[i_tb:][picos[0]] #Tiempo del primer pico de las reververaciones
t_ultimo_pico_rev = tiempo[i_tb:][picos[-1]] #Tiempo del ultimo pico de las reververaciones

Nr = len(picos)
Tr = t_ultimo_pico_rev - t_primer_pico_rev