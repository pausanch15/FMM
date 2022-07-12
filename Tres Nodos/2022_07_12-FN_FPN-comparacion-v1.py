#En este archivo comparo el contador estricto y no estricto en FN y FPN sin reverberaciones

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk
import integra_FN as fn
import integra_FPN_desacoplado_parametro_feedback as fpndespf
from scipy import interpolate
from scipy.signal import find_peaks
import runge_kuta_estimulo as rks
plt.ion()

#%%
#Función que implementa los contadores
def aplica_contadores(tiempo, variables, tb, porcentaje_umbral = 20):
    X, y, Y = variables
    
    #Calculo el tiempo en el cual el estímulo empieza a decrecer
    i_tb = np.where(tiempo>tb)[0][0]
    tiempo_decaimiento = tiempo[i_tb:]

    #Encuentro todos los picos
    picos_todos, caracteristicas_todos = find_peaks(X, height=X)
    altura_picos_todos = caracteristicas_todos['peak_heights']
    
    #Me quedo con el pico máximo
    i_pico_max = np.where(altura_picos_todos==np.max(altura_picos_todos))[0][0]
    pico_max = picos_todos[i_pico_max]
    altura_pico_max = altura_picos_todos[i_pico_max]

    #Empiezo a definir los contadores
    #Encuentro el tiempo y el X una vex que empieza a decaer el estímulo
    tiempo_decaimiento = tiempo[i_tb:]
    X_decaimiento = X[i_tb:]
    #Contador No Estricto: cuento todos los picos una vez que empieza a bajar el estímulo
    picos_NE, caracteristicas_NE = find_peaks(X_decaimiento, height=X_decaimiento)
    altura_picos_NE = caracteristicas_NE['peak_heights']
    c_NE = len(picos_NE) #cotador_NoEstricto
    
    #Hago un promedio entre todos los picos durante el estímulo cuya amplitud es menor que la del mayor.
    tiempo_picos_escalon = np.where(picos_todos<i_tb, picos_todos, 0)
    altura_picos_escalon = np.where(picos_todos<i_tb, altura_picos_todos, 0)
    prom_picos_escalon = np.mean(altura_picos_escalon)
    
    #Contador Estricto: cuento solo los picos con mas/menos un porcentaje de la amplitud del promedio de la altura de los picos durante el escalón
    umbral = porcentaje_umbral/100
    picos_E, caracteristicas_E = find_peaks(X_decaimiento, height=prom_picos_escalon*umbral)
    altura_picos_E = caracteristicas_E['peak_heights']
    c_E = len(picos_E) #cotador_Estricto

    return [picos_NE, altura_picos_NE, picos_E, altura_picos_E, tiempo_decaimiento]

#%%
#Defino el sistema FN
C1 = 4.5
dX1 = 0.01
dYX1 = 7
KYX1 = 0.04
Ty1 = 0.35
dy1 = 0.4
TY1 = 0.67
dY1 = 0.25

tiempo, variables, tiempo_estimulo, estimulo = fn.integra_FN_estimulo(dX1, dYX1, KYX1, Ty1, dy1, TY1, dY1, tiempo_max=1000, resolucion=1000, ti=0, tf=200, S_min=0, S_max=5, tau=20, ts=20, tb=70, condiciones_iniciales=[0.01, 0.01, 0.01])

X, y, Y = variables

#Aplico contadores y ploteo
picos_NE, altura_picos_NE, picos_E, altura_picos_E, tiempo_decaimiento = aplica_contadores(tiempo, variables, tb=70, porcentaje_umbral=60)

plt.figure()
plt.plot(tiempo, X, label='X')
plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
plt.plot(tiempo_decaimiento[picos_E], altura_picos_E, 'o', label='Contador Estricto')
plt.plot(tiempo_decaimiento[picos_NE], altura_picos_NE, '.', label='Contador No Estricto')
plt.grid()
plt.title('FN Con los Valores del Paper')
plt.legend()
plt.show()

#Defino el sistema FPN
dYX2 = 0.14
dX2 = 0.1
Ty2 = 0.21
dy2 = 0.1
TY2 = 0.3
dY2 = 0.1
f = 1

tiempo, variables, tiempo_estimulo, estimulo = fpndespf.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, f, tiempo_max=2000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=0.04, tau=200, ts=30, tb=530, condiciones_iniciales=[0.01, 0.01, 0.01])

epsilon = 0.01
X, y, Y = variables*epsilon

#Aplico contadores y ploteo
picos_NE, altura_picos_NE, picos_E, altura_picos_E, tiempo_decaimiento = aplica_contadores(tiempo, variables*epsilon, tb=530, porcentaje_umbral=60)

plt.figure()
plt.plot(tiempo, X, label='X')
plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
plt.plot(tiempo_decaimiento[picos_E], altura_picos_E, 'o', label='Contador Estricto')
plt.plot(tiempo_decaimiento[picos_NE], altura_picos_NE, '.', label='Contador No Estricto')
plt.grid()
plt.title('FPN Con los Valores del Paper')
plt.legend()
plt.show()

#Barro en f y hago esto mismo
del(f)
f_s = np.linspace(0.7, 1.1, 10)

for f in f_s:
    tiempo, variables, tiempo_estimulo, estimulo = fpndespf.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, f, tiempo_max=2000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=0.04, tau=200, ts=30, tb=530, condiciones_iniciales=[0.01, 0.01, 0.01])
    
    epsilon = 0.01
    X, y, Y = variables*epsilon
    
    picos_NE, altura_picos_NE, picos_E, altura_picos_E, tiempo_decaimiento = aplica_contadores(tiempo, variables*epsilon, tb=530, porcentaje_umbral=60)
    
    plt.figure()
    plt.plot(tiempo, X, label='X')
    plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
    plt.plot(tiempo_decaimiento[picos_E], altura_picos_E, 'o', label='Contador Estricto')
    plt.plot(tiempo_decaimiento[picos_NE], altura_picos_NE, '.', label='Contador No Estricto')
    plt.grid()
    plt.title(f'f={f}')
    plt.legend()
    plt.show()
