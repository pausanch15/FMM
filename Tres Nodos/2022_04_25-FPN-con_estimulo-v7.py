#Hago un barrido en distintos valores de f
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
#Defino una función para hallar los taus 
def encuentra_taus(X):
    picos_todos, caracteristicas_todos = find_peaks(X, height=X)
    altura_picos_todos = caracteristicas_todos['peak_heights']

    #Picos de las reververaciones
    tb = 1030
    umbral_memoria = 0.001 #Diferencia entre la amplitud de los picos durante el escalón y los que consideramos reververaciones
    i_tb = np.where(tiempo>tb)[0][0]
    picos, caracteristicas = find_peaks(X[i_tb:], height=X[i_tb]+umbral_memoria)
    altura_picos = caracteristicas['peak_heights']
    #Empiezo a calcular las cantidades que propone el paper
    t_primer_pico = tiempo[picos_todos[0]] #Tiempo del primer pico de todos
    t_primer_pico_rev = tiempo[i_tb:][picos[0]] #Tiempo del primer pico de las reververaciones
    t_ultimo_pico_rev = tiempo[i_tb:][picos[-1]] #Tiempo del ultimo pico de las reververaciones

    Nr = len(picos)
    Tr = t_ultimo_pico_rev - t_primer_pico_rev
    Tm = t_ultimo_pico_rev - tiempo[i_tb]
    Tpr = Tm - Tr

    return [Nr, Tr, Tm, Tpr]

#%%
#Empiezo el barrido en los parámetros de feedback positivo y negativo
#Barrido en feedback positivo
dYX2 = 0.18 
dX2 = 0.1 
Ty2 = 0.21 
dy2 = 0.1 
TY2 = 0.3 
dY2 = 0.1 
altura_escalon = 0.1
epsilon = 0.01

Nr_s, Tr_s, Tm_s, Tpr_s = [], [], [], []

f_s = np.linspace(0.88, 1.11, 50)

for f in f_s:
    tiempo, variables, tiempo_estimulo, estimulo = fpndesf.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, f, tiempo_max=3000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=altura_escalon, tau=200, ts=30, tb=1030, condiciones_iniciales=[0.01, 0.01, 0.01])

    X, y, Y = variables*epsilon
    Nr, Tr, Tm, Tpr = encuentra_taus(X)
    Nr_s.append(Nr)
    Tr_s.append(Tr)
    Tm_s.append(Tm)
    Tpr_s.append(Tpr)

    #Plots
    plt.figure()
    plt.plot(tiempo, X, label='X')
    plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig(f'resultados/barrido_f_{f}.pdf')
    plt.close()

#Ploteo los taus en función del valor de f
plt.figure()
plt.plot(f_s, Nr_s, '-o', label='Nr')
plt.plot(f_s, Tr_s, '-o', label='Tr')
plt.plot(f_s, Tm_s, '-o', label='Tm')
plt.plot(f_s, Tpr_s, '-o', label='Tpr')
plt.xlabel('f')
plt.grid()
plt.legend()
plt.show()

#Barrido en feedback negativo
f = 1
dX2 = 0.1 
Ty2 = 0.21 
dy2 = 0.1 
TY2 = 0.3 
dY2 = 0.1 
altura_escalon = 0.1
epsilon = 0.01

Nr_s, Tr_s, Tm_s, Tpr_s = [], [], [], []

dYX2_s = np.linspace(0.15, 0.22, 50)

for dYX2 in dYX2_s:
    tiempo, variables, tiempo_estimulo, estimulo = fpndesf.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, f, tiempo_max=3000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=altura_escalon, tau=200, ts=30, tb=1030, condiciones_iniciales=[0.01, 0.01, 0.01])

    X, y, Y = variables*epsilon
    Nr, Tr, Tm, Tpr = encuentra_taus(X)
    Nr_s.append(Nr)
    Tr_s.append(Tr)
    Tm_s.append(Tm)
    Tpr_s.append(Tpr)

    #Plots
    plt.figure()
    plt.plot(tiempo, X, label='X')
    plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig(f'resultados/barrido_dYX2_{dYX2}.pdf')
    plt.close()

#Ploteo los taus en función del valor de f
plt.figure()
plt.plot(f_s, Nr_s, '-o', label='Nr')
plt.plot(f_s, Tr_s, '-o', label='Tr')
plt.plot(f_s, Tm_s, '-o', label='Tm')
plt.plot(f_s, Tpr_s, '-o', label='Tpr')
plt.xlabel('dYX2')
plt.grid()
plt.legend()
plt.show()
