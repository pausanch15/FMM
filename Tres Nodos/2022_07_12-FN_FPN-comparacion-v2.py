#Hace lo mismo que v1 pero obtiene una figura con el numero de picos segùn cada contador para cada sistema

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

#Cosas de matplotlib para hacer los gráficos
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('ggplot')
plt.rc("text", usetex=True)
plt.rc('font', family='serif')
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
#Defino el sistema FPN
dYX2 = 0.14
dX2 = 0.1
Ty2 = 0.21
dy2 = 0.1
TY2 = 0.3
dY2 = 0.1

#%%
#Barro en f y uso los dos contadores
f_s = np.linspace(0.7, 1.12, 20)
N_NE = []
N_E = []

for f in f_s:
    tiempo, variables, tiempo_estimulo, estimulo = fpndespf.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, f, tiempo_max=2000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=0.04, tau=200, ts=30, tb=530, condiciones_iniciales=[0.01, 0.01, 0.01])
    
    epsilon = 0.01
    X, y, Y = variables*epsilon
    
    picos_NE, altura_picos_NE, picos_E, altura_picos_E, tiempo_decaimiento = aplica_contadores(tiempo, variables*epsilon, tb=530, porcentaje_umbral=60)

    N_NE.append(len(picos_NE))
    N_E.append(len(picos_E))

#Plot
plt.figure()
plt.plot(f_s, N_NE, '-o', label='Contador No Estricto')
plt.plot(f_s, N_E, '-s', label='Contador Estricto')
plt.grid(1)
plt.xlabel('$f$', fontsize=15, color='black')
plt.ylabel('Número de Picos', fontsize=15, color='black')
plt.yticks(fontsize=15, color='black')
plt.xticks(fontsize=15, color='black')
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Figuras/contadoresbarridof.pdf', dpi=300)

#%%
#Barro en dYX2 y uso los dos contadores
del(f, dYX2, N_NE, N_E)
f = 1

dYX2_s = np.linspace(0.14, 0.26, 20)
N_NE = []
N_E = []

for dYX2 in dYX2_s:
    tiempo, variables, tiempo_estimulo, estimulo = fpndespf.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, f, tiempo_max=2000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=0.04, tau=200, ts=30, tb=530, condiciones_iniciales=[0.01, 0.01, 0.01])
    
    epsilon = 0.01
    X, y, Y = variables*epsilon
    
    picos_NE, altura_picos_NE, picos_E, altura_picos_E, tiempo_decaimiento = aplica_contadores(tiempo, variables*epsilon, tb=530, porcentaje_umbral=60)

    N_NE.append(len(picos_NE))
    N_E.append(len(picos_E))

#Plot
plt.figure()
plt.plot(dYX2_s, N_NE, '-o', label='Contador No Estricto')
plt.plot(dYX2_s, N_E, '-s', label='Contador Estricto')
plt.grid(1)
plt.xlabel('$d_{YX2}$', fontsize=15, color='black')
plt.ylabel('Número de Picos', fontsize=15, color='black')
plt.yticks(fontsize=15, color='black')
plt.xticks(fontsize=15, color='black')
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig('Figuras/contadoresbarridodyx2.pdf', dpi=300)
