#Levanto el csv de las 10000 corridas, me fijo qué porcentaje dio biestable sobre el total y hago el histograma de las áreas

#Librerías
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mide_biestabilidad_v2 as mb
import runge_kuta as rk
import going_up_comming_down_doble_positivo_modelo_1 as gucd
from scipy import interpolate
import runge_kuta_estimulo as rks
plt.ion()

#%%
#Levanto el csv y veo algunos plots
fname = '2021_12_26-parametros_biestables.csv'
df = pd.read_csv(fname, index_col=0)

#Armo un array con todas las áreas biestables
areas = df.index.to_numpy()

#Me fijo el porcentaje de áreas que dio biestable
porcentaje_biestabilidad = (len(areas)*100)/10000
print(f'El porcentaje de biestabilidad es del {porcentaje_biestabilidad}%.')

#%%
#Veo algunos plots para ver más o menos como dieron
# for n in range(len(areas)):
    # if n%10 != 0: pass
    # params = df.loc[areas[n], :].to_numpy()
    # A_s, B_s, S = gucd.gucd_modelo_1(*params, 1000, 2, 50)
    # 
    # #Ploteo
    # plt.figure()
    # plt.plot(S, A_s, 'o', label='A')
    # plt.plot(S, B_s, '.', label='B')
    # plt.grid()
    # plt.legend()
    # plt.show()
#%%
#Hago el histograma de las áreas
plt.hist(areas, bins='auto', color='c', edgecolor='black', zorder=2)
plt.grid(zorder=1)

#%%
#Intento medir memoria en estos sistemas
#Para eso tengo que pasarle un estímulo escalón al sistema ya estabilizado
#Elijo algún sistema para empezar
area = 10
params = df.loc[areas[area], :].to_numpy()

#Veo cuál es, así sé con qué trabajo
A_s, B_s, S = gucd.gucd_modelo_1(*params, 1000, 2, 50)

#Ploteo
plt.figure()
plt.plot(S, A_s, 'o', label='A')
plt.plot(S, B_s, '.', label='B')
plt.grid()
plt.legend()
plt.show()

#%%
#El modelo que se va a integrar con estímulo y sin él
#wos = without stimulus
#ws = with stimulus
def modelo_wos(vars,params):
    
    S = params[0]  

    # Parámetros de síntesis de A y B
    k_sa = params[1]
    k_sb = params[2]
    K_sa = params[3]
    K_sb = params[4]

    # Parámetros de inhibición mutua
    k_ba = params[5]
    k_ab = params[6]
    K_ba = params[7]
    K_ab = params[8]

    # Variables
    A=vars[0]
    B=vars[1]

    # Sistema de ecuaciones
    dA = (S+B)*k_sa*(1-A)/(K_sa + 1-A) - k_ba*A/(K_ba+A)
    dB = A*k_sb*(1-B)/(K_sb + 1-B) - k_ab*B/(K_ab+B)
    
    return np.array([dA,dB])

def modelo_ws(vars, params, interpolar_estimulo, tiempo):
    
    S = interpolar_estimulo(tiempo) #aca lo interpola

    # Parámetros de síntesis de A y B
    k_sa = params[1]
    k_sb = params[2]
    K_sa = params[3]
    K_sb = params[4]

    # Parámetros de inhibición mutua
    k_ba = params[5]
    k_ab = params[6]
    K_ba = params[7]
    K_ab = params[8]

    # Variables
    A=vars[0]
    B=vars[1]

    # Sistema de ecuaciones
    dA = (S+B)*k_sa*(1-A)/(K_sa + 1-A) - k_ba*A/(K_ba+A)
    dB = A*k_sb*(1-B)/(K_sb + 1-B) - k_ab*B/(K_ab+B)
    
    return np.array([dA,dB])

#%%
#Integro sin estímulo
tiempo_min = 100
tiempo_max = 1000
S = .02

params = np.insert(params, 0, S) #Agrego S a params

condiciones_iniciales = [0, 0]
tiempo_wos, variables_wos = rk.integrar(modelo_wos, params, condiciones_iniciales, tiempo_max)

A_wos = variables_wos[0]
B_wos = variables_wos[1]

#Ploteo
plt.figure()
plt.plot(tiempo_wos, A_wos, 'o', label='A')
plt.plot(tiempo_wos, B_wos, '.', label='B')
plt.grid()
plt.legend(), plt.xlabel('Tiempo')
plt.show()

#%%
#Integro con estímulo
#Pruebo pasar un estímulo escalón
#Definimos el estimulo variable
N_estimulo = 10000 #resolucion para el estimulo
tiempo_max_estimulo = 1000 #tiempo hasta donde llega
tiempo_estimulo = np.linspace(0,tiempo_max_estimulo,N_estimulo) #vector de tiempo para el estimulo
tiempo_subida = 30 #Este sistema en particular, en 25 ya se estacionó
tiempo_bajada = 50

S_alto = 100
S_bajo = .2

#Estimulo, multiplico un vector de unos por uno de ceros y unos (falses y trues de la desigualdad) que me da el escalón
estimulo = S_alto*np.ones(N_estimulo)*(tiempo_estimulo>tiempo_subida)*(tiempo_estimulo<tiempo_bajada) 
estimulo = estimulo + S_bajo*np.ones(N_estimulo)*((tiempo_estimulo<tiempo_subida)+(tiempo_estimulo>tiempo_bajada))

#Vemos el estímulo
plt.figure()
plt.plot(tiempo_estimulo, estimulo, label='Estímulo')
plt.legend()
plt.grid()
plt.show()

#Integro
condiciones_iniciales = [0,0]
tiempo_min = tiempo_bajada + 20
tiempo_max = 1000

interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
tiempo_ws, variables_ws = rks.integrar(modelo_ws, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

A_ws = variables_ws[0]
B_ws = variables_ws[1]

#Grafico
plt.figure(figsize=(10, 7))
plt.subplot(211)
plt.plot(tiempo_estimulo[tiempo_estimulo<tiempo_ws[-1]], estimulo[tiempo_estimulo<tiempo_ws[-1]], label='Estímulo', color='k')
plt.xlabel('Tiempo'); plt.ylabel('Estímulo'); plt.grid()
plt.subplot(212)
plt.plot(tiempo_ws, A_ws, label='A', color='c')
plt.plot(tiempo_ws, B_ws, label='B', color='g')
plt.xlabel('Tiempo'); plt.ylabel('A, B'); plt.legend(); plt.grid(); plt.show()
#%%
#Calculo la memoria
#Interpolamos, extrapolamo
A_wos_int = interpolate.interp1d(tiempo_wos, A_wos, fill_value="extrapolate")
B_wos_int = interpolate.interp1d(tiempo_wos, B_wos, fill_value="extrapolate")

A_ws_int = interpolate.interp1d(tiempo_ws, A_ws, fill_value="extrapolate")
B_ws_int = interpolate.interp1d(tiempo_ws, B_ws, fill_value="extrapolate")

#Construyo un array de tiempos en los cuales evaluar lo que interpolé y lo evalúo. La memoria solo tiene sentido una vez que saqué el escalón, así que y empieza en tiempo_bajada
t = np.linspace(tiempo_bajada, tiempo_max, 1000)

A_wos_ev = A_wos_int(t)
B_wos_ev = B_wos_int(t)

A_ws_ev = A_ws_int(t)
B_ws_ev = B_ws_int(t)

#Calculo memoria
memoria_A = A_ws_ev - A_wos_ev
memoria_B = B_ws_ev - B_wos_ev

#Ploteo
plt.figure()
plt.plot(t, memoria_A, label='Memoria A')
plt.plot(t, memoria_B, label='Memoria B')
plt.show(), plt.grid(), plt.legend()


