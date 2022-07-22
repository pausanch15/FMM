#Ahora trabajamos con sistemas de dos nodos, y feedback negativo.
#Volvemos a ver para qué estímulos hay memoria.
#Esto lo escribió Fede. Yo solo cambio un poco las figuras.

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta_estimulo as rks
from scipy import interpolate #funciones para interpolar
plt.ion()

#%%
#Definimos el modelo: Feedback negativo de 2 nodos A y B. A recibe el estímulo
# le paso al modelo el interpolador del estimulo para que lo interpole a cada paso
def modelo(vars, params, interpolar_estimulo, tiempo):

    S = interpolar_estimulo(tiempo) #aca lo interpola    

    # Parámetros de síntesis de A y B
    k_sa = params[0]
    k_sb = params[1]
    K_sa = params[2]
    K_sb = params[3]

    # Parámetros de inhibición mutua
    k_ba = params[4]
    k_ab = params[5]
    K_ba = params[6]
    K_ab = params[7]

    # Variables
    A=vars[0]
    B=vars[1]

    # Sistema de ecuaciones
    dA = S*k_sa*(1-A)/(K_sa + 1-A) - k_ba*B*A/(K_ba+A)
    dB =   k_sb*(1-B)/(K_sb + 1-B) - k_ab*A*B/(K_ab+B) 
    
    return np.array([dA,dB])

#%%
#Definimos el estimulo variable
N_estimulo = 10000 #resolucion para el estimulo
tiempo_max_estimulo = 1000 #tiempo hasta donde llega
tiempo_estimulo = np.linspace(0,tiempo_max_estimulo,N_estimulo) #vector de tiempo para el estimulo
tiempo_subida = 10
tiempo_bajada = 30

S_alto = 2
S_bajo = .2

#estimulo, multiplico un vector de unos por uno de ceros y unos (falses y trues de la desigualdad) que me da el escalón
estimulo = S_alto*np.ones(N_estimulo)*(tiempo_estimulo>tiempo_subida)*(tiempo_estimulo<tiempo_bajada) 
estimulo = estimulo + S_bajo*np.ones(N_estimulo)*((tiempo_estimulo<tiempo_subida)+(tiempo_estimulo>tiempo_bajada))

#Vemos el estímulo
plt.plot(tiempo_estimulo, estimulo, label='Estímulo')
plt.legend()
plt.grid()
plt.show()

#%%
#Integro para el estímulo escalón
#Primero, para un estímulo que SÍ recuerda
S_alto = 2
S_bajo = .2

estimulo = S_alto*np.ones(N_estimulo)*(tiempo_estimulo>tiempo_subida)*(tiempo_estimulo<tiempo_bajada) 
estimulo = estimulo + S_bajo*np.ones(N_estimulo)*((tiempo_estimulo<tiempo_subida)+(tiempo_estimulo>tiempo_bajada))

condiciones_iniciales = [0,0]
tiempo_min = tiempo_bajada
tiempo_max = 1000

k_sa = 1
k_sb = 1
K_sa = 1
K_sb = 1

k_ba = 1
k_ab = 1
K_ba = 0.01
K_ab = 0.01

params = [k_sa, k_sb, K_sa, K_sb, k_ba, k_ab, K_ba, K_ab]

interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
tiempo, variables = rks.integrar(modelo, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

#Grafico, veo que el estímulo cambia el equilibrio del sistema biestable
#Después del estímulo, la dinámica no regresa
plt.figure(figsize=(10, 7))
plt.subplot(211)
plt.plot(tiempo_estimulo[tiempo_estimulo<tiempo[-1]], estimulo[tiempo_estimulo<tiempo[-1]], label='Estímulo', color='k')
plt.xlabel('Tiempo'); plt.ylabel('Estímulo'); plt.grid()
plt.subplot(212)
plt.plot(tiempo,variables[0], label='A', color='c')
plt.plot(tiempo,variables[1], label='B', color='g')
plt.xlabel('Tiempo'); plt.ylabel('A, B'); plt.legend(); plt.grid(); plt.show()
# plt.savefig('2021_10_27-dos_nodos-con_memoria.pdf', dpi=300, bbox_inches='tight')

#Ahora para un estímulo que NO recuerda
#Después del estímulo, la dinámica sí regresa
S_alto = 1.5
S_bajo = .2

estimulo = S_alto*np.ones(N_estimulo)*(tiempo_estimulo>tiempo_subida)*(tiempo_estimulo<tiempo_bajada) 
estimulo = estimulo + S_bajo*np.ones(N_estimulo)*((tiempo_estimulo<tiempo_subida)+(tiempo_estimulo>tiempo_bajada))

interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
tiempo, variables = rks.integrar(modelo, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

plt.figure(figsize=(10, 7))
plt.subplot(211)
plt.plot(tiempo_estimulo[tiempo_estimulo<tiempo[-1]],estimulo[tiempo_estimulo<tiempo[-1]], label='Estímulo', color='k')
plt.xlabel('Tiempo'); plt.ylabel('Estímulo'); plt.grid()
plt.subplot(212)
plt.plot(tiempo,variables[0], label='A', color='c')
plt.plot(tiempo,variables[1], label='B', color='g')
plt.xlabel('Tiempo'); plt.ylabel('A, B'); plt.legend(); plt.grid(); plt.show()
# plt.savefig('2021_10_27-dos_nodos-sin_memoria.pdf', dpi=300, bbox_inches='tight')

#%%
#Hago un barrido de estímulos y calculo la memoria de cada uno
#Defino los estímulos para el barrido
N_barrido = 10
barrido_estimulos = np.linspace(1.5,2,N_barrido)
S_bajo = 0.2

lista_estimulos_con = []
for i_barrido in range(N_barrido):
    S_alto = barrido_estimulos[i_barrido]
    estimulo = S_alto*np.ones(N_estimulo)*(tiempo_estimulo>tiempo_subida)*(tiempo_estimulo<tiempo_bajada) 
    estimulo = estimulo + S_bajo*np.ones(N_estimulo)*((tiempo_estimulo<tiempo_subida)+(tiempo_estimulo>tiempo_bajada))
    lista_estimulos_con.append(estimulo)

lista_estimulos_sin = []
for i_barrido in range(N_barrido):
    estimulo = S_bajo*np.ones(N_estimulo)
    lista_estimulos_sin.append(estimulo)

#El siguiente plot me dice que lista_tiempos no esta definida... y no la encuentro
#Si lo corro exactamente igual que en colab me pasa lo mismo (esperable, pero valia la pena probar)
# fig, axs = plt.subplots(2, 5, figsize=(20, 10), sharex=True, sharey=True)
# fig.suptitle(f'Estímulos', fontsize=20)
# for i, ax in enumerate(axs.flatten()):
    # tiempo = lista_tiempos[i]
    # estimulo_con = lista_estimulos_con[i][tiempo_estimulo<tiempo[-1]]
    # estimulo_sin = lista_estimulos_sin[i][tiempo_estimulo<tiempo[-1]]
    # ax.plot(tiempo_estimulo[tiempo_estimulo<tiempo[-1]], estimulo_con, color = (0,i/N_barrido,0), lw=3)
    # ax.plot(tiempo_estimulo[tiempo_estimulo<tiempo[-1]], estimulo_sin, color = (0,0,i/N_barrido), lw=3)

#%%
#Integro para cada estímulo y su control
lista_tiempos_con = []
lista_variables_con = []
for i_barrido in range(N_barrido):
    estimulo = lista_estimulos_con[i_barrido]
    interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
    tiempo, variables = rks.integrar(modelo, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)
    lista_tiempos_con.append(tiempo)
    lista_variables_con.append(variables)

lista_tiempos_sin = []
lista_variables_sin = []
for i_barrido in range(N_barrido):
  tiempo_min = lista_tiempos_con[i_barrido][-1]
  tiempo_max = lista_tiempos_con[i_barrido][-1]
  estimulo = lista_estimulos_sin[i_barrido]
  interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
  tiempo, variables = rks.integrar(modelo, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)
  lista_tiempos_sin.append(tiempo)
  lista_variables_sin.append(variables)

#%%
#Grafico
#Hay un umbral a partir del cual el estímulo corre al sistema de equilibrio y a partir de allí "recuerda"
fig, axs = plt.subplots(2, 5, figsize=(20, 10), sharex=True, sharey=True)
fig.suptitle(f'Barrido en Estímulos: Cambia la altura del escalón', fontsize=20)
for i_barrido, ax in enumerate(axs.flatten()):
    tiempo = lista_tiempos_con[i_barrido]
    A = lista_variables_con[i_barrido][0]
    B = lista_variables_con[i_barrido][1]
    ax.plot(tiempo, A, color='c', label='A')
    ax.plot(tiempo, B, color='g', label='B')
    ax.annotate(f'Altura = {np.max(lista_estimulos_con[i_barrido]):.2f}', (30., 0.15))
    ax.legend()
    ax.grid()
# plt.savefig('2021_10_27-dos_nodos-barrido_estimulo.pdf', dpi=300, bbox_inches='tight')

#%%
#Calculo la memoria de cada set del barrido
def calcular_memoria(variables_con,variables_sin,tiempo_con,tiempo_sin):
    tiempo_interpol = np.linspace(0,tiempo_con[-1],10000)
    interpolar_con = interpolate.interp1d(tiempo_con,variables_con)
    interpolar_sin = interpolate.interp1d(tiempo_sin,variables_sin)
    vars_con_interp = interpolar_con(tiempo_interpol)
    vars_sin_interp = interpolar_sin(tiempo_interpol)
    memoria = vars_con_interp - vars_sin_interp
    return tiempo_interpol,memoria

lista_tiempos_memoria = []
lista_memorias = []
for i_barrido in range(N_barrido):
    tiempo_con = lista_tiempos_con[i_barrido]
    tiempo_sin = lista_tiempos_sin[i_barrido]
    variables_con = lista_variables_con[i_barrido]
    variables_sin = lista_variables_sin[i_barrido]
    tiempo, memoria = calcular_memoria(variables_con,variables_sin,tiempo_con,tiempo_sin)
    lista_tiempos_memoria.append(tiempo)
    lista_memorias.append(memoria)

#Grafico
fig, axs = plt.subplots(2, 5, figsize=(20, 10), sharex=True, sharey=True)
fig.suptitle(f'Memoria en A y B: Cambia la altura del escalón', fontsize=20)
for i_barrido, ax in enumerate(axs.flatten()):
    tiempo = lista_tiempos_memoria[i_barrido]
    memoria = lista_memorias[i_barrido]
    ax.plot(tiempo, memoria[0], color='c', label='A')
    ax.plot(tiempo, memoria[1], color='g', label='B')
    ax.vlines(tiempo_bajada, -1, 1, color='r')
    ax.set_ylabel('Memoria'); ax.set_xlabel('Tiempo')
    ax.legend(loc='lower right'); ax.grid()
# plt.savefig('2021_10_27-dos_nodos-memoria.pdf', dpi=300, bbox_inches='tight')
