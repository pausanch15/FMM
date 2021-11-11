#Acá voy a tratar de hacer los dos modelos de doble feedback positivo que estuvimos hablando.
#Primer modelo: Pongo los parámetros de cuánto A afecta a B y cuánto B afecta a A en el término positivo de cada ecuación

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta_estimulo as rks
from scipy import interpolate #funciones para interpolar
plt.ion()

#%%
#Definimos el estimulo variable
N_estimulo = 10000 #resolucion para el estimulo
tiempo_max_estimulo = 1000 #tiempo hasta donde llega
tiempo_estimulo = np.linspace(0,tiempo_max_estimulo,N_estimulo) #vector de tiempo para el estimulo
tiempo_subida = 10
tiempo_bajada = 30

S_alto = 2
S_bajo = .2

estimulo = S_alto*np.ones(N_estimulo)*(tiempo_estimulo>tiempo_subida)*(tiempo_estimulo<tiempo_bajada)
estimulo = estimulo + S_bajo*np.ones(N_estimulo)*((tiempo_estimulo<tiempo_subida)+(tiempo_estimulo>tiempo_bajada))

#Vemos el estímulo
# plt.plot(tiempo_estimulo, estimulo, label='Estímulo')
# plt.legend()
# plt.grid()
# plt.show()

#%%
#Defino el modelo
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
    dA = (S+B)*k_sa*(1-A)/(K_sa + 1-A) - k_ba*A/(K_ba+A)
    dB = A*k_sb*(1-B)/(K_sb + 1-B) - k_ab*B/(K_ab+B)
    
    return np.array([dA,dB])

#Integro
condiciones_iniciales = [0, 0]
tiempo_min = tiempo_bajada
tiempo_max = 1000

k_sa = 1
k_sb = 10
K_sa = 1
K_sb = 0.1

k_ba = 1
k_ab = 1
K_ba = 0.01
K_ab = 0.01

params = [k_sa, k_sb, K_sa, K_sb, k_ba, k_ab, K_ba, K_ab]

interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
tiempo, variables = rks.integrar(modelo, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

# plt.figure(figsize=(10, 7))
plt.figure()
plt.subplot(211)
plt.plot(tiempo_estimulo[tiempo_estimulo<tiempo[-1]], estimulo[tiempo_estimulo<tiempo[-1]], label='Estímulo', color='k')
plt.xlabel('Tiempo'); plt.ylabel('Estímulo'); plt.grid()
plt.subplot(212)
plt.plot(tiempo,variables[0], label='A', color='c')
plt.plot(tiempo,variables[1], label='B', color='g')
plt.xlabel('Tiempo'); plt.ylabel('A, B'); plt.legend(); plt.grid(); plt.show()
