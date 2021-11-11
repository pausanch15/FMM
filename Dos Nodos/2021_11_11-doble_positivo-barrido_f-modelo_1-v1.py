#Acá empiezo a hacer el barrido en los parámetros de este modelo para tratar de encontrar la zona biestable.
#Primer modelo: Pongo los parámetros de cuánto A afecta a B y cuánto B afecta a A en el término positivo de cada ecuación
#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta_estimulo as rks
import runge_kuta as rk
from scipy import interpolate #funciones para interpolar
plt.ion()

#%%
#Defino el modelo
def modelo(vars, params):

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

#%%
#Preparo los parámetros para empezar a buscar la zona biestable
#Primero voy a barrer solamente en los parámetros de feedback: k_sa y k_sb
condiciones_iniciales = [0, 0]

# k_sa = 1
# k_sb = 10
K_sa = 1
K_sb = 0.1

k_ba = 1
k_ab = 1
K_ba = 0.01
K_ab = 0.01

k_sa_s = np.linspace(1, 5, 5)
k_sb_s = np.linspace(8, 13, 6)

tiempo_max = 1000
S_max = 1 #El máximo input que voy a usar
pasos = 50

#Empiezo el recorrido
A_s = []
B_s = []

for ksa in k_sa_s:
    k_sa = ksa
    print()
    print(f'Va por k_sa={ksa}.\n')
    for ksb in k_sb_s:
        lista_condiciones_iniciales = [[0, 0]] 
        # tiempo_max = 1000
        k_sb = ksb
        A_conv = []
        B_conv = []
        # S_max = 1 #El máximo input que voy a usar
        # pasos = 50

        #Ida
        S_ida = np.linspace(0, S_max, pasos) #Los inputs que voy barriendo
        for i, s in enumerate(S_ida):
            condiciones_iniciales = lista_condiciones_iniciales[-1]
            params = [s, k_sa, k_sb, K_sa, K_sb, k_ba, k_ab, K_ba, K_ab]
            tiempo, variables = rk.integrar(modelo, params, condiciones_iniciales, tiempo_max)
            A_conv.append(variables[0][-1])
            B_conv.append(variables[1][-1])
            lista_condiciones_iniciales.append([variables[0][-1], variables[1][-1]])
            del(tiempo, variables)

        #Vuelta
        S_vuelta = np.linspace(S_max, 0, pasos) #Los inputs que voy barriendo: los de antes pero al revés
        for i, s in enumerate(S_vuelta):
            condiciones_iniciales = lista_condiciones_iniciales[-1]
            params = [s, k_sa, k_sb, K_sa, K_sb, k_ba, k_ab, K_ba, K_ab]
            tiempo, variables = rk.integrar(modelo, params, condiciones_iniciales, tiempo_max)
            A_conv.append(variables[0][-1])
            B_conv.append(variables[1][-1])
            lista_condiciones_iniciales.append([variables[0][-1], variables[1][-1]])
            del(tiempo, variables)

        #Guardo en A_s y B_s
        A_s.append(A_conv)
        B_s.append(B_conv)
        del(A_conv, B_conv, lista_condiciones_iniciales) 

#Junto los inputs de ida y vuelta
S = np.concatenate((S_ida, S_vuelta))

#%%
#Ploteo
for j, ksa in enumerate(k_sa_s):
    # fig, axs = plt.subplots(2, 3, figsize=(20, 10), sharex=True, sharey=True)
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    fig.suptitle(f'Barrido: k_sa = {ksa}, k_sb entre {k_sb_s[0]} y {k_sb_s[-1]}.', fontsize=20)
    for i, ax in enumerate(axs.flatten()):
        ax.plot(S[:int(len(S)/2)], A_s[6*j+i][:int(len(A_s[6*j+i])/2)], 'o', fillstyle='none', label='A: Ida', color='indianred')
        ax.plot(S[int(len(S)/2):], A_s[6*j+i][int(len(A_s[6*j+i])/2):], '.', label='A: Vuelta', color='indianred')
        ax.plot(S[:int(len(S)/2)], B_s[6*j+i][:int(len(B_s[6*j+i])/2)], 'o', fillstyle='none', label='B: Ida', color='royalblue')
        ax.plot(S[int(len(S)/2):], B_s[6*j+i][int(len(B_s[6*j+i])/2):], '.', label='B: Vuelta', color='royalblue')
        ax.legend(loc='lower right')
        ax.set_xlabel('Input')
        ax.set_ylabel('A')
        # ax.annotate(f"f={F[i]:.2}", (0.8, 0.3), fontsize=10)
    plt.show()

#Hay biestabilidad para estos valores a partir de k_sa=2
#Sería una buena idea hacer un barrido más fino para 1 < k_sa < 3
