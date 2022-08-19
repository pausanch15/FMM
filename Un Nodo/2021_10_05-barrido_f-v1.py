#Acá realizamos el barrido en distintas intensidades de feedback, siguiendo la idea del box de A Positive-Feedback-Based Bistable ‘Memory Module’ That Governs A Cell Fate Decision
#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk
plt.ion()

#Cosas de matplotlib para hacer los gráficos
import matplotlib as mpl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('ggplot')
plt.rc("text", usetex=True)
plt.rc('font', family='serif')
plt.ion()

#%%
'''
Camino de Ida
Ponemos una condición inicial y un input, dejamos que evolucione hasta un estado estacionario y nos lo guardamos. Para el siguiente, tomamos como condición inicial el estado estacionario alcanzado en el paso anterior, corremos el imput un poquito, dejamos que evoluciones y guardamos el estado estacionario al que llega. Así susecivamente, hasta recorrer todo el eje de input.

Camino de Vuelta
Una vez que llegamos al máximo valor de imput propuesto, empezamos a decrecerlo usando a cada paso el estado estacionario previo como condición inicial, así hasta llegar de nuevo al primer input.

Si tenemos un sistema biestable, esperamos que el camino de ida sea distinto al de vuelta.

En este código, el input es S, por lo que tenemos que barrer muchos S.

Vamos a hacer esto para feedbacks de distintas intensidades, donde 0 es que no hay feedback, y a medida que se lo aumenta, cada vez su acción pesa más.
'''
#%%
#Defino el mismo modelo que usa el paper.
#El k2 de este modelo es el f del paper.
def modelo(vars, params):
    # Parámetros
    S = params[0]
    k1 = params[1]
    k2 = params[2]
    K2 = params[3]
    n = params[4]
    k3 = params[5]

    # Variables
    A=vars[0]

    # Sistema de ecuaciones
    # estimulo + feedback - inactivacion
    dA = k1*S*(1-A) + k2*(1-A)*A**n/((K2**n)+(A**n)) - k3*A
    
    return np.array([dA])

#%%
#Los valores de k2 que voy a ir recorriendo
F = np.linspace(0, 0.2, 10)
print(f'En total recorremos {len(F)} valores de f.')

#%%
#Arranco las idas y vuelta en inputs para cada intensidad de feedback
A_s = [] #La lista con las lista donde guardo los A que obtengo par cada f

for f in F:
    lista_condiciones_iniciales = [0] #Acá voy a ir guardando el estado estacionario del paso anterior. Arranco en 0.
    tiempo_max = 1000
    k1 = 1
    k2 = f
    K2 = 1
    n = 5
    k3 = 0.01
    A_conv = []
    S_max = 0.02 #El máximo input que voy a usar
    pasos = 50

    #Ida
    S_ida = np.linspace(0, S_max, pasos) #Los inputs que voy barriendo
    for i, s in enumerate(S_ida):
        condiciones_iniciales = [lista_condiciones_iniciales[-1]]
        params = [s, k1, k2, K2, n, k3]
        tiempo, variables = rk.integrar(modelo, params, condiciones_iniciales, tiempo_max)
        A_conv.append(variables[0][-1])
        lista_condiciones_iniciales.append(variables[0][-1])
        del(tiempo, variables)

    #Vuelta
    S_vuelta = np.linspace(S_max, 0, pasos) #Los inputs que voy barriendo: los de antes pero al revés
    for i, s in enumerate(S_vuelta):
        condiciones_iniciales = [lista_condiciones_iniciales[-1]]
        params = [s, k1, k2, K2, n, k3]
        tiempo, variables = rk.integrar(modelo, params, condiciones_iniciales, tiempo_max)
        A_conv.append(variables[0][-1])
        lista_condiciones_iniciales.append(variables[0][-1])
        del(tiempo, variables)

    #Guardo el A_conv de este f en la lista A_s
    A_s.append(A_conv)
    del(A_conv)

#Junto los inputs de ida y vuelta
S = np.concatenate((S_ida, S_vuelta))

#%%
#La figura que intenta ser como la del paper
plt.rc("text", usetex=True)
fig, axs = plt.subplots(2, 5, figsize=(40, 7), sharex=True, sharey=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(S[:int(len(S)/2)], A_s[i][:int(len(A_s[i])/2)], 'o', fillstyle='none', label='Ida', color='indianred')
    ax.plot(S[int(len(S)/2):], A_s[i][int(len(A_s[i])/2):], 'o', fillstyle='none', label='Vuelta', color='royalblue')
    ax.legend(loc='center right')
    ax.set_xlabel('Input')
    ax.set_ylabel('A')
    ax.annotate(f"f={F[i]:.2}", (0.013, 0), fontsize=10)
plt.show()

# plt.savefig('barrido_f.pdf', dpi=300, bbox_inches='tight')

#%%
#Hago todo esto para los parámetros que propone Fede. Así hallamos valores de k2 para los cuales no hay biestabilidad, para cuales si hay: leve o irreversible
#Para esto, Fede usó el mismo modelo que el paper
#Los valores de k2 que voy a ir recorriendo
f_min = 0
f_max = 3.5
F = np.linspace(f_min, f_max, 15)
print(f'En total recorremos {len(F)} valores de f.')

#%%
#Arranco las idas y vuelta en inputs para cada intensidad de feedback
A_s = [] #La lista con las lista donde guardo los A que obtengo par cada f

for f in F:
    lista_condiciones_iniciales = [0] #Acá voy a ir guardando el estado estacionario del paso anterior. Arranco en 0.
    tiempo_max = 1000
    k1 = 1
    k2 = f
    K2 = 0.5
    n = 3
    k3 = 1
    A_conv = []
    S_max = 1 #El máximo input que voy a usar
    pasos = 100

    #Ida
    S_ida = np.linspace(0, S_max, pasos) #Los inputs que voy barriendo
    for i, s in enumerate(S_ida):
        condiciones_iniciales = [lista_condiciones_iniciales[-1]]
        params = [s, k1, k2, K2, n, k3]
        tiempo, variables = rk.integrar(modelo, params, condiciones_iniciales, tiempo_max)
        A_conv.append(variables[0][-1])
        lista_condiciones_iniciales.append(variables[0][-1])
        del(tiempo, variables)

    #Vuelta
    S_vuelta = np.linspace(S_max, 0, pasos) #Los inputs que voy barriendo: los de antes pero al revés
    for i, s in enumerate(S_vuelta):
        condiciones_iniciales = [lista_condiciones_iniciales[-1]]
        params = [s, k1, k2, K2, n, k3]
        tiempo, variables = rk.integrar(modelo, params, condiciones_iniciales, tiempo_max)
        A_conv.append(variables[0][-1])
        lista_condiciones_iniciales.append(variables[0][-1])
        del(tiempo, variables)

    #Guardo el A_conv de este f en la lista A_s
    A_s.append(A_conv)
    del(A_conv)

#Junto los inputs de ida y vuelta
S = np.concatenate((S_ida, S_vuelta))

#%%
#La figura que intenta ser como la del paper
fig, axs = plt.subplots(3, 5, figsize=(30, 10), sharex=True, sharey=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(S[:int(len(S)/2)], A_s[i][:int(len(A_s[i])/2)], 'o', fillstyle='none', label='Ida')
    ax.plot(S[int(len(S)/2):], A_s[i][int(len(A_s[i])/2):], '.', label='Vuelta')
    # ax.legend(loc='lower right', fontsize=15)
    ax.legend(fontsize=15)
    ax.grid(1)
    if i%5==0:
        ax.set_ylabel('A', fontsize=15, color='black')
    if i>=10:
        ax.set_xlabel('Input', fontsize=15, color='black')
    ax.annotate(f"$f=${F[i]:.2}", (0.8, 0.3), fontsize=15)
    ax.tick_params(labelsize=15, color='black', labelcolor='black')

#Guardo como tiempo_max_f_min_f_max.pdf
plt.tight_layout()
plt.savefig(f'Figuras/barridof.pdf', dpi=300, bbox_inches='tight')
