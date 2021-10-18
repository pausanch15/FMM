#Acá intento poner en práctica el cotador de memoria que sugieren en Defining a memory quantifier for cell signaling systems Quantifying memory in cell signaling Defining networks topologies that can retain memories from past transient stimulation.
#Para probarlo voy a usar un caso que ya sé que tiene memoria para algunos parámetros de feedback: k1=1, K2=0.5, n=3, k3=1, s_max=1
#Los parámetros de feedback que voy a recorrer son k2=f=0.43, 1.1, 1.7, 2.6, 3.0, 3.5. Sé que hay memoria a partir de 0.43, y que debería verse cómo el contador aumenta entre 0.43 y 2.6.
#Para integrar el sistema con un estímulo contante nulo voy a usar la versión de Runge Kuta que no acepta estímulos dependientes del tiempo, entiendo que es menos costoso omputacionalmente.
#Para el futuro, quizas no sería mala idea incluir estas funciones dentro de otra, la cual elija según un parámetro si aceptar estímulos dependientes del tiempo o no.

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk #El que no acepta estímulos dependientes de t
import runge_kuta_estimulo as rks #El que sí acepta estímulos dependientes de t
from scipy import interpolate
plt.ion()

#%%
#El modelo que se va a integrar con estímulo y sin él
#wos = without stimulus
#ws = with stimulus
def modelo_wos(vars,params):
    
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
    dA = k1*S*(1-A) + k2*(1-A)*A**n/(K2+A)**n - k3*A
    
    return np.array([dA])

def modelo_ws(vars, params, interpolar_estimulo, tiempo):
    
    S = interpolar_estimulo(tiempo) #aca lo interpola

    # Parámetros
    # S = params[0]
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
#Defino el pulso corto
N_estimulo = 10000 
tiempo_max_estimulo = 100 
tiempo_estimulo = np.linspace(0,tiempo_max_estimulo,N_estimulo) 
tiempo_escalon = 2 

estimulo = np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon) - np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon*2)

#%%
#Los valores de f que van a ser interesantes de integrar en cada caso.
F = [0.43, 1.1, 1.7, 2.6, 3.0, 3.5]

#%%
#Integro
A_wos = [] #Respuestas sin estímulo
t_wos = [] #Tiempos sin estímulo

A_ws = [] #Respuestas con estímulo
t_ws = [] #Tiempos con estímulo

condiciones_iniciales = [0]
tiempo_min = 10
tiempo_max = 1000
S = .01
k1 = 1
K2 = 0.5
n = 3
k3 = 1

for f in F:
    k2 = f

    #Primero integro sin el estimulo
    params = [S, k1, k2, K2, n, k3]
    tiempo, variables = rk.integrar(modelo_wos, params, condiciones_iniciales, tiempo_max)
    A_wos.append(variables[0])
    t_wos.append(tiempo)
    del(tiempo, variables, params)

    #Ahora con el estímulo
    params = [0, k1, k2, K2, n, k3]
    interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
    tiempo, variables = rks.integrar(modelo_ws, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)
    A_ws.append(variables[0])
    t_ws.append(tiempo)
    del(tiempo, variables, params, interpolar_estimulo)

#%%
#Me fijo el largo de las listas obtenidas
for a_wos, a_ws in zip(A_wos, A_ws):
    print(len(a_wos))
    print(len(a_ws))
    print()

#%%
#Es un re problema: cada vector de A sin estimulo tiene distinto largo que el A con estímulo que le corresponda. Tampoco tienen la misma longitud entre A que no se corresponden.
#Voy a intentar interpolar estas listas, y despues evaluar en un array de un largo específico
#Preguntar si no sería mejor idea directamente integrar con Runge Kutas de paso no variable.
#Algo que pasa es que al tener tiempos tan distintos como vimos antes, hay vectores que llegan hasta tipo 50 y otros hasta 200... no alcanza entonces solo con interpolar, hay que extrapolar las interpolaciones de aquellos vectores de 50 puntos a valores más grandes para no perder información.
#Uso par esto el parámetro fill_value="extrapolate".
A_wos_int = [] #La lista de a_wos interpolados
A_ws_int = [] #La lista de a_ws interpolados

for awos, aws, twos, tws in zip(A_wos, A_ws, t_wos, t_ws):
    awos, aws, twos, tws = np.array(awos), np.array(aws), np.array(twos), np.array(tws)
    A_wos_int.append(interpolate.interp1d(twos, awos, fill_value="extrapolate"))
    A_ws_int.append(interpolate.interp1d(tws, aws, fill_value="extrapolate"))

#Construyo un array de tiempos en los cuales evaluar lo que interpolé y lo evalúo
t = np.linspace(0, 15, 1000)

A_wos_ev = [] #La lista de a_wos interpolados evaluados en t
A_ws_ev = [] #La lista de a_ws interpolados evaluados en t

for awos_int, aws_int in zip(A_wos_int, A_ws_int):
    A_wos_ev.append(awos_int(t))
    A_ws_ev.append(aws_int(t))

#%%
#Ploteo para comparar los resultados originales con los interpolados y extrapolados.
for i in range(len(A_wos)):
    # plt.rc("text", usetex=True)
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    fig.suptitle(f'Valor de Feedback f={F[i]}', fontsize=16)
    
    axs[0].plot(t_wos[i], A_wos[i], label='Sin Estímulo', color='k')
    axs[0].plot(t_ws[i], A_ws[i], label='Con Estímulo', color='c')
    axs[0].set_title('Originales')
    axs[0].legend()
    axs[0].grid()
    axs[0].set_ylabel('A')

    axs[1].plot(t, A_wos_ev[i], label='Sin Estímulo', color='k')
    axs[1].plot(t, A_ws_ev[i], label='Con Estímulo', color='c')
    axs[1].set_title('Interpolados')
    axs[1].legend()
    axs[1].grid()
    axs[1].set_xlabel('Tiempo')
    axs[1].set_ylabel('A')

    # plt.savefig(f'2020_10_18-interpolacion_f={F[i]}.pdf', dpi=300, bbox_inches='tight')


#%%
#Todo esto era para tener arrays de igual largo y poder definir el contador de memoria, así que trato de hacer eso.
M = [] #Lista en la que voy a guardar los contadores

for awos, aws in zip(A_wos_ev, A_ws_ev):
    M.append(aws - awos)

#Hago la figura que dijo Ale
plt.figure()
plt.plot(tiempo_estimulo, estimulo, 'k--', label='Estímulo', alpha=0.45)
for i in range(len(F)):
    plt.plot(t, M[i], label=f'f={F[i]}')
plt.legend()
plt.grid()
plt.xlim(0, 15)
plt.ylabel('Contador de Memoria')
plt.xlabel('Tiempo')
plt.savefig(f'2020_10_18-memoria_muchos_f_t=15.pdf', dpi=300, bbox_inches='tight')

