#Es lo mismo del que dice v1, pero cambio la condición inicial.
#En el anterior lo que pasaba es que aún sin el estímulo, partíamos de una condición inicial que no correspondía al equilibrio, y por lo tanto el sistema evolucionaba hasta un equilibrio... y en el medio de eso lo perturbábamos. Era demasiado, y se perdía la gracia de ver el efecto del estímulo solamente.
#Para arreglar esto, primero corro una vez el sistema de la condición inicial anterior, lo dejo evolucionar hasta que estacione, y me guardo este valor. Luego repito todo lo del código anterior pero como condición inicial paso este valor del estado estacionario.

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
#Integro primero sin el estímulo para ver a qué valor estaciona
# condiciones_iniciales = [0]
tiempo_min = 100
tiempo_max = 1000
S = .01
k1 = 1
K2 = 0.5
n = 3
k3 = 1

equilibrios_wos = [] #Me guardo los valores al que cada uno estaciona sin est
equilibrios_ws = [] #Me guardo los valores al que cada uno estaciona con est

for f in F:
    k2 = f

    #Primero integro sin el estimulo
    condiciones_iniciales = [0]
    params = [S, k1, k2, K2, n, k3]
    tiempo, variables = rk.integrar(modelo_wos, params, condiciones_iniciales, tiempo_max)
    equilibrios_wos.append(variables[0][-1])
    del(tiempo, variables, params, condiciones_iniciales)

    #Ahora con el estímulo
    condiciones_iniciales = [0]
    params = [0, k1, k2, K2, n, k3]
    interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
    tiempo, variables = rks.integrar(modelo_ws, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)
    equilibrios_ws.append(variables[0][-1])
    del(tiempo, variables, params, interpolar_estimulo, condiciones_iniciales)
#Lo que está pasando es que los casos biestables convergen a la solución de mayor valor, y por eso arrancan de más arriba.
#Antes, cuando partía del 0 en todos los casos, nunca llegaba al equilibrio, por lo que no veía nada de esto, y todos arrancaban forzadamente del 0.
#Sigo así, y después pregunto si debería corregir algo, o cambiar algo.

#%%
#Con estos valores de condiciones iniciales, repito lo del código anterior.
A_wos = [] #Respuestas sin estímulo
t_wos = [] #Tiempos sin estímulo

A_ws = [] #Respuestas con estímulo
t_ws = [] #Tiempos con estímulo

# condiciones_iniciales = [0]
tiempo_min = 100
tiempo_max = 1000
S = .01
k1 = 1
K2 = 0.5
n = 3
k3 = 1

for f, eq_wos, eq_ws in zip(F, equilibrios_wos, equilibrios_ws):
    k2 = f

    #Primero integro sin el estimulo
    condiciones_iniciales = [eq_wos]
    params = [S, k1, k2, K2, n, k3]
    tiempo, variables = rk.integrar(modelo_wos, params, condiciones_iniciales, tiempo_max)
    A_wos.append(variables[0])
    t_wos.append(tiempo)
    del(tiempo, variables, params, condiciones_iniciales)

    #Ahora con el estímulo
    condiciones_iniciales = [eq_ws]
    params = [0, k1, k2, K2, n, k3]
    interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
    tiempo, variables = rks.integrar(modelo_ws, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)
    A_ws.append(variables[0])
    t_ws.append(tiempo)
    del(tiempo, variables, params, interpolar_estimulo, condiciones_iniciales)

#%%
#Interpolamos, extrapolamos y obtenemos la figura
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

#Ploteo para comparar los resultados originales con los interpolados y extrapolados.
for i in range(len(A_wos)):
    plt.rc("text", usetex=True)
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    fig.suptitle(f'Valor de Feedback f={F[i]}', fontsize=16)
    
    axs[0].plot(t_wos[i], A_wos[i], label='Sin Estímulo', color='k')
    axs[0].plot(t_ws[i], A_ws[i], label='Con Estímulo', color='c')
    axs[0].set_title('Originales')
    axs[0].legend()
    axs[0].grid()
    axs[0].set_ylabel('A')
    axs[0].set_xlim(0, 15)

    axs[1].plot(t, A_wos_ev[i], label='Sin Estímulo', color='k')
    axs[1].plot(t, A_ws_ev[i], label='Con Estímulo', color='c')
    axs[1].set_title('Interpolados')
    axs[1].legend()
    axs[1].grid()
    axs[1].set_xlabel('Tiempo')
    axs[1].set_ylabel('A')
    axs[1].set_xlim(0, 15)

    plt.savefig(f'2020_10_22-interpolacion_f={F[i]}.pdf', dpi=300, bbox_inches='tight')

#%%
#Pruebo el contador de memoria.
#Para los biestables espero que de cualquier cosa.
M = [] #Lista en la que voy a guardar los contadores

for awos, aws in zip(A_wos_ev, A_ws_ev):
    M.append(aws - awos)
    for i, ti in enumerate(t):
        if ti < tiempo_escalon*2:
            M[-1][i] = 'nan'

#Hago la figura que dijo Ale
plt.figure()
plt.plot(tiempo_estimulo, estimulo, 'k--', label='Estímulo', alpha=0.45)
for i in range(len(F)):
    plt.plot(t, M[i], label=f'f={F[i]}')
plt.legend(loc='upper left')
plt.grid()
plt.xlim(0, 15)
plt.ylabel('Contador de Memoria')
plt.xlabel('Tiempo')
plt.savefig(f'2020_10_22-memoria_muchos_f_t=15.pdf', dpi=300, bbox_inches='tight')
