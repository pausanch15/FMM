#Acá pruebo estimular al sistema con distintos imputs (no necesariamente constantes) y ver cómo evolucionan en el tiempo
#Para poder pasar un vector dependiente de tiempo como estímulo a Runge Kuta de paso variable, hay que modificar el modelo con el que veníamos trabajando.
#Corregimos el modelo pasándole el vector de estímulo y el de tiempos (pasarle la curva estímulo vs tiempo), y en cada paso interpola la curva al punto de tiempo de ese paso. Entonces calculamos el primer tiempo con RK, y habiendo interpolado antes el estimulo con el vector de tiempo original, evaluamos la función que salió de la interpolación en el tiempo del RK.

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta_estimulo as rks
from scipy import interpolate #funciones para interpolar
plt.ion()

#%%
#Le paso al modelo el interpolador del estimulo para que lo interpole a cada paso
def modelo(vars, params, interpolar_estimulo, tiempo):
    
    S = interpolar_estimulo(tiempo) #aca lo interpola

    # Parámetros
#    S = params[0]
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
#Defino el estímulo variable: para arrancar uso uno que se prende en un tiempo y permanece prendido. Lo ploteo para ver cómo es.
N_estimulo = 10000 #resolucion para el estimulo
tiempo_max_estimulo = 100 #tiempo hasta donde llega
tiempo_estimulo = np.linspace(0,tiempo_max_estimulo,N_estimulo) #vector de tiempo para el estimulo
tiempo_escalon = 2 #lugar donde esta el escalon

estimulo = np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon) 

# plt.rc("text", usetex=True)
plt.figure()
plt.plot(tiempo_estimulo, estimulo, 'k')
plt.grid()
plt.show()

#%%
#Integro para el stímulo escalón. Primero uso los parámetros que propone Fede. Más adelante pruebo con los que propone Juan (los del Ferrel corregidos).
condiciones_iniciales = [0]
tiempo_min = 10
tiempo_max = 1000

k1 = 1
k2 = 10
K2 = 0.5
n = 3
k3 = 1
params = [0, k1, k2, K2, n, k3]

interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
tiempo, variables = rks.integrar(modelo, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

# plt.rc("text", usetex=True)
plt.figure()
plt.plot(tiempo, variables[0], label='A', color='c')
plt.plot(tiempo_estimulo, estimulo, label='Estímulo', color='k')
plt.xlim(0, 10)
plt.title(f'Parámetros de Fede\nk1 = {params[1]}, k2 = {params[2]}, K2 = {params[3]}, n = {params[4]}, k3 = {params[5]}')
plt.legend(); plt.grid(); plt.show()

#Guardo como k2_n_constante.pdf
plt.savefig(f'{k2}_{n}_constante.pdf', dpi=300, bbox_inches='tight')

#%%
#Pruebo con valores sacados de lo que propone Juan.
#Uso un f (k2) para el cual no hay biestabilidad, otro para el que sí la hay y otro para el cual se vuelve irreversible. 0.022, 0.11 y 0.2 respectivamente.
condiciones_iniciales = [0]
tiempo_min = 100
tiempo_max = 1000

k1 = 1
K2 = 1
n = 5
k3 = 0.01

F = [0.022, 0.11, 0.2]
for f in F:
    k2 = f
    params = [0, k1, k2, K2, n, k3]
    
    interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
    tiempo, variables = rks.integrar(modelo, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

    # plt.rc("text", usetex=True)
    plt.figure()
    plt.plot(tiempo, variables[0], label='A', color='c')
    plt.plot(tiempo_estimulo, estimulo, label='Estímulo', color='k')
    plt.title(f'Parámetros de Juan\nk1 = {params[1]}, k2 = {params[2]}, K2 = {params[3]}, n = {params[4]}, k3 = {params[5]}')
    plt.legend(); plt.grid(); plt.show()

#%%
#Pruebo ahora con un pulso que se prende por un rato y después se apaga
N_estimulo = 10000 
tiempo_max_estimulo = 100 
tiempo_estimulo = np.linspace(0,tiempo_max_estimulo,N_estimulo) 
tiempo_escalon = 2

estimulo = np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon) - np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon*2)

# plt.rc("text", usetex=True)
plt.figure()
plt.plot(tiempo_estimulo, estimulo, 'k')
plt.grid()
plt.show()

#%%
#Pruebo con los parámetros de Fede
condiciones_iniciales = [0]
tiempo_min = 10
tiempo_max = 1000

k1 = 1
k2 = 10
K2 = 0.5
n = 3
k3 = 1
params = [0, k1, k2, K2, n, k3]

interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
tiempo, variables = rks.integrar(modelo, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

# plt.rc("text", usetex=True)
plt.figure()
plt.plot(tiempo, variables[0], label='A', color='c')
plt.plot(tiempo_estimulo, estimulo, label='Estímulo', color='k')
plt.xlim(0, 10)
plt.title(f'Parámetros de Fede\nk1 = {params[1]}, k2 = {params[2]}, K2 = {params[3]}, n = {params[4]}, k3 = {params[5]}')
plt.legend(); plt.grid(); plt.show()

#Guardo como k2_n_pulso.pdf
plt.savefig(f'{k2}_{n}_pulso.pdf', dpi=300, bbox_inches='tight')

#%%
#Pruebo ahora con dos pulsos consecutivos que se prenden por un rato y después se apagan
N_estimulo = 10000 
tiempo_max_estimulo = 100 
tiempo_estimulo = np.linspace(0,tiempo_max_estimulo,N_estimulo) 
tiempo_escalon = 2 
tiempo_entre_escalones = 20

estimulo = np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon) - np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon*2) + np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon+tiempo_entre_escalones) - np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon*2+tiempo_entre_escalones)

plt.figure()
plt.plot(tiempo_estimulo, estimulo, 'k')
plt.grid()
plt.show()

#%%
#Pruebo con los parámetros de Fede
condiciones_iniciales = [0]
tiempo_min = 100
tiempo_max = 1000

k1 = 1
k2 = 10
K2 = 0.5
n = 3
k3 = 1
params = [0, k1, k2, K2, n, k3]

interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
tiempo, variables = rks.integrar(modelo, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

# plt.rc("text", usetex=True)
plt.figure()
plt.plot(tiempo, variables[0], label='A', color='c')
plt.plot(tiempo_estimulo, estimulo, label='Estímulo', color='k')
# plt.xlim(0, 10)
plt.title(f'Parámetros de Fede\nk1 = {params[1]}, k2 = {params[2]}, K2 = {params[3]}, n = {params[4]}, k3 = {params[5]}')
plt.legend(); plt.grid(); plt.show()

#Guardo como k2_n_dosestimulos.pdf
plt.savefig(f'{k2}_{n}_dosestimulos.pdf', dpi=300, bbox_inches='tight')
