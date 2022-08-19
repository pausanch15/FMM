#Hago lo mismo que en la versión 1 de este código, pero de ahora en más trabajo siempre con lo que propuso Fede y los valores e f que hallé en el barrido.

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta_estimulo as rks
from scipy import interpolate #funciones para interpolar
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
#Defino los distintos estímulos dependientes del tiempo que voy a pasar.
N_estimulo = 10000 
tiempo_max_estimulo = 100 
tiempo_estimulo = np.linspace(0,tiempo_max_estimulo,N_estimulo) 
tiempo_escalon = 2 
tiempo_entre_escalones = 20

constante = np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon)
pulso = np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon) - np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon*2)
dos_pulsos = np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon) - np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon*2) + np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon+tiempo_entre_escalones) - np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon*2+tiempo_entre_escalones)

plt.figure()
plt.plot(tiempo_estimulo, constante, label='Constante')
plt.plot(tiempo_estimulo, pulso, label='Pulso')
plt.plot(tiempo_estimulo, dos_pulsos, '--', label='Dos Pulsos')
plt.grid()
plt.legend()
plt.show()

#%%
#Los valores de f que van a ser interesantes de integrar en cada caso.
F = [0.43, 1.1, 1.7, 2.6, 3.0, 3.5]

#%%
#Defino los parámetros e integro.
estimulo = constante #El estímulo que quiero pasar en etse caso
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

plt.figure()
plt.plot(tiempo, variables[0], label='A', color='c')
plt.plot(tiempo_estimulo, estimulo, '--', label='Estímulo', color='k')
plt.xlim(0, 10)
plt.title(f'Parámetros de Fede\nk1 = {params[1]}, k2 = {params[2]}, K2 = {params[3]}, n = {params[4]}, k3 = {params[5]}')
plt.legend(); plt.grid(); plt.show()
# plt.savefig('barrido_f.pdf', dpi=300, bbox_inches='tight')

#%%
#Integro el estímulo constante para todos los f que elegí.
tiempos = []
As = []

for f in F:
    estimulo = constante #El estímulo que quiero pasar en etse caso
    condiciones_iniciales = [0]
    tiempo_min = 100
    tiempo_max = 1000
    
    k1 = 1
    k2 = f
    K2 = 0.5
    n = 3
    k3 = 1
    params = [0, k1, k2, K2, n, k3]
    
    interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
    tiempo, variables = rks.integrar(modelo, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

    tiempos.append(tiempo)
    As.append(variables[0])

# plt.rc("text", usetex=True)
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
fig.suptitle(f'Estímulo Constante', fontsize=13)
for i, ax in enumerate(axs.flatten()):
    ax.plot(tiempo_estimulo, estimulo, '--', color='k', alpha=0.35)
    ax.plot(tiempos[i], As[i], color='c')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('A')
    ax.annotate(f"f={F[i]}", (65, 0.2), fontsize=10)
plt.show()
# plt.savefig('2021_10_12-estimulo_constante.pdf', dpi=300, bbox_inches='tight')

#%%
#Ahora para el pulso
tiempos = []
As = []

for f in F:
    estimulo = pulso #El estímulo que quiero pasar en etse caso
    condiciones_iniciales = [0]
    tiempo_min = 100
    tiempo_max = 1000
    
    k1 = 1
    k2 = f
    K2 = 0.5
    n = 3
    k3 = 1
    params = [0, k1, k2, K2, n, k3]
    
    interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
    tiempo, variables = rks.integrar(modelo, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

    tiempos.append(tiempo)
    As.append(variables[0])

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(tiempo_estimulo, estimulo, '-', color='k')
    ax.plot(tiempos[i], As[i])
    if i>=3:
        ax.set_xlabel('Tiempo', fontsize=15, color='black')
    if i%3==0:
        ax.set_ylabel('A', fontsize=15, color='black')
    ax.annotate(f"$f={F[i]}$", (10, 0.2), fontsize=15)
    ax.tick_params(labelsize=15, color='black', labelcolor='black')
    ax.set_xlim(0, 20)
    ax.grid(True)
plt.tight_layout()
plt.savefig('Figuras/pulsocorto.pdf', dpi=300, bbox_inches='tight')

#%%
#Ahora para los dos pulsos
tiempos = []
As = []

for f in F:
    estimulo = dos_pulsos #El estímulo que quiero pasar en etse caso
    condiciones_iniciales = [0]
    tiempo_min = 100
    tiempo_max = 1000
    
    k1 = 1
    k2 = f
    K2 = 0.5
    n = 3
    k3 = 1
    params = [0, k1, k2, K2, n, k3]
    
    interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
    tiempo, variables = rks.integrar(modelo, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

    tiempos.append(tiempo)
    As.append(variables[0])

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
for i, ax in enumerate(axs.flatten()):
    ax.plot(tiempo_estimulo, estimulo, '-', color='k')
    ax.plot(tiempos[i], As[i])
    if i>=3:
        ax.set_xlabel('Tiempo', fontsize=15, color='black')
    if i%3==0:
        ax.set_ylabel('A', fontsize=15, color='black')
    ax.annotate(f"$f={F[i]}$", (28, 0.8), fontsize=15)
    ax.set_xlim(0, 60)
    ax.tick_params(labelsize=15, color='black', labelcolor='black')
    ax.grid(1)
    # ax.legend(fontsize=15)
plt.show()
plt.savefig('Figuras/dospulsos.pdf', dpi=300, bbox_inches='tight')

#%%
#Varío el tiempo entre pulsos para los dos pulsos aplicados al f=1.7, que corresponde a biestabilidad reversible.
N_estimulo = 10000 
tiempo_max_estimulo = 100 
tiempo_estimulo = np.linspace(0,tiempo_max_estimulo,N_estimulo) 
tiempo_escalon = 2 
tiempo_entre_escalones = np.arange(0, 20, 3.5)

plt.figure()
for t in tiempo_entre_escalones:
    dos_pulsos = np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon) - np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon*2) + np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon+t) - np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon*2+t)

    plt.plot(tiempo_estimulo, dos_pulsos, label=f'Tiempo Entre Pulsos = {t}')
    
plt.grid()
plt.legend()
plt.show()

#%%
# F = [1.6, 1.65, 1.7, 1.75, 1.8, 1.9, 2]
F = [1.7, 1.9]

for f in F:
    tiempos = []
    As = []
    estimulos = []
    
    for t in tiempo_entre_escalones:
        dos_pulsos = np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon) - np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon*2) + np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon+t) - np.ones(N_estimulo)*(tiempo_estimulo>tiempo_escalon*2+t)
        
        estimulo = dos_pulsos #El estímulo que quiero pasar en etse caso
        condiciones_iniciales = [0]
        tiempo_min = 100
        tiempo_max = 1000
        
        k1 = 1
        k2 = f #Bietsabilidad reversible
        K2 = 0.5
        n = 3
        k3 = 1
        params = [0, k1, k2, K2, n, k3]
        
        interpolar_estimulo = interpolate.interp1d(tiempo_estimulo,estimulo)
        tiempo, variables = rks.integrar(modelo, params, interpolar_estimulo, condiciones_iniciales, tiempo_max, tiempo_min)

        tiempos.append(tiempo)
        As.append(variables[0])
        estimulos.append(estimulo)

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)
    for i, ax in enumerate(axs.flatten()):
        ax.plot(tiempo_estimulo, estimulos[i], '-', color='k')
        ax.plot(tiempos[i], As[i])
        if i>=3:
            ax.set_xlabel('Tiempo', fontsize=15, color='black')
        if i%3==0:
            ax.set_ylabel('A', fontsize=15, color='black')
        ax.set_xlim(0, 50)
        ax.grid(True)
        ax.tick_params(labelsize=15, color='black', labelcolor='black')
        
    plt.tight_layout()
    plt.savefig(f'Figuras/dospulsosreversiblef={f}.pdf', dpi=300)
         
