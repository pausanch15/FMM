#Esto o retomo casi 6 mesos después y me doy cuenta de que hay varias cosas que no estaba haciendo bien. Intento recalcular la memoria con todo lo que aprendí hasta el momento.
#Ya no hago todo el cálculo en este código sino que me construyo una función que lo haga.

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk #El que no acepta estímulos dependientes de t
import runge_kuta_estimulo as rks #El que sí acepta estímulos dependientes de t
from scipy import interpolate
import mide_memoria_un_nodo as mm
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
#Los valores de f que van a ser interesantes de integrar en cada caso.
F = [0.43, 1.1, 1.7, 2.6, 3.0, 3.5]

#Valores de los otros parámetros
# S = .01
k1 = 1
K2 = 0.5
n = 3
k3 = 1

#Calculo la memoria
M = []
for f in F:
    # memoria = mm.mide_memoria(k2=f, k1=1, K2=0.5, k3=1, n=3, S_alto=3, S_bajo=0.01, plot_estimulo=True, plot_memoria=True)
    memoria = mm.mide_memoria(k2=f, k1=1, K2=0.5, k3=1, n=3, S_alto=3, S_bajo=0.01, plot_estimulo=False, plot_memoria=False)
    # plt.savefig(f'2022_03_28-memoria_f={f}.pdf', dpi=300, bbox_inches='tight')
    M.append(memoria)
    del(memoria)

#%%    
#Hago la figura que dijo Ale
plt.figure() 
# plt.plot(tiempo_estimulo, estimulo, 'k--', label='Estímulo', alpha=0.45)
for i in range(len(F)):
    plt.plot(M[i], 'o', label=f'f={F[i]}')
plt.legend(loc='upper right')
plt.grid()
plt.xlim(-0.2, 0.2)
plt.ylabel('Contador de Memoria')
plt.xlabel('Tiempo')
# plt.savefig(f'2020_10_22-memoria_muchos_f_t=15.pdf', dpi=300, bbox_inches='tight')
