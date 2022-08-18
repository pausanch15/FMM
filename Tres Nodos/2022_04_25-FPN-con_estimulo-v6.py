#Para los casos en los que sí hay reververaciones, intento calcular las cantidades que propone el paper de Mitra

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk
import integra_FPN_desacoplado_parametro_feedback as fpndesf
from scipy import interpolate
import runge_kuta_estimulo as rks
from scipy.signal import find_peaks
plt.ion()

#Cosas de matplotlib para hacer los gráficos
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('ggplot')
plt.rc("text", usetex=True)
plt.rc('font', family='serif')
plt.ion()

#%%
#Encontrar los picos de las reververaciones debería ser exctamente igual que antes
#Lo que cambia es que hay que pedir que solo se tengan en cuenta los mayores que los que tiene el sistema durante el escalón.
dYX2 = 0.18 
dX2 = 0.1 
Ty2 = 0.21 
dy2 = 0.1 
TY2 = 0.3 
dY2 = 0.1 
altura_escalon = 0.1
f = 1
epsilon = 0.01

tiempo, variables, tiempo_estimulo, estimulo = fpndesf.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, f, tiempo_max=3000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=altura_escalon, tau=200, ts=30, tb=1030, condiciones_iniciales=[0.01, 0.01, 0.01])

X, y, Y = variables*epsilon

#Todos los picos
picos_todos, caracteristicas_todos = find_peaks(X, height=X)
altura_picos_todos = caracteristicas_todos['peak_heights']

#Picos de las reververaciones
tb = 1030
umbral_memoria = 0.001 #Diferencia entre la amplitud de los picos durante el escalón y los que consideramos reververaciones
i_tb = np.where(tiempo>tb)[0][0]
picos, caracteristicas = find_peaks(X[i_tb:], height=X[i_tb]+umbral_memoria)
altura_picos = caracteristicas['peak_heights']

#%%
#Ploteo los picos de reverberaciones
plt.figure()
plt.plot(tiempo, X, label='X')
# plt.plot(tiempo[picos_todos], altura_picos_todos, '.', label='Todos los Picos')
plt.plot(tiempo[i_tb:][picos], altura_picos, 'o', label='Memoria')
plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
plt.grid(1)
plt.legend(fontsize=15)
plt.yticks(fontsize=15, color='black')
plt.xticks(fontsize=15, color='black')
plt.xlabel('Tiempo', fontsize=15, color='black')
plt.tight_layout()
plt.savefig('Figuras/picosrev.pdf', dpi=300)

#%%
#Empiezo a calcular las cantidades que propone el paper
t_primer_pico = tiempo[picos_todos[0]] #Tiempo del primer pico de todos
t_primer_pico_rev = tiempo[i_tb:][picos[0]] #Tiempo del primer pico de las reververaciones
t_ultimo_pico_rev = tiempo[i_tb:][picos[-1]] #Tiempo del ultimo pico de las reververaciones

Nr = len(picos)
Tr = t_ultimo_pico_rev - t_primer_pico_rev
Tm = t_ultimo_pico_rev - tiempo[i_tb]
Tpr = Tm - Tr

#%%
#Plots
plt.figure()
plt.plot(tiempo, X, label='X')
plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')

#Plot Tr
plt.hlines(-0.01, t_primer_pico_rev, t_ultimo_pico_rev, label=r'$\tau_R$', color='#348ABD')
plt.vlines(t_primer_pico_rev, altura_picos[0], -0.01, linestyle='dashed', color='#348ABD')
plt.vlines(t_ultimo_pico_rev, altura_picos[-1], -0.01, linestyle='dashed', color='#348ABD')

#Plot Tm
plt.hlines(-0.02, tiempo[i_tb], t_ultimo_pico_rev, label=r'$\tau_M$', color='#988ED5')
plt.vlines(tiempo[i_tb], altura_escalon, -0.02, linestyle='dashed', color='#988ED5')
plt.vlines(t_ultimo_pico_rev, altura_picos[-1], -0.02, linestyle='dashed', color='#988ED5')

#Plot Tpr
plt.hlines(-0.03, tiempo[i_tb], t_primer_pico_rev, label=r'$\tau_{PR}$', color='#777777')
plt.vlines(tiempo[i_tb], altura_escalon, -0.03, linestyle='dashed', color='#777777')
plt.vlines(t_primer_pico_rev, altura_picos[0], -0.03, linestyle='dashed', color='#777777')

plt.grid(1)
plt.legend(fontsize=15)
plt.yticks(fontsize=15, color='black')
plt.xticks(fontsize=15, color='black')
plt.xlabel('Tiempo', fontsize=15, color='black')
plt.tight_layout()
plt.savefig('Figuras/tausrev.pdf', dpi=300)
