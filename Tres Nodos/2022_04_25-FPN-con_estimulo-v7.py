#Hago un barrido en distintos valores de f
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
#Defino una función para hallar los taus 
def encuentra_taus(X):
    picos_todos, caracteristicas_todos = find_peaks(X, height=X)
    altura_picos_todos = caracteristicas_todos['peak_heights']

    #Picos de las reververaciones
    tb = 1030
    umbral_memoria = 0.001 #Diferencia entre la amplitud de los picos durante el escalón y los que consideramos reververaciones
    i_tb = np.where(tiempo>tb)[0][0]
    picos, caracteristicas = find_peaks(X[i_tb:], height=X[i_tb]+umbral_memoria)
    altura_picos = caracteristicas['peak_heights']
    #Empiezo a calcular las cantidades que propone el paper
    t_primer_pico = tiempo[picos_todos[0]] #Tiempo del primer pico de todos
    t_primer_pico_rev = tiempo[i_tb:][picos[0]] #Tiempo del primer pico de las reververaciones
    t_ultimo_pico_rev = tiempo[i_tb:][picos[-1]] #Tiempo del ultimo pico de las reververaciones

    Nr = len(picos)
    Tr = t_ultimo_pico_rev - t_primer_pico_rev
    Tm = t_ultimo_pico_rev - tiempo[i_tb]
    Tpr = Tm - Tr

    return [Nr, Tr, Tm, Tpr]

#%%
#Empiezo el barrido en los parámetros de feedback positivo y negativo
#Barrido en feedback positivo
dYX2 = 0.18 
dX2 = 0.1 
Ty2 = 0.21 
dy2 = 0.1 
TY2 = 0.3 
dY2 = 0.1 
altura_escalon = 0.1
epsilon = 0.01

f_Nr_s, f_Tr_s, f_Tm_s, f_Tpr_s = [], [], [], []

f_s = np.linspace(0.88, 1.11, 50)

for f in f_s:
    tiempo, variables, tiempo_estimulo, estimulo = fpndesf.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, f, tiempo_max=3000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=altura_escalon, tau=200, ts=30, tb=1030, condiciones_iniciales=[0.01, 0.01, 0.01])

    X, y, Y = variables*epsilon
    Nr, Tr, Tm, Tpr = encuentra_taus(X)
    f_Nr_s.append(Nr)
    f_Tr_s.append(Tr)
    f_Tm_s.append(Tm)
    f_Tpr_s.append(Tpr)

    #Plots
    # plt.figure()
    # plt.plot(tiempo, X, label='X')
    # plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
    # plt.grid()
    # plt.legend()
    # plt.show()
    # # plt.savefig(f'resultados/barrido_f_{f}.pdf')
    # plt.close()

#%%
#Ploteo los taus en función del valor de f
plt.figure()
plt.plot(f_s, f_Nr_s, '-o', label=r'$N_R$')
plt.plot(f_s, f_Tr_s, '-o', label=r'$\tau_R$')
plt.plot(f_s, f_Tm_s, '-o', label=r'$\tau_M$')
plt.plot(f_s, f_Tpr_s, '-o', label=r'$\tau_{PR}$')
plt.grid(1)
plt.legend(fontsize=15)
plt.yticks(fontsize=15, color='black')
plt.xticks(fontsize=15, color='black')
plt.xlabel('$f$', fontsize=15, color='black')
plt.tight_layout()
plt.savefig('Figuras/taubarrf.pdf', dpi=300)

#%%
#Barrido en feedback negativo
f = 1
dX2 = 0.1 
Ty2 = 0.21 
dy2 = 0.1 
TY2 = 0.3 
dY2 = 0.1 
altura_escalon = 0.1
epsilon = 0.01

d_Nr_s, d_Tr_s, d_Tm_s, d_Tpr_s = [], [], [], []

dYX2_s = np.linspace(0.15, 0.22, 50)

for dYX2 in dYX2_s:
    tiempo, variables, tiempo_estimulo, estimulo = fpndesf.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, f, tiempo_max=3000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=altura_escalon, tau=200, ts=30, tb=1030, condiciones_iniciales=[0.01, 0.01, 0.01])

    X, y, Y = variables*epsilon
    Nr, Tr, Tm, Tpr = encuentra_taus(X)
    d_Nr_s.append(Nr)
    d_Tr_s.append(Tr)
    d_Tm_s.append(Tm)
    d_Tpr_s.append(Tpr)

    #Plots
    # plt.figure()
    # plt.plot(tiempo, X, label='X')
    # plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
    # plt.grid()
    # plt.legend()
    # plt.show()
    # # plt.savefig(f'resultados/barrido_dYX2_{dYX2}.pdf')
    # plt.close()

#%%
#Ploteo los taus en función del valor de dYX2
plt.figure()
plt.plot(dYX2_s, d_Nr_s, '-o', label=r'$N_R$')
plt.plot(dYX2_s, d_Tr_s, '-o', label=r'$\tau_R$')
plt.plot(dYX2_s, d_Tm_s, '-o', label=r'$\tau_M$')
plt.plot(dYX2_s, d_Tpr_s, '-o', label=r'$\tau_{PR}$')
plt.grid(1)
plt.legend(fontsize=15, loc=7)
plt.yticks(fontsize=15, color='black')
plt.xticks(fontsize=15, color='black')
plt.xlabel('$d_{YX2}$', fontsize=15, color='black')
plt.tight_layout()
plt.savefig('Figuras/taubarrdyx2.pdf', dpi=300)

#%%
#Hago la figura con los dos ejes, como sugirió Fede
#Primero para el barrido en f
#Creating figure
fig = plt.figure(figsize=(8, 5))
 
#Plotting taus
ax = fig.add_subplot(111)
ax.plot(f_s, f_Tr_s, '-o', label=r'$\tau_R$')
ax.plot(f_s, f_Tm_s, '-o', label=r'$\tau_M$')
ax.plot(f_s, f_Tpr_s, '-o', label=r'$\tau_{PR}$')
 
#Creating Twin axes for Nr
ax2 = ax.twinx()
ax2.plot(f_s, f_Nr_s, '-o', label=r'$N_R$', color='#777777')
 
#Adding legend
ax.legend(loc='lower left', fontsize=15)
ax2.legend(loc='upper right', fontsize=15)
 
#adding grid
ax.grid(1)
 
#Adding labels
ax.set_xlabel(r"$f$", fontsize=15, color='black')
ax.set_ylabel(r"Tiempo", fontsize=15, color='black')
ax2.set_ylabel(r"Número de Picos", fontsize=15, color='black')

#Pongo el estilo estrecho, y guardo
ax.tick_params(labelsize=15, color='black', labelcolor='black')
ax2.tick_params(labelsize=15, color='black', labelcolor='black')
plt.tight_layout()
plt.savefig('Figuras/taubarrf.pdf', dpi=300)

#Ahora para el barrido en dYX2
#Creating figure
fig = plt.figure(figsize=(8, 5))
 
#Plotting taus
ax = fig.add_subplot(111)
ax.plot(dYX2_s, d_Tr_s, '-o', label=r'$\tau_R$')
ax.plot(dYX2_s, d_Tm_s, '-o', label=r'$\tau_M$')
ax.plot(dYX2_s, d_Tpr_s, '-o', label=r'$\tau_{PR}$')
 
#Creating Twin axes for Nr
ax2 = ax.twinx()
ax2.plot(dYX2_s, d_Nr_s, '-o', label=r'$N_R$', color='#777777')
 
#Adding legend
ax.legend(loc='lower left', fontsize=15)
ax2.legend(loc='upper right', fontsize=15)
 
#adding grid
ax.grid(1)
 
#Adding labels
ax.set_xlabel(r"$d_{YX2}$", fontsize=15, color='black')
ax.set_ylabel(r"Tiempo", fontsize=15, color='black')
ax2.set_ylabel(r"Número de Picos", fontsize=15, color='black')

#Pongo el estilo estrecho, y guardo
ax.tick_params(labelsize=15, color='black', labelcolor='black')
ax2.tick_params(labelsize=15, color='black', labelcolor='black')
plt.tight_layout()
plt.savefig('Figuras/taubarrdyx2.pdf', dpi=300)

#%%
#Agrego la figura que muestra Tm/tau paraambos barridos
tau = 200

f_Tm_s = np.array(f_Tm_s)
d_Tm_s = np.array(d_Tm_s)

f_Tm_s_tau = f_Tm_s/tau
d_Tm_s_tau = d_Tm_s/tau

plt.figure()
plt.plot(f_s, f_Tm_s_tau, '-o', label=r'$\tau_M/\tau$')
plt.grid(1)
plt.legend(fontsize=15)
plt.yticks(fontsize=15, color='black')
plt.xticks(fontsize=15, color='black')
plt.xlabel('$f$', fontsize=15, color='black')
plt.tight_layout()
plt.savefig('Figuras/taumsobretaubarrf.pdf', dpi=300)

plt.figure()
plt.plot(dYX2_s, d_Tm_s_tau, '-o', label=r'$\tau_M/\tau$')
plt.grid(1)
plt.legend(fontsize=15)
plt.yticks(fontsize=15, color='black')
plt.xticks(fontsize=15, color='black')
plt.xlabel('$d_{YX2}$', fontsize=15, color='black')
plt.tight_layout()
plt.savefig('Figuras/taumsobretaubarrdyx2.pdf', dpi=300)
