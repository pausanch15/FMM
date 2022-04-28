#Intento hacer el escalón

#%%
#Librerías
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

#%%
#Eje del tiempo
resolucion = 10000
ti = 0
tf = 500
tiempo = np.linspace(ti, tf, resolucion)

#Eje de estímulo
S_min = 0
S_max = 5
S = np.ones(resolucion)

#Hago el escalon
ts = int(0.3*resolucion)
tb = int(0.7*resolucion)
S[0:ts] = S_min
S[ts:tb] = S_max
S[tb:] = S_min

#Ahora la bajada exponencial
tau = 10
bajada = S_max*np.exp((-tiempo)/tau)
S[tb:] = bajada[:resolucion-tb]

#Ploteo
# plt.plot(tiempo, S)
# plt.show()

#%%
#Armo la función que devuelve el escalon
def escalon(resolucion, ti, tf, S_min, S_max, tau, ts, tb):    
    #Eje del tiempo
    tiempo = np.linspace(ti, tf, resolucion)
    
    #Eje de estímulo
    S = np.ones(resolucion)
    
    #Hago el escalon
    ts = int(ts*(1/tf)*resolucion)
    tb = int(tb*(1/tf)*resolucion)
    S[0:ts] = S_min
    S[ts:tb] = S_max
    S[tb:] = S_min
    
    #Ahora la bajada exponencial
    bajada = S_max*np.exp((-tiempo)/tau)
    S[tb:] = bajada[:resolucion-tb]

    #Devuelvo el escalon
    return tiempo, S

#Pruebo la funcion
tiempo_estimulo, estimulo = escalon(resolucion=1000, ti=0, tf=237, S_min=0, S_max=3, tau=10, ts=30, tb=50)

plt.figure()
plt.plot(tiempo_estimulo, estimulo)
plt.title('Con la Funcion')
plt.show()
