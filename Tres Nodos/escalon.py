#Intento hacer el escalón

#%%
#Librerías
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

#%%
#Eje del tiempo
resolucion = 1000
ti = 0
tf = 100
tiempo = np.linspace(ti, tf, resolucion)

#Eje de estímulo
S_min = 0
S_max = 5
S = np.ones(resolucion)

#Hago el escalón
ts = 30
tb = 70
S[0:ts] = S_min
S[ts:tb] = S_max
S[tb:] = S_min

#Ahora la bajada exponencial
tau = 10
bajada = S_max*np.exp(-tiempo[tb:]/tau)
S[tb:] = bajada

# caida = (S_max/(np.exp(-tb)-np.exp(-tf))) * (np.exp(-tiempo)-np.exp(-tf))
# S[tb:] = caida[tb:]

# tau = 10
# bajada = S_max*np.exp(tb/tau)*np.exp(-tiempo/tau)
# S[tb:] = bajada[tb:]

# tau = 10
# bajada = S_max*np.exp(-tiempo/tau)
# S[tb:] = bajada[tb:]

#Ploteo
plt.plot(tiempo, S)
# plt.plot(tiempo, caida)
plt.show()
