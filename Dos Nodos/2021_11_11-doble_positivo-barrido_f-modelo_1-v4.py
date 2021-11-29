#Acá tomo cosas del v1 para ver cómo usar todo esto para hacer el barrido con lhs
#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import mide_biestabilidad as mb
import runge_kuta as rk
import going_up_comming_down_doble_positivo_modelo_1 as gucd
plt.ion()

#%%
#Pruebo la función que hace el going up comming down para el modelo 1
A_s, B_s, S = gucd.gucd_modelo_1(1, 0.1, 1, 1, 0.01, 0.01, 2., 3., 1000, 2, 50)
#%%
#Ploteo
plt.figure()
plt.plot(S, A_s, 'o', label='A')
plt.plot(S, B_s, '.', label='B')
plt.grid()
plt.legend()
plt.show()

#%%
#Pruebo la función que mide el área biestable
area = mb.mide_biestabilidad(A_s, S)
