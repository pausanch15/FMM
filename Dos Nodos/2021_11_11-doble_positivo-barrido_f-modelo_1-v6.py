#Acá tomo algunos de los parámetros para los que encontré biestabilidad, ploteo los resultados y hago algunos análisis.
#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import mide_biestabilidad as mb
import runge_kuta as rk
import going_up_comming_down_doble_positivo_modelo_1 as gucd
import latin_hypercube_sampling as lhs
from time import time
import pandas as pd
plt.ion()

#Traigo el csv con los resultados
fname = '2021_11_29-parametros_biestables.csv'
df = pd.read_csv(fname, index_col=0)

#Armo un array con todas las áreas biestables
areas = df.index.to_numpy()

#Elijo algún conjunto de parámetros e integro el modelo para ellos
n = 10
params = df.loc[areas[n], :].to_numpy()
A_s, B_s, S = gucd.gucd_modelo_1(*params, 1000, 2, 50)

#Ploteo
plt.figure()
plt.plot(S, A_s, 'o', label='A')
plt.plot(S, B_s, '.', label='B')
plt.grid()
plt.legend()
plt.show()
