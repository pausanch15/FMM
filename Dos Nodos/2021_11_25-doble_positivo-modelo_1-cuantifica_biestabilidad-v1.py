#Acá pruebo el cuantificador de biestabilidad de la tesis de Débora.
#Uso el csv que creé en el código henera_df: '2021_11_18-doble_positivo-modelo_1-ksa_2-ksb_4_6.csv'

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
plt.ion()

#%%
#Traigo el csv como dataframe
fname = '2021_11_18-doble_positivo-modelo_1-ksa_2-ksb_0_3.csv'
k_sa = float(re.findall("ksa_([^-]*)", fname)[0])
df = pd.read_csv(fname, header=[0,1], index_col=0)

#Creo el vector de inputs a partir de df
S = df.index.to_numpy()

#Y los k_sb
k_sb_s = [float(re.findall("\d.\d*", ksb)[0]) for ksb in df.columns.levels[0] if ksb != 'S']

#%%
#Elijo uno de los posibles modelos y trato de cuantificar biestabilidad en ese
#Para k_sb = 0.0 no debería haber biestabilidad
#Para k_sb = 3.0 debería haber biestabilidad

#Primero traigo A_s y B_s para k_sb = 3.0
A_s = df['k_sb=3.000']['A'].to_numpy()
B_s = df['k_sb=3.000']['B'].to_numpy()

#Voy a medir biestabilidad en A_s nomás (Ale dijo que con uno de los dos bastaba)
#Trato de hallar discontinuidad para la ida y la vuelta por separado
mitad = int(len(S)/2)

S_ida = S[:mitad]
S_vuelta = S[mitad:]

A_s_ida = A_s[:mitad]
A_s_vuelta = A_s[mitad:]

#np.diff(arr) me devuelve un array que en cada lugar tiene arr[i+1] - arr[i]
dif_ida = np.abs(np.diff(A_s_ida))
dif_vuelta = np.abs(np.diff(A_s_vuelta))

#Me fijo cuánto mayor que la media es la diferencia mayor en un caso donde sé que hay biestabilidad
print(f'Para la ida, la diferencia media es de {np.mean(dif_ida)} y la máxima es {np.max(dif_ida)}')
print(f'Para la vuelta, la diferencia media es de {np.mean(dif_vuelta)} y la máxima es {np.max(dif_vuelta)}')

#%%
#Ahora traigo A_s y B_s para k_sb = 0.0
A_s = df['k_sb=0.000']['A'].to_numpy()
B_s = df['k_sb=0.000']['B'].to_numpy()

#Voy a medir biestabilidad en A_s nomás (Ale dijo que con uno de los dos bastaba)
#Trato de hallar discontinuidad para la ida y la vuelta por separado
mitad = int(len(S)/2)

S_ida = S[:mitad]
S_vuelta = S[mitad:]

A_s_ida = A_s[:mitad]
A_s_vuelta = A_s[mitad:]

#np.diff(arr) me devuelve un array que en cada lugar tiene arr[i+1] - arr[i]
dif_ida = np.abs(np.diff(A_s_ida))
dif_vuelta = np.abs(np.diff(A_s_vuelta))

#Me fijo cuánto mayor que la media es la diferencia mayor en un caso donde sé que hay biestabilidad
print(f'Para la ida, la diferencia media es de {np.mean(dif_ida)} y la máxima es {np.max(dif_ida)}')
print(f'Para la vuelta, la diferencia media es de {np.mean(dif_vuelta)} y la máxima es {np.max(dif_vuelta)}')

#%%
#Al parece, la diferencia es de un orden de magnitud entre la diferencia media y máxima cuando hay biestabilidad.
#Uso ese criterio para decidir en cada caso si hay biestabilidad o no

#Vuelvo a trabajar para k_sb = 3.00, as{i calculo el ancho de la zona biestable y si puedo el área también
#Traigo nuevamente A_s y B_s para k_sb = 3.0
A_s = df['k_sb=3.000']['A'].to_numpy()
B_s = df['k_sb=3.000']['B'].to_numpy()

#Voy a medir biestabilidad en A_s nomás (Ale dijo que con uno de los dos bastaba)
#Trato de hallar discontinuidad para la ida y la vuelta por separado
mitad = int(len(S)/2)

S_ida = S[:mitad]
S_vuelta = S[mitad:]

A_s_ida = A_s[:mitad]
A_s_vuelta = A_s[mitad:]

#np.diff(arr) me devuelve un array que en cada lugar tiene dif[i] = arr[i+1] - arr[i]
dif_ida = np.abs(np.diff(A_s_ida))
dif_vuelta = np.abs(np.diff(A_s_vuelta))

#Me fijo cuánto mayor que la media es la diferencia mayor en un caso donde sé que hay biestabilidad
print(f'Para la ida, la diferencia media es de {np.mean(dif_ida)} y la máxima es {np.max(dif_ida)}')
print(f'Para la vuelta, la diferencia media es de {np.mean(dif_vuelta)} y la máxima es {np.max(dif_vuelta)}')

#Igualmente, en el paper de Débora usan como criterio para la discontinuidad que la diferencia en un punto es mayor que 5 veces la anterior y mayor que 1e-3

#Busco el S_on y S_off que dice la tesis
dif_ida_max = np.max(dif_ida)
dif_vuelta_max = np.max(dif_vuelta)

S_on = S_ida[np.where(dif_ida == dif_ida_max)[0][0]]
S_off = S_vuelta[np.where(dif_vuelta == dif_vuelta_max)[0][0]]

#Ploteo a ver si estoy haciendo las cosas que quiero
plt.figure()
plt.plot(S_ida, A_s_ida, 'o', label='A_s Ida')
plt.plot(S_vuelta, A_s_vuelta, '.', label='A_s Vuelta')
plt.plot(S_on, 0, 'o', label='S_on')
plt.plot(S_off, 0, 'o', label='S_off')
plt.legend()
plt.grid()

plt.show()

#%%
#Intento calcular el área de la zona biestable 
ds = S[1] - S[0]
area_ida = np.trapz(A_s_ida, dx=ds)
area_vuelta = np.trapz(A_s_vuelta, dx=ds)

area_biestable = area_vuelta - area_ida

