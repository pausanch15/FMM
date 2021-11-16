#Acá trato de levantar el df creado en genera-df
#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
plt.ion()

#%%
#Traigo el csv como dataframe
df = pd.read_csv('2021_11_16-doble_positivo-modelo_1-ksa_2-ksb_4_6.csv')

#Creo el vector de inputs a partir de df
s = df.iloc[1:, 0].to_numpy()

#Y los k_sb
k_sb_s = [float(re.findall("\d.\d*", ksb)[0]) for i, ksb in enumerate(list(df.iloc[0, 1:].to_dict().keys())) if i%2 == 0]
