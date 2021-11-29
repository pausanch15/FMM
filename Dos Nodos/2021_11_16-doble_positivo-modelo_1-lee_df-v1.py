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
fname = '2021_11_18-doble_positivo-modelo_1-ksa_2-ksb_0_3.csv'
k_sa = float(re.findall("ksa_([^-]*)", fname)[0])
df = pd.read_csv(fname, header=[0,1], index_col=0)

#Creo el vector de inputs a partir de df
S = df.index.to_numpy()

#Y los k_sb
k_sb_s = [float(re.findall("\d.\d*", ksb)[0]) for ksb in df.columns.levels[0] if ksb != 'S']

#%%
#El gráfico
#Ploteo
fig, axs = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(20, 10))
fig.suptitle(f'Barrido: k_sa={k_sa}, k_sb entre {k_sb_s[0]} y {k_sb_s[-1]}.', fontsize=20)
for i, (ax, name_ksb) in enumerate(zip(axs.flatten(), df.columns.levels[0])):
    A_s = df[name_ksb]['A']
    B_s = df[name_ksb]['B']
    ax.plot(S, A_s, 'o', fillstyle='none', label='A', color='indianred')
    ax.plot(S, B_s, 'o', label='B', color='royalblue')
    ax.legend(loc='upper left')
    ax.set_xlabel('Input')
    ax.set_ylabel('A')
    ax.annotate(f"k_sb={k_sb_s[i]:.2}", (1.2, 0.2), fontsize=10)
plt.show()

#Guardo la figura
plt.savefig('2021_11_18-doble_positivo-modelo_1-ksa_2-ksb_0_3.pdf', dpi=300, bbox_inches='tight')
