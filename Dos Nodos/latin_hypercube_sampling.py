import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
#%%
def lhs(n_params,n_sets,random_seed=0):
    # Seteo la semilla random para poder generar los mismos params al azar
    random.seed(random_seed)
    np.random.seed(random_seed)
    # defino el ancho de filas y columnas
    ancho = 1/n_sets
    # aca guardo los params
    params_lhs = np.zeros([n_sets,n_params])
    # para cada param, para cada set elijo un valor random dentro de la celda
    for i_sets in range(n_sets):
        for i_params in range(n_params):
            params_lhs[i_sets,i_params]=(random.random()+i_sets)*ancho
    # mezclo todos los sets, en cada param para sacar la correlacion en celdas
    for i_params in range(n_params):
        np.random.shuffle(params_lhs[:,i_params])
    return params_lhs
