#Este código tiene la función que recibe un csv y devuelve el área biestable de cada sistema en caso de ser biestable
#En caso de ser monoestable no devuelve nada
#Da la opción de imprimir si se trata de un sistema biestable o no

#Librerías
import numpy as np
import pandas as pd
import re

def mide_biestabilidad(csv, p=False):
    '''
    csv = string con el nombre del archivo con los modelos a evaluar
    p = print. Si p=False (dafault), la función no imprime nada. Si p=True, imprime si se trata de un caso monoestable o biestable para cada modelo del csv
    '''
    #Traigo el csv como dataframe
    fname = csv
    k_sa = float(re.findall("ksa_([^-]*)", fname)[0])
    df = pd.read_csv(fname, header=[0,1], index_col=0)
    
    #Creo el vector de inputs a partir de df
    S = df.index.to_numpy()
    
    #%%
    #Lo que debería después ser una función
    mitad = int(len(S)/2)
    
    S_ida = S[:mitad]
    S_vuelta = S[mitad:]
    
    ds = S[1] - S[0]
    
    for ksb in df.columns.levels[0]:    
        #Rescato A_s y B_s
        A_s = df[ksb]['A'].to_numpy()
        B_s = df[ksb]['B'].to_numpy()
    
        #Como criterio voy a medir biestabilidad en A_s
        #Separo entonces en ida y vuelta
        A_s_ida = A_s[:mitad]
        A_s_vuelta = A_s[mitad:]
    
        #Calculo los arrays de las diferencias punto a punto
        dif_ida = np.abs(np.diff(A_s_ida))
        dif_vuelta = np.abs(np.diff(A_s_vuelta))
    
        #Me fijo donde está la mayor diferencia: su valor y su índice
        max_dif_ida = np.max(dif_ida)
        ind_max_dif_ida = np.where(dif_ida == max_dif_ida)[0][0]
        
        max_dif_vuelta = np.max(dif_vuelta)
        ind_max_dif_vuelta = np.where(dif_vuelta == max_dif_vuelta)[0][0]
        
        #Comparo con 5 veces la diferencia anterior
        #Acá decido si hay discontinuidad o no
        #Si hay, calculo el área biestable y la devuelvo
        if (max_dif_ida > 5*dif_ida[ind_max_dif_ida-1]) and (max_dif_ida > 1e-3):
            if (max_dif_vuelta > 5*dif_vuelta[ind_max_dif_vuelta-1]) and (max_dif_vuelta > 1e-3):
                area_ida = np.trapz(A_s_ida, dx=ds)
                area_vuelta = np.trapz(A_s_vuelta, dx=ds)
                
                area_biestable = area_vuelta - area_ida

                if p==True:
                    print(f'Para k_sa = {k_sa}, k_sb = {ksb} el área biestable es de {area_biestable}.')
                return area_biestable
                del(A_s, B_s, A_s_ida, A_s_vuelta, dif_ida, dif_vuelta, max_dif_ida, ind_max_dif_ida, max_dif_vuelta, ind_max_dif_vuelta, area_ida, area_vuelta, area_biestable)
    
        #Si no hay discontinuidad, que me diga que para este ksb el sistema es monoestable
        else:
            if p==True:
                print(f'Para k_sa = {k_sa}, k_sb = {ksb} el sistema es monoestable.')

            del(A_s, B_s, A_s_ida, A_s_vuelta, dif_ida, dif_vuelta, max_dif_ida, ind_max_dif_ida, max_dif_vuelta, ind_max_dif_vuelta)
