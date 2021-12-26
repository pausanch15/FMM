#Hace lo mismo que el otro pero filtra por ancho, alto y área de biestabilidad

#Librerías
import numpy as np
import pandas as pd
import re

#Función para cuando los datos estan en csv
def mide_biestabilidad_csv(csv, p=False):
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

    #Creo la lista en donde voy a ir guardando las áreas biestables
    areas_biestables = []
    
    #La función
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
        
        #Defino S_on y S_off y el ancho W
        S_on = S_ida[ind_max_dif_ida]
        S_off = S_vuelta[ind_max_dif_vuelta]
        W = S_on - S_off

        #Empiezo a filtrar
        #Por ancho
        if W > 1e-4:
            #Por alto a la ida
            if (max_dif_ida > 5*dif_ida[ind_max_dif_ida-1]) and (max_dif_ida > 1e-3):
                #Por alto a la vuelta
                if (max_dif_vuelta > 5*dif_vuelta[ind_max_dif_vuelta-1]) and (max_dif_vuelta > 1e-3):
                    #Si todo lo anterior se cumplió, calculo las áreas
                    area_ida = np.trapz(A_s_ida, dx=ds)
                    area_vuelta = np.trapz(A_s_vuelta, dx=ds)

                    area_biestable = area_vuelta - area_ida

    del(A_s, B_s, A_s_ida, A_s_vuelta, dif_ida, dif_vuelta, max_dif_ida, ind_max_dif_ida, max_dif_vuelta, ind_max_dif_vuelta)

    if len(areas_biestables)>0:
        return areas_biestables

#Función para cuando los datos están en listas
def mide_biestabilidad(A_s, S, p=False):
    '''
    p = print. Si p=False (dafault), la función no imprime nada. Si p=True, imprime si se trata de un caso monoestable o biestable para cada modelo del csv
    '''    
    #Como criterio voy a medir biestabilidad en A_s
    #Separo entonces en ida y vuelta
    mitad = int(len(S)/2)

    S_ida = S[:mitad]
    S_vuelta = S[mitad:]
    
    ds = S[1] - S[0]
    
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

    #Defino S_on y S_off y el ancho W
    S_on = S_ida[ind_max_dif_ida]
    S_off = S_vuelta[ind_max_dif_vuelta]
    W = S_on - S_off

    #Empiezo a filtrar
    #Por ancho
    if W > 1e-4:
        #Por alto a la ida
        if (max_dif_ida > 5*dif_ida[ind_max_dif_ida-1]) and (max_dif_ida > 1e-3):
            #Por alto a la vuelta
            if (max_dif_vuelta > 5*dif_vuelta[ind_max_dif_vuelta-1]) and (max_dif_vuelta > 1e-3):
                #Si todo lo anterior se cumplió, calculo las áreas
                area_ida = np.trapz(A_s_ida, dx=ds)
                area_vuelta = np.trapz(A_s_vuelta, dx=ds)

                area_biestable = area_vuelta - area_ida

                #Filtro por área
                if area_biestable > 1e-4:
                    return area_biestable
    
    else:
        return 0
