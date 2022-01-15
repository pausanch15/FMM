#Hace lo mismo que los otros, pero ya no para un csv. Además trato de hacer que la función ya me devuelva alto y ancho de biestabilidad, además del área. También trato de hacer que los threshholds sean parámetros que se le pasan a la función y no unos fijos predeterminados.

#Librerías
import numpy as np
import pandas as pd
import re

#Función para cuando los datos están en listas
def mide_biestabilidad(A_s, S, W_lim=1e-4, area_lim=1e-4):
    '''
    W_lim = Ancho de la región biestable límite. Solo toma como "biestable" a un sistema cuyo ancho sea mayor a este valor. El default es 1e-4.
    
    area_lim = Área de la región biestable límite. Solo toma como "biestable" a un sistema cuya área sea mayor a este valor. El default es 1e-4.

    Devuelve una lista de la forma [area_biestable, ancho_biestable, alto_biestable_off, alto_biestable_on]
    '''    
    #Como criterio voy a medir biestabilidad en A_s
    #Separo entonces en ida y vuelta
    mitad = int(len(S)/2)
    
    S = np.asarray(S)
    A_s = np.asarray(A_s)
    
    S_ida = S[:mitad]
    S_vuelta = S[mitad:]
    
    ds = S[1] - S[0]
    
    A_s_ida = A_s[:mitad]
    A_s_vuelta = A_s[mitad:]

    #Calculo los arrays de las diferencias punto a punto
    dif_ida = np.abs(np.diff(A_s_ida))
    dif_vuelta = np.abs(np.diff(A_s_vuelta))

    #Me fijo donde está la mayor diferencia: su valor y su índice
    # max_dif_ida = np.max(dif_ida)
    # ind_max_dif_ida = np.where(dif_ida == max_dif_ida)[0][0]
    ind_max_dif_ida = np.argmax(dif_ida)
    max_dif_ida = dif_ida[ind_max_dif_ida]
    
    # max_dif_vuelta = np.max(dif_vuelta)
    # ind_max_dif_vuelta = np.where(dif_vuelta == max_dif_vuelta)[0][0]
    ind_max_dif_vuelta = np.argmax(dif_vuelta)
    max_dif_vuelta = dif_vuelta[ind_max_dif_vuelta]

    #Defino S_on y S_off y el ancho W
    S_on = S_ida[ind_max_dif_ida]
    S_off = S_vuelta[ind_max_dif_vuelta]
    W = S_on - S_off

    #Empiezo a filtrar
    #Por ancho
    if W < W_lim:
        return None
    
    #Por alto a la ida y a la vuelta
    if (max_dif_ida > 5*dif_ida[ind_max_dif_ida-1]) and (max_dif_ida > 1e-3) \
    and (max_dif_vuelta > 5*dif_vuelta[ind_max_dif_vuelta-1]) and (max_dif_vuelta > 1e-3):
        #Si todo lo anterior se cumplió, calculo las áreas
        # area_ida = np.trapz(A_s_ida, dx=ds)
        # area_vuelta = np.trapz(A_s_vuelta, dx=ds)
# 
        # area_biestable = area_vuelta - area_ida

        zona_biestable = A_s_vuelta - A_s_ida
        area_biestable = np.trapz(zona_biestable, dx=ds)

        #Filtro por área
        if area_biestable > area_lim:
            return [area_biestable, W, max_dif_vuelta, max_dif_ida] 

    return None
