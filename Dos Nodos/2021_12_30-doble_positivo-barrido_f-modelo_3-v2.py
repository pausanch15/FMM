#Tomo los parámetros que me pasó Fede para los que sé que hay biestabilidad y me fijo si mide_biestabilidad_v3.py reconoce al sistema como biestable.

#Librerías
import matplotlib.pyplot as plt
import numpy as np
import mide_biestabilidad_v3 as mb
import runge_kuta as rk
import going_up_comming_down_doble_positivo_modelo_3 as gucd
plt.ion()

#Integro para parámetros que sé que dan biestabilidad
K_sa = .1
K_sb = .1
k_ba = 1
k_ab = 1
K_ba = .1
K_ab = .1
k_sa = 1
k_sb = 1

tiempo_max = 100
S_max = 1
S_min = 0
pasos = 1000

A_s, B_s, S = gucd.gucd_modelo_3(K_sa, K_sb, k_ba, k_ab, K_ba, K_ab, k_sa, k_sb, tiempo_max, S_max, S_min, pasos)

#Ploteo
plt.figure()
plt.plot(S, A_s, 'o', label='A')
plt.plot(S, B_s, '.', label='B')
# plt.plot(S, label='S')
plt.xlabel('S')
# plt.ylabel('Variables')
plt.legend()
plt.grid()
plt.show()

#Me fijo a ver si mi código mide_biestabilidad_v3.py encuentra biestabilidad en este sistema
resultado_medicion = mb.mide_biestabilidad(A_s, S) #No lo toma como biestable!

#Intento arreglar esto. Me traigo para acá el código mide_biestabilidad_v3.py y trato de arreglarlo. La copìo no como función para poder ir viendo paso a paso cosas.
W_lim=1e-4
area_lim=1e4 #Ups... debería ser 1e-4

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
if W > W_lim:
    #Por alto a la ida
    if (max_dif_ida > 5*dif_ida[ind_max_dif_ida-1]) and (max_dif_ida > 1e-3):
        #Por alto a la vuelta
        if (max_dif_vuelta > 5*dif_vuelta[ind_max_dif_vuelta-1]) and (max_dif_vuelta > 1e-3):
        
            #Probando cada cosa por separado, veo que hasta acá todas las condiciones se cumplen (son True)
            
            #Si todo lo anterior se cumplió, calculo las áreas
            area_ida = np.trapz(A_s_ida, dx=ds)
            area_vuelta = np.trapz(A_s_vuelta, dx=ds)

            area_biestable = area_vuelta - area_ida

            #Las condiciones se cumplen y area_biestable existe, así que hasta acá está todo ok!

            #ACÁ ESTA EL ERROR!! Dice que area_biestable > area_lim es False (lo cual es coherente con el valor erroneo de límite que tenía) y después no corre el else. esto último es un error de cómo están indentadas las cosas, hay que corregirlo igual, a pesar de el error que encontré en el límite de área que estaba tomando

            #Filtro por área
            if area_biestable > area_lim:
                resultado = [area_biestable, W, max_dif_vuelta, max_dif_ida]

else:
    resultado = [0]

#Copio y pego la parte que voy a cambiar y acá la cambio, para no pisar el lugar donde marco los errores que tuve. No cambio el valor incorrecto de área límite que tenía así corrijo el error de indentación
#Empiezo a filtrar
#Por ancho
if W > W_lim:
    #Por alto a la ida
    if (max_dif_ida > 5*dif_ida[ind_max_dif_ida-1]) and (max_dif_ida > 1e-3):
        #Por alto a la vuelta
        if (max_dif_vuelta > 5*dif_vuelta[ind_max_dif_vuelta-1]) and (max_dif_vuelta > 1e-3):
            
            #Si todo lo anterior se cumplió, calculo las áreas
            area_ida = np.trapz(A_s_ida, dx=ds)
            area_vuelta = np.trapz(A_s_vuelta, dx=ds)

            area_biestable = area_vuelta - area_ida

            #Filtro por área
            if area_biestable > area_lim:
                resultado = [area_biestable, W, max_dif_vuelta, max_dif_ida]

            else:
                resultado = [0]

#Ahora sí!!

#Arreglado esto, pruebo poniendo el valor correcto de límite de área a ver si ahora sí filtra al sistema como biestable
W_lim=1e-4
area_lim=1e-4 

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
if W > W_lim:
    #Por alto a la ida
    if (max_dif_ida > 5*dif_ida[ind_max_dif_ida-1]) and (max_dif_ida > 1e-3):
        #Por alto a la vuelta
        if (max_dif_vuelta > 5*dif_vuelta[ind_max_dif_vuelta-1]) and (max_dif_vuelta > 1e-3):
            
            #Si todo lo anterior se cumplió, calculo las áreas
            area_ida = np.trapz(A_s_ida, dx=ds)
            area_vuelta = np.trapz(A_s_vuelta, dx=ds)

            area_biestable = area_vuelta - area_ida

            #Filtro por área
            if area_biestable > area_lim:
                resultado = [area_biestable, W, max_dif_vuelta, max_dif_ida]

            else:
                resultado = [0]

#Ahora sí encuentra al sistema como biestable!!

#Corrijo mide_biestabilidad_v3.py con estos cambios

#Pruebo mide_biestabilidad_v3.py con estos cambios
resultado_medicion = mb.mide_biestabilidad(A_s, S) #Ahora sí!!
