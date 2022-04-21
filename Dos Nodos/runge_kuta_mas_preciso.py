#Hace lo mismo que el otro runge_kuta pero toma como constante a una diferencia mucho menor.
#Librerías
import numpy as np

#Runge Kutta de órdenes 3 y 4
def rk4(modelo,params, variables, dt):
    k_1 = dt * modelo(variables,params)
    k_2 = dt * modelo(variables + 0.5 * k_1,params)
    k_3 = dt * modelo(variables + 0.75 * k_2,params)
    d_rk3 = 0.2222222222222222 * k_1 + 0.3333333333333333 * k_2 + 0.4444444444444444 * k_3
    k_4 = dt * modelo(variables + d_rk3,params)
    d_rk4 = 0.2916666666666667 * k_1 + 0.25 * k_2 + 0.3333333333333333 * k_3 + 0.125 * k_4
    return d_rk3,d_rk4

#Aplicamos Runge Kutta. Si el error es chico agrandamos el paso temporal, si es grande lo achicamos.
def integrar(modelo, params, condiciones_iniciales, tiempo_max, tiempo_inicial = 0,
             error_min = 10**-6, error_max = 10**-4, max_iter=10000):
    
    cant_variables = len(condiciones_iniciales)
    
    dt_max = .1
    dt_min = 0.001 

    variables = np.zeros([cant_variables,max_iter])
    variables[:,0] = condiciones_iniciales
    
    tiempo = np.ones(max_iter)*tiempo_inicial
    
    i = 0
    dt = dt_max
    transitorio = True
    while i<max_iter-1 and tiempo[i]<tiempo_max+1 and transitorio:
        if dt < dt_min:
            dt = dt_min
        elif dt > dt_max:
            dt = dt_max
        dk4,dk5 = rk4(modelo, params, variables[:,i], dt)
        error = np.sum(np.abs(dk4-dk5))
        if error > error_max and dt > dt_min:
            dt = 0.5*dt
        else:
            variables[:,i+1] = variables[:,i] + dk5
            tiempo[i+1] = tiempo[i] + dt
            i+=1
            if error < error_min:
                dt = 2 * dt
        if np.max(np.abs(variables[:,i]-variables[:,i-1]))<0.00000001 and i>1:
            transitorio = False
    return tiempo[:i],variables[:,:i]
