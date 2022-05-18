#Integro lo mismo que en el v2, pero esta vez intento agregar unas barritas para hacer barridos en los parámetros.

#%%
#Librerías
import matplotlib.pyplot as plt
import numpy as np
import runge_kuta as rk
import integra_FPN_desacoplado as fpndes
from scipy import interpolate
import runge_kuta_estimulo as rks
import plotly.graph_objects as go
plt.ion()

#%%
#Integramos
#Primero con los parámetros que propone el paper
# dYX2 = 0.14
# dX2 = 0.1
# Ty2 = 0.21
# dy2 = 0.1
# TY2 = 0.3
# dY2 = 0.1
# 
# tiempo, variables, tiempo_estimulo, estimulo = fpndes.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, tiempo_max=2000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=0.04, tau=200, ts=30, tb=530, condiciones_iniciales=[0.01, 0.01, 0.01])
# 
# X, y, Y = variables
# 
# #Para tratar de analizar la caida, disminuyo la amplitud de las oscilaciones en un factor epsilon
# epsilon = 0.01
# X, y, Y = variables*epsilon
 # 
# plt.figure()
# plt.plot(tiempo, X, label='X')
# plt.plot(tiempo, y, label='y')
# plt.plot(tiempo, Y, label='Y')
# plt.plot(tiempo_estimulo, estimulo, 'k', label='Estímulo')
# plt.grid()
# plt.title('FPN Con los Valores del Paper')
# plt.legend()
# plt.show()

#%%
#Pruebo la librería para hacer barritas
#Pruebo primero barrer solo en la altura del escalón
dYX2 = 0.14
dX2 = 0.1
Ty2 = 0.21
dy2 = 0.1
TY2 = 0.3
dY2 = 0.1
epsilon = 0.01

#Create figure
fig = go.Figure()

#Add traces, one for each slider step
for step in np.arange(0, 5, 0.1):
    print(f'Va por step = {step}')
    print()
    # fig.add_trace(
        # go.Scatter(
            # visible=False,
            # line=dict(color="#00CED1", width=6),
            # name="𝜈 = " + str(step),
            # x=np.arange(0, 10, 0.01),
            # y=np.sin(step * np.arange(0, 10, 0.01))))
    S_alto_actual = step*0.015
    tiempo, variables, tiempo_estimulo, estimulo = fpndes.integra_FPN_estimulo(dYX2, dX2, Ty2, dy2, TY2, dY2, tiempo_max=2000, resolucion=10000, ti=0, tf=2000, S_min=0, S_max=S_alto_actual, tau=200, ts=30, tb=530, condiciones_iniciales=[0.01, 0.01, 0.01])
    X, y, Y = variables*epsilon
    
    fig.add_trace(
        go.Scatter(
            visible=False,
            line=dict(color="#00CED1", width=6),
            name=f'X',
            x=tiempo,
            y=X))

    # del(tiempo, variables, tiempo_estimulo, estimulo, X, y, Y)

#Make 10th trace visible
fig.data[10].visible = True

#Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

fig.show()

