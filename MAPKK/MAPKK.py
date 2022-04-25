# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 12:58:49 2018

@author: USUARIO
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 12:54:23 2018

@author: Jule
"""


import numpy as np
import pylab as plb
import pickle

Arch=open('RLS_fosfo_piola.pkl', 'wb')
guardar=0 #si es =1 guarda
from numba import njit,float64,vectorize
from scipy.signal import find_peaks
#%%
#@jit(nopython=True)

T0=1#10#/
d0=0.1 #10
j=2 #0 conserva con Am, 1 con d, 2 no conserva #3 es senoidal
L0=0.2#1/d0#0.01
S0=10**np.arange(-2,1,0.2)

acum=0#si es 0 miro entre s,s+T si es 1 miro acumulado

Tf=30#-S0

# frec=np.arange(1/Tf,1/d0,0.1)

frec=np.arange(1/Tf,1/d0,0.1)

# k1=1
# a1=1
# d1=1

# k2=1
# a2=1
# d2=1

# k3=1
# a3=1
# d3=1

# k4=1
# a4=1
# d4=1

# k5=1
# a5=1
# d5=1

# k6=1
# a6=1
# d6=1

# at=1
# d1t=1 #input
# d2t=1
# mt=1
# e2t=1

a1=1500
d1=4.5
k1=10.5


a2=1500
d2=1.5
k2=10.5

a3=1500
d3=12
k3=10.5

a4=1500
d4=12
k4=10.5

a5=1500
d5=12
k5=10.5

a6=1500
d6=12
k6=10.5

a7=1500
d7=12
k7=10.5

a8=1500
d8=12
k8=10.5

a9=1500
d9=12
k9=10.5

a10=1500
d10=12
k10=10.5


At=.1
D1t=L0 #input
D2t=.012
Mt=1.2
E2t=.06
Nt=4.8
E3t=0.05

necs=15
#%%
@njit
def int_rk4(f,x,dt,params):
    
    k_1 = f(x,params)
    k_2 = f(x+dt*0.5*k_1,params)
    k_3 = f(x+dt*0.5*k_2,params)
    k_4 = f(x+dt*k_3,params)
    y=x + dt*(k_1/6 + k_2/3 + k_3/3 + k_4/6)
    return y


#%%

@njit
def ecs(x,params):
#Mecanistico    
    D1t=params

    C1=x[0]
    C2=x[1]
    C3=x[2]
    C4=x[3]
    C5=x[4]
    C6=x[5]
    C7=x[6]
    C8=x[7]
    C9=x[8]
    C10=x[9]
    
    Ap=x[10]
    M1=x[11]
    M2=x[12]
    N1=x[13]
    N2=x[14]
    

    D1=D1t-C1
    D2=D2t-C2
    A=At-Ap-C1-C2-C3-C5
    M0=Mt-M1-M2-C3-C4-C5-C6-C7-C9
    N0=Nt-N1-N2-C7-C8-C9-C10
    E3=E3t-C8-C10
    E2=E2t-C4-C6
    
    
    dC1=a1*A*D1-(d1+k1)*C1
    dC2=a2*Ap*D2-(d2+k2)*C2
    dC3=a3*Ap*M0-(d3+k3)*C3
    dC4=a4*M1*E2-(d4+k4)*C4
    dC5=a5*M1*Ap-(d5+k5)*C5
    dC6=a6*M2*E2-(d6+k6)*C6
    dC7=a7*N0*M2-(k7+d7)*C7
    dC8=a8*N1*E3-(d8+k8)*C8
    dC9=a9*N1*M2-(d9+k9)*C9
    dC10=a10*N2*E3-(d10+k10)*C10
    dAp=k1*C1-a2*Ap*D2+d2*C2-a3*Ap*M0+(k3+d3)*C3-a5*M1*Ap+(d5+k5)*C5
    dM1=k3*C3-a4*M1*E2+d4*C4-a5*M1*Ap+d5*C5+k6*C6
    dM2=k5*C5-a6*M2*E2+d6*C6-a7*N0*M2+(d7+k7)*C7-a9*N1*M2+(d9+k9)*C9 
    dN1=k7*C7-a8*N1*E3+d8*C8-a9*N1*M2+d9*C9+k10*C10
    dN2=k9*C9-a10*N2*E3+d10*C10
    
    
    
    
    
 
    y=float64([dC1,dC2,dC3,dC4,dC5,dC6,dC7,dC8,dC9,dC10,dAp,dM1,dM2,dN1,dN2])
    
    return y

#%%
@njit    
def integra(t,dt,sis,Lt):
    i=0
    while i<len(t)-1:
        sis[:,i+1]=int_rk4(ecs,sis[:,i],dt,Lt[i])
        i=i+1
    return sis    
#%%

def Lin(t,p):
    y=0
    
    if j==0: #conservacion de dosis con amplitud
       d=d0
       Am=L0*p/T0
    
    if j==1: #conservacion de dosis con duracion de pulso
       d=0.1*p
       Am=L0
    
    if j==2: #sin conservacion de dosis
       d=d0
       Am=L0
    
    if j==0 or j==1 or j==2:
        if np.mod(t,p)<=d:
           y=Am
    
    if j==3:
       y=1*(1+np.sin(2*np.pi/p*t))/2
    return y

#%%
Lin=np.vectorize(Lin)
    
T=1/frec
dt=0.0001
tf=10
#    print(S0)
t=float64(np.arange(0,tf,dt))

   # t_int=np.where(t>=19*T)
sis=float64(np.zeros([necs,len(t)]))
   
Lt=float64(np.ones(len(t)))#e10*(1+0.1*np.sin(2*np.pi*frec*t))
sis=integra(t,dt,sis,Lt) 
# z1=sis[0,:]
# z2=sis[1,:]

#%%
c1=sis[0,:]
c2=sis[1,:]
c3=sis[2,:]
c4=sis[3,:]
c5=sis[4,:]
c6=sis[5,:]
c7=sis[6,:]
c8=sis[7,:]
c9=sis[8,:]
c10=sis[9,:]
ap=sis[10,:]
m1=sis[11,:]
m2=sis[12,:]
n1=sis[13,:]
n2=sis[14,:]
#%%
plb.figure()

# plb.plot(t,z1)
# plb.plot(t,z2)
plb.subplot(121)
plb.plot(t,c1)
plb.plot(t,c2)
plb.plot(t,c3)
plb.plot(t,c4)
plb.plot(t,c5)
plb.plot(t,c6)
plb.plot(t,c7)
plb.plot(t,c8)
plb.plot(t,c9)
plb.plot(t,c10)

plb.subplot(122)
plb.plot(t,ap,label='A*')
#plb.plot(t,m1,label='M1')
plb.plot(t,m2,label='M2')
#plb.plot(t,n1,label='N1')
plb.plot(t,n2,label='N2')

plb.legend()
