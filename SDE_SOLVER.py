# -*- coding: utf-8 -*-
"""
Generate SDE DATA
"""

import sys
sys.path.insert(0,'/content/')


import scipy.io
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt

Nt=40
Num_sim=1000
a=0.0
b=1.0

t=np.zeros([Nt,1],dtype=float)
for I in range(0,Nt):
    t[I,0]=a+(b-a)/(Nt-1)*I

dt=(b-a)/(Nt-1)


y=np.ones([Nt,int(Num_sim)],dtype=float)
B=np.ones([Nt,Num_sim],dtype=float)

mu=1.1
sigma=0.5

for J in range(0,Num_sim):
    for I in range(0,Nt-1):
        B[I,J]=np.random.normal()
        y[I+1,J]=y[I,J]+mu*dt+np.sqrt(dt)*sigma*B[I,J]
#        y[I+1,J]=(1+mu*dt+np.sqrt(dt)*sigma*B[I,J])*y[I,J]
#    y[I+1,0]=y[I,0]+dt*mu*y[I,0]+np.sqrt(dt)*sigma*np.random.normal()*y[I]**1


y_mean=np.zeros([Nt,1],dtype=float)
for I in range(0,Num_sim):
    y_mean[:,0]=y_mean[:,0]+y[:,I]
y_mean=y_mean/Num_sim

plt.figure()
plt.plot(t,y)

plt.figure()
plt.plot(t,y_mean)

y_var=np.zeros([Nt,1],dtype=float)
for J in range(0,Num_sim):
    y_temp=np.zeros([Nt,1],dtype=float)
    for I in range(0,Nt):
        y_temp[I,0]=(y_mean[I,0]-y[I,J])**2+y_temp[I,0]
#    y_std=y_std+np.sqrt(y_temp/((Nt)*dt))
    y_var=y_var+y_temp/((Nt)*dt)
#y_std=np.sqrt(y_std/(Nt*dt))/Num_sim

y_var=y_var/Num_sim


plt.figure()
plt.plot(t,y_var)



savemat('data_SDE.mat',{'B':B,'t':t, 'y':y,'y_mean':y_mean,'y_var':y_var,'dt':dt , 'Nt':Nt, 'Num_sim':Num_sim, 'a':a, 'b':b})










Num_sim=16000
y=np.ones([Nt,int(Num_sim)],dtype=float)
B=np.ones([Nt,Num_sim],dtype=float)

for J in range(0,Num_sim):
    for I in range(0,Nt-1):
        B[I,J]=np.random.normal()
#        y[I+1,J]=y[I,J]+mu*dt+np.sqrt(dt)*sigma*B[I,J]
        y[I+1,J]=(1+mu*dt+np.sqrt(dt)*sigma*B[I,J])*y[I,J]
#    y[I+1,0]=y[I,0]+dt*mu*y[I,0]+np.sqrt(dt)*sigma*np.random.normal()*y[I]**1



#Forming approximate distribution:
Nx=30
rho=np.zeros([Nx,Nt],dtype=float)
x=np.zeros([Nx],dtype=float)
aa=0.0
bb=6.0
for I in range(0,Nx):
    x[I]=aa+(bb-aa)/(Nx-1)*(I)

dx=(bb-aa)/(Nx-1)

#for J in range(0,Num_sim):
#    for I in range(0,Nt-1):
#        B[I,J]=np.random.normal()
#        y[I+1,J]=y[I,J]+mu*dt+np.sqrt(dt)*sigma*B[I,J]

#for K in range(0,Nx-1):
#    for J in range(0,Num_sim):
#        for I in range(0,Nt-1):
#            if x[K]<=y[I+1,J] and y[I+1,J]<=x[K+1]:
#                rho[K,I]=rho[K,I]+1

for J in range(0,Num_sim):
    for K in range(0,Nx-1):
        for I in range(0,Nt):
            if x[K]<=y[I,J] and y[I,J]<=x[K+1]:
                rho[K,I]=rho[K,I]+1

#Normalization constant
c=np.zeros([Nt],dtype=float)
for I in range(0,Nt):
    c[I]=sum(rho[:,I])*(bb-aa)/(Nx-1)

for K in range(0,Nx):
    for I in range(0,Nt):
        rho[K,I]=rho[K,I]/c[I]




#rho=rho/(Num_sim)#*(bb-aa)/(Nx-1)


nn=int(0.2*Nt)

t1=t[nn,0]




x_mesh,y_mesh=np.meshgrid(t[nn:Nt-1],x)

cut_rho=np.zeros([Nx,Nt-nn-1],dtype=float)
for I in range(0,Nt-nn-1):
    for J in range(0,Nx):
        cut_rho[J,I]=rho[J,I+nn]
  



fig = plt.figure()
ax2 = plt.axes(projection='3d')
ax2.plot_wireframe(x_mesh,y_mesh, cut_rho, color='r')
plt.title('Probability_density_function',fontsize=14,fontweight='bold')





savemat('data_SDE_density.mat',{'rho':cut_rho,'t':t,'dx':dx,'dt':dt ,'Nt':Nt, 'Nx':Nx,'aa':aa, 'bb':bb,'t1':t1})


