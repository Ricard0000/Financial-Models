# -*- coding: utf-8 -*-
#This does market fitting:

    
import sys
sys.path.insert(0, '../../Utilities/')
import scipy.io
import pandas as pd
import csv
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import time
from scipy.io import savemat
from mpl_toolkits.mplot3d import Axes3D
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from scipy import interpolate
import math





class PDE_CLASSIFY:
    def __init__(self,x,v,Nx,N_layers):
        self.x = x
        self.v = v
        self.Nx=Nx
        self.N_layers=N_layers

        self.R1,self.R2,self.R3,self.R4,self.D1,self.D2,self.b1,self.b2 = self.initialize_NNg(N_layers)
        
        self.x_tf = tf.placeholder(tf.float32, shape=[Nx])
        self.v_tf = tf.placeholder(tf.float32, shape=[Nx])

        # tf Graphs
        self.v_pred,self.R1_tf,self.R2_tf,self.R3_tf,self.R4_tf,self.D1_tf,self.D2_tf,self.b1_tf,self.b2_tf = self.net_uv(self.x_tf)

        # Loss
        self.loss = tf.reduce_mean(tf.square(self.v_pred))

    # Optimizers
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')   
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)


    def initialize_NNg(self, N_layers):
#        R1=[]
#        R2=[]
#        D1=[]
#        D2=[]
 #       b1=[]
 #       b2=[]
        R1=tf.Variable(1.0,dtype=tf.float32)
        R2=tf.Variable(1.0,dtype=tf.float32)
        R3=tf.Variable(1.0,dtype=tf.float32)
        R4=tf.Variable(1.0,dtype=tf.float32)
        D1=tf.Variable(1.0,dtype=tf.float32)
        D2=tf.Variable(1.0,dtype=tf.float32)
        b1=tf.Variable(1.0,dtype=tf.float32)
        b2=tf.Variable(1.0,dtype=tf.float32)     
        return R1,R2,R3,R4,D1,D2,b1,b2

    def Model(self,x,R1,R2,R3,R4,D1,D2,b1,b2):

        line=R1*x+b1
        bump=R3*tf.math.erf(R2*(x-Mid))
        Left=line+bump 
        Right=0.0
#        Left=R1*x+0*R2/(x+D1)+b1
#        Right=R3*x+0*R4/(x+D2)+b2
        return Left,Right
       
    def net_uv(self,x):
        R1=self.R1
        R2=self.R2
        R3=self.R3
        R4=self.R4
        D1=self.D1
        D2=self.D2
        b1=self.b1
        b2=self.b2
        line,Right = self.Model(x,self.R1,self.R2,self.R3,self.R4,self.D1,self.D2,self.b1,self.b2)
        K_g=v-line        

        return K_g,R1,R2,R3,R4,D1,D2,b1,b2
    
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter):
        tf_dict = {self.x_tf: self.x,
                   self.v_tf: self.v}
        
        start_time = time.time()
        L=0
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            # Print
            loss_value = self.sess.run(self.loss, tf_dict)
            Losss[L,0]=loss_value
            L=L+1
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()

            
    def predictrho(self,x):
        tf_dict = {self.x_tf: x,
                   self.v_tf: v}
        R1=self.sess.run(self.R1_tf)
        R2=self.sess.run(self.R2_tf)
        R3=self.sess.run(self.R3_tf)
        R4=self.sess.run(self.R4_tf)
        D1=self.sess.run(self.D1_tf)
        D2=self.sess.run(self.D2_tf)
        b1=self.sess.run(self.b1_tf)
        b2=self.sess.run(self.b2_tf)    
        return R1,R2,R3,R4,D1,D2,b1,b2









class Small_scale_model:
    def __init__(self,xm,Monthly,Nm,N_dat,N_layers):
        self.xm = xm
        self.Monthly = Monthly
        self.Nm=Nm
        self.N_dat=N_dat        
        self.N_layers=N_layers

        self.a0,self.a1,self.a2,self.a3 = self.initialize_NNg(N_layers)

        self.xm_tf = tf.placeholder(tf.float32, shape=[N_dat,Nm])
        self.Monthly_tf = tf.placeholder(tf.float32, shape=[N_dat,Nm])

        # tf Graphs
        self.Monthly_pred,self.a0_tf,self.a1_tf,self.a2_tf,self.a3_tf = self.net_uv(self.xm_tf)

        # Loss
        self.loss = tf.reduce_mean(tf.square(self.Monthly_pred))

    # Optimizers
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')   

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)


    def initialize_NNg(self, N_layers):
        a0=tf.Variable(tf.ones([12],dtype=tf.float32),dtype=tf.float32)
        a1=tf.Variable(tf.ones([12],dtype=tf.float32),dtype=tf.float32)
        a2=tf.Variable(tf.ones([12],dtype=tf.float32),dtype=tf.float32)
        a3=tf.Variable(tf.ones([12],dtype=tf.float32),dtype=tf.float32)
        return a0,a1,a2,a3

    def Model(self,x,a0,a1,a2,a3):
        model=a0+a1*xm+a2*xm**2+a3*xm**3
        return model
       
    def net_uv(self,x):
        a0=self.a0
        a1=self.a1
        a2=self.a2
        a3=self.a3

        model = self.Model(xm,self.a0,self.a1,self.a2,self.a3)
        K_g=Monthly-model

        return K_g,a0,a1,a2,a3
    
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter):
        tf_dict = {self.xm_tf: self.xm,
                   self.Monthly_tf: self.Monthly}
        
        start_time = time.time()
        L=0
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            # Print
            loss_value = self.sess.run(self.loss, tf_dict)
            Losss2[L,0]=loss_value
            L=L+1
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()

            
    def predict(self,xm):
        tf_dict = {self.xm_tf: xm,
                   self.Monthly_tf: Monthly}
        a0=self.sess.run(self.a0_tf)
        a1=self.sess.run(self.a1_tf)
        a2=self.sess.run(self.a2_tf)
        a3=self.sess.run(self.a3_tf)

        return a0,a1,a2,a3

























if __name__ == "__main__":


    rows=[]
    with open('real_sales_per_day.csv','rt') as csvfile:
        csvreader=csv.reader(csvfile)
        fields=next(csvreader)
        for row in csvreader:
            rows.append(row)
    Nx=312
    v=np.zeros([Nx],dtype=float)
    for I in range(0,Nx):
        v[I]=float("%s"%rows[I][1])
    
    
    
    Norm=max(v)-min(v)
    v=(v-Norm/2)*(1/Norm)
    
    x=np.zeros([Nx],dtype=float)
    Lmask=np.zeros([Nx],dtype=float)
    Rmask=np.zeros([Nx],dtype=float)
    for I in range(0,Nx):
        x[I]=I    

    Mid=192
    TempN=Nx-Mid
    for I in range(0,Mid):
        Lmask[I]=1.0
    for I in range(0,TempN):
        Rmask[Mid+I]=1.0    
    
    w=v[1:Nx]-v[0:Nx-1]
    xx=x[0:Nx-1]

    fig = plt.figure()
    plt.plot(x,v, color='black', linewidth=2, label='none')
    
    fig = plt.figure()
    plt.plot(xx,w, color='black', linewidth=2, label='none')
    
    a=min(w)
    b=max(w)

    N=30
    xxx=np.zeros([N],dtype=float)
    histogram=np.zeros([N],dtype=float)
    for I in range(0,N):
        xxx[I]=a+(b-a)/(N-1)*I

    for J in range(0,N-1):
        for I in range(0,Nx-1):
            if (xxx[J]<=w[I]) and (w[I]<=xxx[J+1]):
                histogram[J]=histogram[J]+1

    fig = plt.figure()
    plt.plot(xxx,histogram, color='black', linewidth=2, label='none')

    N_layers=2
    Nsteps=28000
    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I
        
    modelrho = PDE_CLASSIFY(x,v,Nx,N_layers)

    start_time = time.time()
    modelrho.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    R1,R2,R3,R4,D1,D2,b1,b2= modelrho.predictrho(x)

    LLosss=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        LLosss[I,0]=np.log(Losss[I,0])

    fig = plt.figure()
    plt.scatter(Losss_domain,LLosss, color='black', linewidth=2, label='Log of Loss vs Number of Iterations')
    
#    Left=R1*x+1*R2/(x+D1)+b1
#    Right=R3*x+1*R4/(x+D2)+b2
#    Model=Rmask*Right+Lmask*Left
    bump=np.zeros([Nx],dtype=float)
    line=R1*x+b1
    for I in range(0,Nx):
        bump[I]=R3*math.erf(R2*(x[I]-Mid))
    Model=line+bump 

    fig = plt.figure()
    plt.scatter(x,Model, color='black', linewidth=2, label='Log of Loss vs Number of Iterations')
    plt.scatter(x,v, color='black', linewidth=2, label='Log of Loss vs Number of Iterations')

    w=v-Model
    fig = plt.figure()
    plt.scatter(w,Model, color='black', linewidth=2, label='Log of Loss vs Number of Iterations')

    a=min(w)
    b=max(w)

    N=30
    xxx=np.zeros([N],dtype=float)
    histogram=np.zeros([N],dtype=float)
    for I in range(0,N):
        xxx[I]=a+(b-a)/(N-1)*I

    for J in range(0,N-1):
        for I in range(0,Nx-1):
            if (xxx[J]<=w[I]) and (w[I]<=xxx[J+1]):
                histogram[J]=histogram[J]+1

    fig = plt.figure()
    plt.plot(xxx,histogram, color='black', linewidth=2, label='none')
    
    fig = plt.figure()
    plt.plot(x[200:312],w[200:312], color='black', linewidth=2, label='none')

    fig = plt.figure()
    plt.plot(x[300:312],w[300:312], color='black', linewidth=2, label='none')



    Nm=12    
    Monthly=np.zeros([round(312/Nm),Nm],dtype=float)
    dMonthly=np.zeros([round(312/Nm),Nm],dtype=float)
    xm=np.zeros([round(312/Nm),Nm],dtype=float)
    for I in range(0,round(312/Nm)):
        xm[I,0]=1/12
        xm[I,1]=2/12
        xm[I,2]=3/12
        xm[I,3]=4/12
        xm[I,4]=5/12
        xm[I,5]=6/12
        xm[I,6]=7/12
        xm[I,7]=8/12
        xm[I,8]=9/12
        xm[I,9]=10/12
        xm[I,10]=11/12
        xm[I,11]=12/12    
        
        
    dw=(np.roll(w,1,axis=0)-np.roll(w,-1,axis=0))/2.0
    L=0

    N_dat=round(312/Nm)
    for I in range(0,N_dat):
        for J in range(Nm):
            Monthly[I,J]=w[L]
            dMonthly[I,J]=dw[L]
            L=L+1

    fig = plt.figure()
    plt.plot(xm[0,:],Monthly[0,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[1,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[2,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[3,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[4,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[5,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[6,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[7,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[8,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[9,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[10,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[11,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[12,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[13,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[14,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[15,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[16,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[17,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[18,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[19,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[20,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[21,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[22,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[23,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[24,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],Monthly[25,:], color='black', linewidth=2, label='none')



    fig = plt.figure()
    plt.plot(xm[0,:],dMonthly[0,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[1,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[2,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[3,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[4,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[5,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[6,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[7,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[8,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[9,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[10,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[11,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[12,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[13,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[14,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[15,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[16,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[17,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[18,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[19,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[20,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[21,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[22,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[23,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[24,:], color='black', linewidth=2, label='none')
    plt.plot(xm[0,:],dMonthly[25,:], color='black', linewidth=2, label='none')


    N_layers=2
    Nm=12
    Nsteps=28000
    Losss2=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I

    model_small = Small_scale_model(xm,Monthly,Nm,N_dat,N_layers)
    start_time = time.time()
    model_small.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    a0,a1,a2,a3 = model_small.predict(xm)

    monthly_prediction=a0*xm[0,:]**0+a1*xm[0,:]+a2*xm[0,:]**2+a3*xm[0,:]**3
    
    LLosss2=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        LLosss2[I,0]=np.log(Losss2[I,0])

    fig = plt.figure()
    plt.plot(x[1:13],monthly_prediction, color='black', linewidth=2, label='none')



    L=0
    Total=np.zeros([312],dtype=float)
    for I in range(0,N_dat):
        for J in range(Nm):
            Total[L]=Model[L]+monthly_prediction[J]
            L=L+1


    fig = plt.figure()
    plt.plot(x,Total, color='red', linewidth=1, label='none')
    plt.plot(x,v, color='black', linewidth=2, label='none')
    
    fig = plt.figure()
    plt.plot(x,abs(v-Total), color='black', linewidth=2, label='none')
    
    
    w2=v-Total
    
    a=min(w2)
    b=max(w2)

    N=30
    xxx=np.zeros([N],dtype=float)
    histogram=np.zeros([N],dtype=float)
    for I in range(0,N):
        xxx[I]=a+(b-a)/(N-1)*I

    for J in range(0,N-1):
        for I in range(0,Nx-1):
            if (xxx[J]<=w2[I]) and (w2[I]<=xxx[J+1]):
                histogram[J]=histogram[J]+1

    fig = plt.figure()
    plt.plot(xxx,histogram, color='black', linewidth=2, label='none')
    
    
    
    