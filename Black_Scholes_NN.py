# -*- coding: utf-8 -*-
"""
Black Scholes equation via deep neural nets
This code is a simple modification of "Physics
informed Neural Networks" by Maizar Raissi
This is an early version code so it might not
be free of errors/it is not written in production
level.
@author: Ricardo
"""

import sys
sys.path.insert(0, '../../Utilities/')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import time
import math

def Normal_CDF(inpt):
    out=0.5*(1+math.erf(inpt/np.sqrt(2)))
    return out


class NeuralNetworkCode:
    def __init__(self,S_flat,t_flat,Payoff_flat,LeftBC,RightBC,layers,Losss,VV,Mask_flat):

        self.layers = layers
        self.Losss=Losss
        self.S_flat=S_flat
        self.t_flat=t_flat
        self.Payoff_flat = Payoff_flat
        self.LeftBC = LeftBC
        self.RightBC = RightBC
        self.VV = VV
        self.Mask_flat=Mask_flat

        self.Payoff_flat_tf = tf.placeholder(tf.float32, shape=[Nt*Ns,1])

        # Initialize NNs
        self.weights, self.biases, self.D1, self.D2 = self.initialize_NN(layers)

        self.V_final, self.Pinn, self.V_pred = self.net_uv(self.S_flat, self.t_flat)

        #Define the Loss
        self.loss =0.01*tf.reduce_mean(tf.abs(self.Pinn))+ tf.reduce_mean(tf.abs(self.V_final-self.Payoff_flat))+tf.reduce_mean(tf.abs(0.0 - self.D1))+tf.reduce_mean(tf.abs(0.0 - self.D2))

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

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        D1=0.0*tf.Variable(tf.ones([Ns*Nt,1],dtype=tf.float32), dtype=tf.float32)
        D2=0.0*tf.Variable(tf.ones([Ns*Nt,1],dtype=tf.float32), dtype=tf.float32)
        return weights, biases, D1, D2
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, weights, biases, D1, D2):
        SS=D1+S_flat
        TT=D2+t_flat
        num_layers = len(weights) + 1
        H=tf.concat([SS,TT], 1)
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
#            H = tf.tanh(tf.add(tf.matmul(H, W), b))
#            H = tf.nn.relu(tf.add(tf.matmul(H, W), b))
            H = tf.nn.sigmoid(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        print(W.shape)
        print(H.shape)
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H


    def net_uv(self, S_flat, t_flat):
        V = self.neural_net(self.weights, self.biases, self.D1, self.D2)
        V_s=tf.gradients(V,self.D1)[0]
        V_ss=tf.gradients(V_s,self.D1)[0]
        V_t=tf.gradients(V,self.D2)[0]
        pinn=V_t-(0.5*sigma**2*self.S_flat**2*V_ss-r*self.S_flat*V_s+r*V)
        V_final=V*Mask_flat

        return V_final,pinn,V


    def callback(self, loss):
        print('Loss:', loss)


    def train(self, nIter):
        tf_dict = {self.Payoff_flat_tf: Payoff_flat}
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
                print('It: %d, Loss: %.3e, time: %.3e' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
        
    def predict(self,):
        tf_dict = {self.Payoff_flat_tf: Payoff_flat}
        V_pred=self.sess.run(self.V_pred)
        return V_pred
    
if __name__ == "__main__": 

#    layers=44
    layers = [2, 5, 5, 5, 5, 5, 1]
#    layers = [2, 10, 10, 10, 10, 1]#Split
    Sa=0#Left BC
    Sb=50#Right BC
    ta=0#Starting time
    tb=2#Ending time
    Ns=50#Number of grid points in S-space
    Nt=50# Number of grid points in time

    #Defining the parameters in the Black-Scholes equation
    T=tb
    t=0
    sigma=0.25
    r=0.5
    E=25
    
    #Defining grid points (S,t)
    S=np.zeros([Ns], dtype=float)
    t=np.zeros([Nt], dtype=float)
    for I in range(0,Ns):
        S[I] = Sa+ (Sb-Sa)/(Ns-1)*(I)
    for I in range(0,Nt):
        t[I] = ta+ (tb-ta)/(Nt-1)*(I)

    #This is the exact solution. We will use this to test the accuracy 
    #of the predicted neural network solution.
    V=np.zeros([Ns,Nt],dtype=float)
    for J in range(0,Nt):
        for I in range(0,Ns):
            if (S[I]<=0):
                V[I,J]=0
            else:
                d1=(np.log(S[I]/E)+(r+0.5*sigma**2)*(T-t[J]))/(sigma*np.sqrt(T-t[J]))
                d2=(np.log(S[I]/E)+(r-0.5*sigma**2)*(T-t[J]))/(sigma*np.sqrt(T-t[J]))
                Term1=Normal_CDF(d1)
                Term2=Normal_CDF(d2)
                V[I,J]=S[I]*Term1-E*np.exp(-r*(T-t[J]))*Term2


    Mask=np.zeros([Ns,Nt],dtype=float)
    for I in range(0,Ns):
        Mask[I,Nt-1]=1
    for J in range(0,Nt):
        Mask[0,J]=1
        Mask[Ns-1,J]=1

    #Defining the payoff function
    Payoff=np.zeros([Ns,1],dtype=float)
    for I in range(0,Ns):
        Payoff[I,0]=max(S[I]-E,0)

    #Defining the Left BC:
    LeftBC=np.zeros([Nt,1],dtype=float)
    for J in range(0,Nt):
       LeftBC[J]=V[0,J]

    #Defining the Right BC:
    RightBC=np.zeros([Nt,1],dtype=float)
    for J in range(0,Nt):
       RightBC[J]=V[Ns-1,J]
        


    #Flattening the data i.e. creating a tile like structure
    S_flat=np.zeros([Ns*Nt,1], dtype=float)
    t_flat=np.zeros([Ns*Nt,1], dtype=float)
    V_flat=np.zeros([Ns*Nt,1], dtype=float)
    Mask_flat=np.zeros([Ns*Nt,1], dtype=float)
    Payoff_flat=np.zeros([Ns*Nt,1], dtype=float)
    L=0
    for I in range(0,Ns):
        for J in range(0,Nt):
            S_flat[L,0]=S[I]
            L=L+1
    L=0
    for I in range(0,Ns):
        for J in range(0,Nt):
            t_flat[L,0] = t[J]
            V_flat[L,0] = V[I,J]
            Mask_flat[L,0] = Mask[I,J]
            Payoff_flat[L,0] = V[I,J]*Mask[I,J]
            L=L+1

    #You need to define the number of iterations/Epocs for the training
    #This is what Nsteps does
    Nsteps=200000
    
    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I

    model = NeuralNetworkCode(S_flat,t_flat,Payoff_flat,LeftBC,RightBC,layers,Losss,V_flat,Mask_flat)
        
    start_time = time.time()
    model.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    V_pred = model.predict()


    LLosss=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        LLosss[I,0]=np.log(Losss[I,0])

    fig = plt.figure()
    plt.scatter(Losss_domain,LLosss, color='black', linewidth=2)
    

    #Restructuring the data from 1-d tensor to 2-d tensor
    #for the purpose of plotting    
    V_pred_3d=np.zeros([Ns,Nt],dtype=float)
    L=0
    for I in range(0,Ns):
        for J in range(0,Nt):
           V_pred_3d[I,J]=V_pred[L,0]
           L=L+1

    S_mesh,t_mesh=np.meshgrid(t,S)
    
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(S_mesh,t_mesh, V_pred_3d, color='r')
    plt.title('Neural Network Solution',fontsize=14,fontweight='bold')
    
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(S_mesh,t_mesh, V, color='r')
    plt.title('Exact Solution',fontsize=14,fontweight='bold')

    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(S_mesh,t_mesh, abs(V-V_pred_3d), color='r')
    plt.title('Error', fontsize=14,fontweight='bold')    
    
