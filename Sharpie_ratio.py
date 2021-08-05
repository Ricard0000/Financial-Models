# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 23:06:44 2021
Problem: You want to inveset 1,000 dollars on 10 stocks. Suppose you know
the volatility and return on investment for each of the stock.
How do you maximize your profit and at
the same time minimize your risk?
One way to do this is to use the sharpe ratio.
This code is based on the adam-optimizer.
This is not financial advice, use this code at your own risk. Note: This
assumes uncorrelated stocks! The problem becomes harder when there is
correlation among the underlyings.
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
def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig


class NeuralNetworkCode:
    def __init__(self,sigma,stock_value,Investment,Losss):

        self.Losss=Losss
        self.sigma = sigma
        self.stock_value=stock_value
        self.Investment = Investment

        self.weights = self.initialize_NN(Nx)

        self.sigma_tf = tf.placeholder(tf.float32, shape=[Nx,1])
        self.stock_value_tf = tf.placeholder(tf.float32, shape=[Nx,1])

        self.Ratio, self.constraint, self.const = self.sharpie_ratio(self.weights,self.stock_value,self.sigma)

        self.loss = tf.reduce_mean(tf.square(1/(abs(self.Ratio)+0.00001)))+tf.reduce_mean(tf.square(self.const-1.0))#+100*tf.reduce_mean(tf.abs(self.constraint-1.0))

        self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    def initialize_NN(self, Nx):
        weights = []
        for l in range(Nx):
            w=abs(np.random.random())*tf.Variable(1.0, dtype=tf.float32)
            weights.append(w)
        return weights

    def sharpie_ratio(self,weights,stock_value,sigma):
        constraint=0.0
        Numerator=0.0
        Ratio=0.0
        for I in range(0,Nx):
            constraint=constraint+tf.nn.sigmoid(weights[I])
        for I in range(0,Nx):
#            Numerator=Numerator+self.Investment*tf.nn.sigmoid(weights[I])*stock_value[I]
            Numerator=Numerator+tf.nn.sigmoid(weights[I])*stock_value[I]
        Denominator=0.0
        for I in range(0,Nx):
            Denominator=Denominator+tf.nn.sigmoid(weights[I])**2*sigma[I]**2            
        Ratio=Numerator/tf.sqrt(Denominator)
        return Ratio,weights,constraint


    def callback(self, loss):
        print('Loss:', loss)


    def train(self, nIter):
        tf_dict = {self.sigma_tf: sigma,
                   self.stock_value_tf: stock_value}
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
        tf_dict = {self.sigma_tf: sigma,
                   self.stock_value_tf: stock_value}
        weights=self.sess.run(self.weights)
        return weights


if __name__ == "__main__": 

    #You are investing in Nx number of stocks
    Nx=10

    #Random_standard_deviations
    sigma=np.zeros([Nx,1],dtype=float)
    #Random_worth_of_stocks
    stock_value=np.zeros([Nx,1],dtype=float)

    #If you want specific values for volatility and expected return,
    #This is where you would modify the code (the next 3 lines)
    for I in range(0,Nx):
        sigma[I,0]=np.random.random()#Using random volatility (standard deviation)
        stock_value[I,0]=np.random.random()#Using random return on investment

    """    
    stock_value[0,0]=0.1
    sigma[0,0]=0.1
    stock_value[1,0]=0.15
    sigma[1,0]=0.2
    #Set Nx=2
    #Test with 2 stocks, answer should be Sharpe Ratio=1.25
    """
    Investment=1000.0

    Nsteps=50000

    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I

    model = NeuralNetworkCode(sigma,stock_value,Investment,Losss)

    start_time = time.time()
    model.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    
    LLosss=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        LLosss[I,0]=np.log(abs(Losss[I,0]))

    fig = plt.figure()
    plt.scatter(Losss_domain,LLosss, color='black', linewidth=2)
    plt.title('Loss: More training could give better Sharpe Ratio!',fontsize=14,fontweight='bold')    

    weights = model.predict()

    print('This is how to distribute your cash:')
    for I in range(0,Nx):
        print(str.format('{0:.2f}',Investment*sigmoid(weights[I])))
        
        
    print('The computed sharpe ratio is:')
    nsr=0
    dsr=0
    for I in range(0,Nx):
        nsr=nsr+(sigmoid(weights[I])*stock_value[I,0])
    for I in range(0,Nx):
        dsr=(sigmoid(weights[I])*sigma[I,0])**2+dsr

    print(nsr/np.sqrt(dsr))
    
    
    
    
    
    
    
    
    
    

