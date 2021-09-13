

import sys
sys.path.insert(0, '../../Utilities/')
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import pandas_datareader as pdr
import datetime as dt

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

#        self.loss = tf.reduce_mean(tf.square(1/(abs(self.Ratio)+0.0000001)))+1*tf.reduce_mean(tf.square(self.const-1.0))#+100*tf.reduce_mean(tf.abs(self.constraint-1.0))

        self.loss = tf.reduce_mean(1/(1+tf.square(abs(self.Ratio))))+tf.reduce_mean(tf.square(self.const-1.0))#+100*tf.reduce_mean(tf.abs(self.constraint-1.0))

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
    Nx=20

    #Loading data
    ticker1 = ["MMM"]
    ticker2 = ["ABT"]
    ticker3 = ["ABBV"]
    ticker4 = ["ABMD"]
    ticker5 = ["ACN"]
    ticker6 = ["ATVI"]
    ticker7 = ["ADBE"]
    ticker8 = ["AMD"]
    ticker9 = ["AAP"]
    ticker10 = ["AES"]    
    ticker11 = ["AFL"]
    ticker12 = ["A"]
    ticker13 = ["APD"]
    ticker14 = ["AKAM"]
    ticker15 = ["ALK"]
    ticker16 = ["ALB"]
    ticker17 = ["ARE"]
    ticker18 = ["ALGN"]
    ticker19 = ["ALLE"]
    ticker20 = ["LNT"]
    ticker21 = ["ALL"]
    ticker22 = ["GOOGL"]
    ticker23 = ["GOOG"]
    ticker24 = ["MO"]
    ticker25 = ["ALB"]
    
    ticker26 = ["AMZN"]
    ticker27 = ["AMCR"]
    ticker28 = ["AEE"]
    ticker29 = ["AAL"]
    ticker30 = ["AEP"]
    ticker31 = ["AXP"]
    ticker32 = ["AIG"]
    ticker33 = ["AMT"]
    ticker34 = ["AWK"]
    ticker35 = ["AMP"]
    ticker36 = ["ABC"]
    ticker37 = ["AME"]
    ticker38 = ["AMGN"]
    ticker39 = ["APH"]
    ticker40 = ["ADI"]
    ticker41 = ["ANSS"]
    ticker42 = ["ANTM"]
    ticker43 = ["AON"]
    ticker44 = ["AOS"]
    ticker45 = ["APA"] 
    ticker46 = ["AAPL"]    
    ticker47 = ["AMAT"]
    ticker48 = ["APTV"]
    ticker49 = ["ADM"]
    ticker50 = ["ANET"]

    ticker_list=[ticker1,ticker2,ticker3,ticker4,ticker5,ticker6,ticker7,
                 ticker8,ticker9,ticker10,ticker11,ticker12,ticker13,ticker14,
                 ticker15,ticker16,ticker17,ticker18,ticker19,ticker20,
                 ticker21,ticker22,ticker23,ticker24,ticker25,ticker26,
                 ticker27,ticker28,ticker29,ticker30,ticker31,ticker32,
                 ticker33,ticker34,ticker35,ticker36,ticker37,ticker38,
                 ticker39,ticker40,ticker41,ticker42,ticker43,ticker44,
                 ticker45,ticker46,ticker47,ticker48,ticker49,ticker50,
                 ]
    #test example 1
#    Start_year=2019
#    Start_month=4
#    Start_day=15
    
#    End_year=2019
#    End_month=11
#    End_day=1

    #test example 2
    Start_year=2014
    Start_month=3
    Start_day=1
    
    End_year=2021
    End_month=8
    End_day=1



    
    start = dt.datetime(Start_year, Start_month, Start_day)
    end = dt.datetime(End_year, End_month, End_day)

    data = pdr.get_data_yahoo(ticker1, start, end)
    v=data.values
    vclose=v[:,1]
    N_ret=vclose.size
    arr=np.zeros([N_ret,Nx])#This array has all stock closing values

    for I in range(0,Nx):
        print(I)
        data = pdr.get_data_yahoo(ticker_list[I], start, end)
        v=data.values
        arr[:,I]=v[:,1]#Array of closing values
    vs=np.roll(vclose,1,0)#Shift of the Closing Price for differncing purposes
    vs[0]=0.0    
    
    N_ret=vclose.size
    
    #The returns are defined by RETURN_{i} = (R_{i+1}-R_{i})/(R_{i})
    #Rather its the log of R
    
    domain=np.linspace(0,N_ret/252,N_ret)    
    reg=np.zeros([N_ret,Nx],dtype=float)
    ######################################################
    #
    #         EXPONENTIAL MODEL: Ae^(rt)
    # The growth rate r is set to average growth rate 
    # determined by the log of the returns.
    # The initial condition A is not set to the value
    # at time 0, instead it is set to a constant which
    # best fits the data and solved via normal equations
    #
    #####################################################   
    dt=1/252 # Usually, there is a 252 days/year data. Hence the reason for 1/252
    ret=np.zeros([N_ret,Nx],dtype=float)
    for I in range(0,Nx):
        ret[:,I]=np.roll(np.log(arr[:,I]),-1,0)-np.log(arr[:,I])
    mu=np.zeros([Nx],dtype=float)
    Mean=np.zeros([Nx],dtype=float)
    s=np.zeros([Nx],dtype=float)
    std=np.zeros([Nx],dtype=float)
    for I in range(0,Nx):
        mu[I]=sum(ret[1:N_ret-1,I])/((N_ret-2)*dt)
        Mean[I]=mu[I]*dt
        s[I]=sum((ret[1:N_ret-1,I]-Mean[I])**2)
        std[I]=np.sqrt(s[I]/((N_ret-3)*dt))

    time_ax=np.linspace(0,N_ret,N_ret)
    for I in range(0,Nx):
        A=np.matmul(np.transpose(np.exp(Mean[I]*time_ax[:])),arr[:,I])
        A=A/np.matmul(np.transpose(np.exp(Mean[I]*time_ax[:])),np.exp(Mean[I]*time_ax[:]))
        reg[:,I]=A*np.exp(Mean[I]*time_ax)

    SSE=np.zeros([Nx],dtype=float)
    for I in range(0,Nx):
        SSE[I]=sum((arr[:,I]-reg[:,I])**2)

    sigma=np.zeros([Nx,1],dtype=float)
    stock_value=np.zeros([Nx,1],dtype=float)
    
    for I in range(0,Nx):
        stock_value[I,0]=mu[I]
        sigma[I,0]=np.sqrt(SSE[I]/(N_ret-1))


    """
    #########################################
    #
    #         LINEAR MODEL
    ##########################################    


    #compute regression line:
    xbar=np.mean(domain)
    ybar=np.zeros([Nx],dtype=float)
    for I in range(0,Nx):
        ybar[I]=np.mean(arr[:,I])

    temp1=domain-xbar
    
    SSxy=np.zeros([Nx],dtype=float)
    SSxx=np.zeros([Nx],dtype=float)
    b1=np.zeros([Nx],dtype=float)
    b0=np.zeros([Nx],dtype=float)
    for I in range(0,Nx):
        temp2=arr[:,I]-ybar[I]
        SSxy[I]=sum(temp1*temp2)
        SSxx[I]=sum(temp1*temp1)
        b1[I]=SSxy[I]/SSxx[I]
        b0[I]=ybar[I]-b1[I]*xbar
        

    SSE=np.zeros([Nx],dtype=float)
    for I in range(0,Nx):
        reg[:,I]=b0[I]+b1[I]*domain
        SSE[I]=sum((arr[:,I]-reg[:,I])**2)
            
    #computing Std using regression line:
#    mu=np.sum(ret1)/((N_ret-1)*dt)
    
    sigma=np.zeros([Nx,1],dtype=float)
    stock_value=np.zeros([Nx,1],dtype=float)
    
    for I in range(0,Nx):
        stock_value[I,0]=b1[I]
        sigma[I,0]=np.sqrt(SSE[I]/(N_ret-1))
    """        
        
        
        
        
        
        

    #If you want specific values for volatility and expected return,
    #This is where you would modify the code (the next 3 lines)
#    for I in range(0,Nx):
#        sigma[I,0]=np.random.random()#Using random volatility (standard deviation)
#        stock_value[I,0]=Mean1


    """
    plt.style.use('fivethirtyeight')
    plt.figure()
    plt.hist(ret[:,0],bins=60,edgecolor='black')
    plt.title('Distribution of returns Stock1')
    string1=str(float(N_ret)/float(252))
    string2=str(Start_month)+'/'+str(Start_day)+'/'+str(Start_year)
    plt.xlabel(string1[0]+string1[1]+string1[2]+'-year time period starting from '+string2)

    plt.figure()
    plt.hist(ret[:,1],bins=60,edgecolor='black')
    plt.title('Distribution of returns Stock2')
    string1=str(float(N_ret)/float(252))
    string2=str(Start_month)+'/'+str(Start_day)+'/'+str(Start_year)
    plt.xlabel(string1[0]+string1[1]+string1[2]+'-year time period starting from '+string2)
    """

    for I in range(0,Nx):
        fig = plt.figure()
        plt.scatter(domain,arr[:,I], color='black',linewidth=2,label='Stock'+str(I+1)+'closing')
        plt.scatter(domain,reg[:,I], color='red',linewidth=2,label='regression line')
        plt.legend()
        plt.xlabel('t-time (years)')
        plt.ylabel('$S(t)$')
        plt.title('Stock '+str(I+1)+' vs Regression')
        plt.show()


    Investment=1000.0

    Nsteps=120000

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
        LLosss[I,0]=-np.log(abs(Losss[I,0]))

    fig = plt.figure()
    plt.scatter(Losss_domain,LLosss, color='black', linewidth=2)
    plt.title('Loss: More training could give better Sharpe Ratio!',fontsize=14,fontweight='bold')    

    weights = model.predict()

    print('The computed sharpe ratio is:')
    nsr=0
    dsr=0
    for I in range(0,Nx):
        nsr=nsr+(sigmoid(weights[I])*stock_value[I,0])
    for I in range(0,Nx):
        dsr=(sigmoid(weights[I])*sigma[I,0])**2+dsr

    print(nsr/np.sqrt(dsr))
    
    temp_dist=np.zeros([Nx],dtype=float)
    for I in range(0,Nx):
        temp_dist[I]=(sigmoid(weights[I])*stock_value[I,0])

    check=0
    print('This is how to distribute your cash:')
    for I in range(0,Nx):
        print(str.format('{0:.2f}',Investment*sigmoid(weights[I])*stock_value[I,0]/sum(temp_dist)))
        v=Investment*sigmoid(weights[I])*stock_value[I,0]/sum(temp_dist)
        check=check+v        

    print('This number is for checking if the algorithm worked well:')
    print(check)
    print('If this number is not the invested value, you might need to')
    print('train the algorithm more.')
