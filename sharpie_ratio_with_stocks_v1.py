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

        self.loss = -tf.reduce_mean(tf.square(abs(self.Ratio)))+tf.reduce_mean(tf.square(self.const-1.0))#+100*tf.reduce_mean(tf.abs(self.constraint-1.0))

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
    Nx=2

    #Loading data
    ticker1 = ["AAPL"]
    ticker2 = ["IBM"]

    #test example 1
#    Start_year=2019
#    Start_month=4
#    Start_day=15
    
#    End_year=2019
#    End_month=11
#    End_day=1

    #test example 2
    Start_year=2019
    Start_month=1
    Start_day=1
    
    End_year=2021
    End_month=8
    End_day=1



    
    start = dt.datetime(Start_year, Start_month, Start_day)
    end = dt.datetime(End_year, End_month, End_day)


    data1 = pdr.get_data_yahoo(ticker1, start, end)
    data2 = pdr.get_data_yahoo(ticker2, start, end)

    v1=data1.values #This is numeric/floating point data with no strings
    vclose1=v1[:,1]#Closing price
    vs1=np.roll(vclose1,1,0)#Shift of the Closing Price for differncing purposes
    vs1[0]=0.0    


    v2=data2.values #This is numeric/floating point data with no strings
    vclose2=v2[:,1]#Closing price
    vs2=np.roll(vclose2,1,0)#Shift of the Closing Price for differncing purposes
    vs2[0]=0.0
  
    
    
    N_ret=vclose1.size

    #The returns are defined by RETURN_{i} = (R_{i+1}-R_{i})/(R_{i})
    #Rather its the log of R
    ret1=np.zeros([N_ret-4],dtype=float)
    ret2=np.zeros([N_ret-4],dtype=float)
    for I in range(2,N_ret-2):
#        ret1[I-2]=(vclose1[I+1]-vclose1[I])/vclose1[I]
        ret1[I-2]=np.log(vclose1[I+1])-np.log(vclose1[I])
#        ret2[I-2]=(vclose2[I+1]-vclose2[I])/vclose2[I]
        ret2[I-2]=np.log(vclose2[I+1])-np.log(vclose2[I])


    domain=np.linspace(0,N_ret/252,vclose1.size)
    #compute regression line:
    xbar=np.mean(domain)
    ybar1=np.mean(vclose1)
    ybar2=np.mean(vclose2)

    temp1=domain-xbar

    temp2=vclose1-ybar1
    SSxy_1=sum(temp1*temp2)
    SSxx_1=sum(temp1*temp1)

    temp2=vclose2-ybar2
    SSxy_2=sum(temp1*temp2)
    SSxx_2=sum(temp1*temp1)


    b1_1=SSxy_1/SSxx_1
    b0_1=ybar1-b1_1*xbar   
    reg1=b0_1+b1_1*domain
    
    b1_2=SSxy_2/SSxx_2
    b0_2=ybar2-b1_2*xbar
    reg2=b0_2+b1_2*domain
    
    #computing Std using regression line:
    SSE_1=sum((vclose1-reg1)**2)
    SSE_2=sum((vclose2-reg2)**2)
    
    
    dt=1/252 # Usually, there is a 252 days/year data. Hence the reason for 1/252
#    mu=np.sum(ret1)/((N_ret-1)*dt)
    mu1=np.mean(ret1)/dt
    mu2=np.mean(ret2)/dt

    std1=np.std(ret1, dtype=np.float64)/np.sqrt(dt)
    std2=np.std(ret2, dtype=np.float64)/np.sqrt(dt)
    #Mean and mu are related by: Mean = Mu*dt
    Mean1=mu1*dt
    Mean2=mu2*dt
    
    
    sigma=np.zeros([Nx,1],dtype=float)
    stock_value=np.zeros([Nx,1],dtype=float)
    
    

#    stock_value[0,0]=Mean1
#    stock_value[1,0]=Mean2
    stock_value[0,0]=mu1
    stock_value[1,0]=mu2

    stock_value[0,0]=b1_1
    stock_value[1,0]=b1_2

    sigma[0,0]=np.sqrt(SSE_1/(N_ret-1))
    sigma[1,0]=np.sqrt(SSE_2/(N_ret-1))


    #If you want specific values for volatility and expected return,
    #This is where you would modify the code (the next 3 lines)
#    for I in range(0,Nx):
#        sigma[I,0]=np.random.random()#Using random volatility (standard deviation)
#        stock_value[I,0]=Mean1



    
    
    

    plt.style.use('fivethirtyeight')
    plt.figure()
    plt.hist(ret1,bins=60,edgecolor='black')
    plt.title('Distribution of returns Stock1')
    string1=str(float(N_ret)/float(252))
    string2=str(Start_month)+'/'+str(Start_day)+'/'+str(Start_year)
    plt.xlabel(string1[0]+string1[1]+string1[2]+'-year time period starting from '+string2)

    plt.figure()
    plt.hist(ret2,bins=60,edgecolor='black')
    plt.title('Distribution of returns Stock2')
    string1=str(float(N_ret)/float(252))
    string2=str(Start_month)+'/'+str(Start_day)+'/'+str(Start_year)
    plt.xlabel(string1[0]+string1[1]+string1[2]+'-year time period starting from '+string2)


    fig = plt.figure()
    plt.scatter(domain,vclose1[0:N_ret], color='black',linewidth=2,label='Stock 1 closing')
    plt.scatter(domain,reg1[0:N_ret], color='red',linewidth=2,label='regression line')
    plt.legend()
    plt.xlabel('t-time (years)')
    plt.ylabel('$S(t)$')
    plt.title('Stock1 vs Regression')
    plt.show()

    fig = plt.figure()
    plt.scatter(domain,vclose2[0:N_ret], color='black',linewidth=2,label='Stock 2 closing')
    plt.scatter(domain,reg2[0:N_ret], color='red',linewidth=2,label='regression line')
    plt.legend()
    plt.xlabel('t-time (years)')
    plt.ylabel('$S(t)$')
    plt.title('Stock2 vs Regression')
    plt.show()



    Investment=1000.0

    Nsteps=60000

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

    
    print('This is how to distribute your cash:')
    for I in range(0,Nx):
        print(str.format('{0:.2f}',Investment*sigmoid(weights[I])*stock_value[I,0]/sum(temp_dist)))
        
        
