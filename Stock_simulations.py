"""
Brownian motion simulations of Stocks

Random Walk simulation based on these statistics:
   s_{i+1}=s_{i}(1+mu*dt+sigma*phi*dt^(1/2))
   
where s is the stock values. Here we use s=closing value
I do not claim this is what is actually happening
in real life, but its the model Stochastic differential
equation I choose to work with.

I also plot the distribution of returns: This should
look approximately normal (In theory)   
"""
import sys
sys.path.insert(0, '../../Utilities/')
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import pandas_datareader as pdr
import datetime as dt
import matplotlib.patches as patches
import pandas as pd

#Definition of the cumulative distribution function
def Normal_CDF(inpt):
    out=0.5*(1+math.erf(inpt/np.sqrt(2)))
    return out

def histogram_create(Normal_distribution,N_space,n_c,Nt):
    histogram_domain = np.linspace(-n_c,n_c,N_space)
    histogram=np.zeros([N_space],dtype=float)
    for I in range(0,N_space):
        for J in range(0,Nt):
            if ((-n_c+n_c/(N_space/2)*I<=Normal_distribution[J]) & (Normal_distribution[J]<=-n_c+n_c/(N_space/2)*(I+1))):
                histogram[I]=histogram[I]+1
    return histogram_domain, histogram/np.sqrt(Nt)

if __name__ == "__main__": 


    #Loading AAPL data 252 day/year
    #ticker = ["AAPL", "IBM", "TSLA"]
#    ticker = ["AAPL"]
    ticker = ["AAPL"]

    Start_year=2019
    Start_month=1
    Start_day=1
    
    End_year=2021
    End_month=1
    End_day=1
    
    
    start = dt.datetime(Start_year, Start_month, Start_day)
    end = dt.datetime(End_year, End_month, End_day)




#    start = dt.datetime(1995, 2, 1)
#    end = dt.datetime(1996, 11, 1)

    data = pdr.get_data_yahoo(ticker, start, end)
    v=data.values #This is numeric/floating point data with no strings
    vclose=v[:,1]#Closing price
    vs=np.roll(vclose,1,0)#Shift of the Closing Price for differncing purposes
    vs[0]=0.0
    
    N_ret=vclose.size



#The returns are defined by RETURN_{i} = (R_{i+1}-R_{i})/(R_{i})
AAPL_ret=np.zeros([N_ret-1],dtype=float)
for I in range(0,N_ret-1):
    AAPL_ret[I]=(vclose[I+1]-vclose[I])/vclose[I]





domain=np.linspace(0,N_ret/252,vclose.size-1)

dt=1/252 # Usually, there is a 252 days/year data. Hence the reason for 1/252
#Mean of the Returns: 1/M*sum(R_{i})
mu=np.sum(AAPL_ret)/((N_ret-1)*dt)
#Mean and mu are related by: Mean = Mu*dt
Mean=mu*dt

#Plotting the distribution of returns(Should appear like a Normal Distribution)
plt.style.use('fivethirtyeight')
plt.figure()
plt.hist(AAPL_ret,bins=60,edgecolor='black')
plt.title('Distribution of AAPL returns')
string1=str(float(N_ret)/float(252))
string2=str(Start_month)+'/'+str(Start_day)+'/'+str(Start_year)
plt.xlabel(string1[0]+string1[1]+string1[2]+'-year time period starting from '+string2)


#n_a=np.min(AAPL_ret)
#n_b=np.max(AAPL_ret)
#n_c=max(abs(n_a),abs(n_b))
#n_c=n_c+n_c/8
#histogram_domain, histogram=histogram_create(AAPL_ret,60,n_c,N_ret-1)
#fig = plt.figure()


#plt.bar(range(len(histogram_domain)), histogram)
#plt.title('Distribution of AAPL returns')
#plt.xlabel('2-year time period')
#plt.ylabel('Number of Returns within chosen interval size')








#size1,size2=data.shape
#Computing Standard Deviation: sqrt{1/((M-1)dt)*sum(R_{i}-Mean)}
s=0
for I in range(0,N_ret-1):
    s=(AAPL_ret[I]-Mean)**2+s
AAPL_std=np.sqrt(s/((N_ret-2)*dt))



#Random Walk simulation based on these statistics:
#   s_{i+1}=s_{i}(1+mu*dt+sigma*phi*dt^(1/2))   #

#Setting starting value
#This is a single simulation of a Brownian motion with starting value
#S(0)=AAPL/ticker[0] and mean and standard deviation given by AAPL/ticker
simulation=np.zeros([N_ret],dtype=float)
simulation[0]=vclose[0]
for I in range(0,N_ret-1):
    simulation[I+1]=simulation[I]*(1+mu*dt+np.sqrt(dt)*AAPL_std*np.random.normal())


fig = plt.figure()
plt.scatter(domain,simulation[1:N_ret], color='black',linewidth=2)
plt.legend()
plt.xlabel('t-time (years)')
plt.ylabel('$B(t)$')
plt.title('Brownian motion simulation')
plt.show()














"""
#The code here plots Brownian motion simulations starting at
#time 0-T_f. This is just extra information since we already have
#the data at that time.

d=10 #Number of simulations
multiple_simulation=np.zeros([N_ret,d],dtype=float)
multiple_simulation[0,:]=vclose[0]
for I in range(0,N_ret-1):
    multiple_simulation[I+1,:]=multiple_simulation[I,:]*(1+mu*dt+np.sqrt(dt)*AAPL_std*np.random.normal(size=(1,d)))


fig = plt.figure()
plt.title('Single simulation of the SDE vs Exact Stock')
plt.scatter(domain,vclose[1:N_ret], color='black', linewidth=2,label='Exact Stock Value')
plt.scatter(domain,simulation[1:N_ret], color='red', linewidth=2,label='Single Simulation')
plt.legend()
plt.xlabel('2-year time period')
plt.ylabel('Value')
plt.show()


fig = plt.figure()
for I in range(0,d):
    plt.plot(domain,multiple_simulation[1:N_ret,I],linewidth=1)
plt.scatter(domain,vclose[1:N_ret], color='black', linewidth=2,label='Exact Stock Value')
plt.title('Multiple simulation of the SDE vs Exact Stock')
plt.legend()
plt.xlabel('2-year time period')
plt.ylabel('Value')
plt.show()
"""


#Brownian Motion simulation for predicting future values "Day" days in advance:
Day=50 #Days_in_the_future
string3=str(Day)
d=1 #Number of simulations

future_simulations=np.zeros([N_ret+Day,d],dtype=float)
domain=np.linspace(0,(N_ret+Day)/252,N_ret+Day)
string1=str(float(N_ret+Day)/float(252))


"""

for I in range(0,d):
    future_simulations[0:N_ret,I]=vclose

for I in range(0,Day):
    future_simulations[N_ret+I,:]=future_simulations[N_ret+I-1,:]*(1+mu*dt+np.sqrt(dt)*AAPL_std*np.random.normal(size=(1,d)))

fig = plt.figure()
"""
last_val=vclose[N_ret-1]*np.ones([N_ret+Day],dtype=float)


average_pred=np.zeros([N_ret+Day],dtype=float)
average_pred[0:N_ret]=vclose

plus_one_std=np.zeros([N_ret+Day],dtype=float)
plus_one_std[0:N_ret]=vclose

minus_one_std=np.zeros([N_ret+Day],dtype=float)
minus_one_std[0:N_ret]=vclose

plus_two_std=np.zeros([N_ret+Day],dtype=float)
plus_two_std[0:N_ret]=vclose

minus_two_std=np.zeros([N_ret+Day],dtype=float)
minus_two_std[0:N_ret]=vclose


for I in range(0,Day):
    average_pred[N_ret+I]=average_pred[N_ret+I-1]*(1+mu*dt)
    plus_one_std[N_ret+I]=plus_one_std[N_ret+I-1]*(1+(mu+AAPL_std)*dt)
    minus_one_std[N_ret+I]=minus_one_std[N_ret+I-1]*(1+(mu-AAPL_std)*dt)
    plus_two_std[N_ret+I]=plus_two_std[N_ret+I-1]*(1+(mu+2*AAPL_std)*dt)
    minus_two_std[N_ret+I]=minus_two_std[N_ret+I-1]*(1+(mu-2*AAPL_std)*dt)




fig = plt.figure()
plt.plot(domain,last_val,linewidth=1,color='red',label='Last Value')
for I in range(0,d):
    plt.plot(domain,future_simulations[:,I],linewidth=1)
plt.plot(domain,plus_one_std,linewidth=2,color='blue',label='plus one standard deviation')
plt.plot(domain,minus_one_std,linewidth=2,color='blue',label='minus one standard deviation')
plt.plot(domain,plus_two_std,linewidth=2,color='green',label='plus two standard deviation')
plt.plot(domain,minus_two_std,linewidth=2,color='green',label='minus two standard deviation')
plt.plot(domain,average_pred,linewidth=2,color='black',label='average_prediction')
plt.legend(bbox_to_anchor =(0.5, 1.75))
#plt.legend()
plt.title('Brownian motion based prediction '+string3+' days from last value')
plt.xlabel(string1[0]+string1[1]+string1[2]+'-year time period starting from '+string2)
plt.ylabel('Value')
plt.show()






#fig = plt.figure()
#plt.scatter(domain,plus_one_std,linewidth=2, color='black', linewidth=2)
##plt.scatter(domain,vclose[1:N_ret], color='black', linewidth=2)
#plt.legend()
#plt.xlabel('t-time')
#plt.ylabel('$B(t)$')
#plt.show()













