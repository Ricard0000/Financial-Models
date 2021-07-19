"""
This is a work in progress, not all simulations are here
Brownian motion simulations:
-    Basic simulations
-    Quadratic variation of Brownian motion
-    Stopping time
-    Exit times
-    Maximum and minimum of Brownian motion
-    Distribution of hitting times
-    Reflection principle
-    zeros of Brownian motion
"""
import sys
sys.path.insert(0, '../../Utilities/')
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import time
import math


#Definition of the cumulative distribution function
def Normal_CDF(inpt):
    out=0.5*(1+math.erf(inpt/np.sqrt(2)))
    return out

def histogram_create(Normal_distribution,N_space,n_c):
    histogram_domain = np.linspace(-n_c,n_c,N_space)
    histogram=np.zeros([N_space],dtype=float)
    for I in range(0,N_space):
        for J in range(0,Nt):
            if ((-n_c+n_c/(N_space/2)*I<=Normal_distribution[J]) & (Normal_distribution[J]<=-n_c+n_c/(N_space/2)*(I+1))):
                histogram[I]=histogram[I]+1
    return histogram_domain, histogram

if __name__ == "__main__": 

    
    Ts=0.0#Start time is zero
    Tf=1.0 #final time is Tf    

#Creating t_i, from Ts to Tf
    Nt=10000
    t = np.linspace ( Ts, Tf, Nt)#time axis

    #selecting Nt random numbers from the normal distribution 
    Normal_distribution=np.zeros([Nt],dtype=float)
    for I in range ( 0, Nt ):
        Normal_distribution[I] = np.random.normal(loc=0.0, scale=1.0, size=None)
    
    n_a=np.min(Normal_distribution)
    n_b=np.max(Normal_distribution)
    n_c=max(abs(n_a)+1,abs(n_b)+1)
    
    #Use this code to plot a histogram: the distribution of the random normal numbers
    N_space=20 #(Should be less than Nt)
    histogram_domain, histogram=histogram_create(Normal_distribution,N_space,n_c)
    
    #Plot of the distribution
    fig = plt.figure()
    plt.scatter(histogram_domain,histogram, color='black', linewidth=2)
    plt.title('Distribution of Normal')
    plt.xlabel('Domain')
    plt.ylabel('Number of Points within selected spacing')





    #Defining a Brownian motion with standard deviation of sigma 
    #and starting point at x0
    sigma=1.0
    x0=5
    B=np.zeros([Nt],dtype=float)
    
    for I in range(0,Nt-1):
        B[I+1]=B[I]+sigma/np.sqrt(Nt)*Normal_distribution[I]
    B=B+x0

    fig = plt.figure()
    plt.scatter(t,B, color='black', linewidth=2)
    plt.title('Brownian Motion')






    #Computing the Quadratic Variation of B at selected times
    N_QV=20 #number of selected times
    t_QV = np.linspace ( Ts, Tf, N_QV)#time axis for Quadratic Variation from Ts to Tf
    QV=np.zeros([N_QV],dtype=float)
    space=int(Nt/N_QV)
    
    #Definition of Quadratic Variation
    for I in range(0,N_QV-1):
        for J in range(0,(I+1)*space):
            QV[I+1]=abs(B[J+1]-B[J])**2+QV[I+1]
    #Plot of Quadratic Variation (Should be aproximately linear with slope 1, starting at zero)
    fig = plt.figure()
    plt.scatter(t_QV,QV, color='black', linewidth=2)
    plt.title('Quadratic Variation')





#while the above code 

    #At what time does it reach x0 +/- 1?







