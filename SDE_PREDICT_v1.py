
import sys
sys.path.insert(0, '../../Utilities/')
import scipy.io
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import savemat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow_probability as tfp
import math


#Set random seed.
#global_seed = 42
#N_chains = 5
#np.random.seed(global_seed)
#seeds = np.random.randint(0, 429, size=N_chains)


def coeff_SDE(ww,N_layers,N_terms):

    if N_layers==0:
        c=ww[N_layers]
        Const=c[2]
        Lin_y=c[0]
        Lin_t=c[1]
        y_2=0
        yt=0
        t_2=0
    elif N_layers==1:
        c0=ww[0]
        c1=ww[1]
        Const=c1[N_layers+N_terms]+c1[1]*c0[0,2]*c0[1,2]
        Lin_y=c1[0]+c1[2]*(c0[0,0]*c0[1,2]+c0[1,0]*c0[0,2])
        Lin_t=c1[1]+c1[2]*(c0[0,1]*c0[1,2]+c0[1,1]*c0[0,2])
        y_2=c1[2]*(c0[0,0]*c0[1,0])
        yt=c1[2]*(c0[0,1]*c0[1,1])
        t_2=c1[2]*(c0[0,1]*c0[1,0]+c0[0,0]*c0[1,1])
    return Const, Lin_y,Lin_t,y_2,yt,t_2


def histogram_create(Normal_distribution,N_space,n_c,Nt):
    histogram_domain = np.linspace(-n_c,n_c,N_space)
    histogram=np.zeros([N_space],dtype=float)
    for I in range(0,N_space):
        for J in range(0,Nt):
            if ((-n_c+n_c/(N_space/2)*I<=Normal_distribution[J]) & (Normal_distribution[J]<=-n_c+n_c/(N_space/2)*(I+1))):
                histogram[I]=histogram[I]+1
    return histogram_domain, histogram





def Normal_CDF(inpt):
    out=0.5*(1+math.erf(inpt/np.sqrt(2)))
    return out



class SDE_CLASSIFY:
    def __init__(self,t,y_mean,y_var,B,Nt,N_terms,N_layers,Losss):

        self.t = t
        self.y_mean = y_mean
        self.y_var = y_var
        self.B = B
        self.Losss=Losss

        self.Nt=Nt
        self.N_terms=N_terms
        self.Num_sim=Num_sim
        self.N_layers=N_layers

        self.weights_mu, self.weights_sigma = self.initialize_NN(N_terms,N_layers)

        self.y_tf = tf.placeholder(tf.float32, shape=[Nt,1])

        self.mean_pred, self.var_pred, self.weights_mu_tf, self.weights_sigma_tf = self.net_uv(self.y_mean,self.B)

        # Loss
        gamma=0.00001
        self.loss = tf.reduce_mean(tf.square(self.mean_pred-self.y_mean[:,0]))+tf.reduce_mean(tf.square(self.var_pred-self.y_var[:,0]))+gamma*(tf.reduce_mean(tf.abs(self.weights_mu_tf[0]))+tf.reduce_mean(tf.abs(self.weights_sigma_tf[0])))


    # Optimizers
        self.optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)
        

    def initialize_NN(self,N_terms, N_layers):
        weights_mu=[]
        scale=0.1
        for I in range(0,N_layers):
            W=scale*tf.Variable(tf.random.uniform([2,N_terms+1+I],dtype=tf.float32),dtype=tf.float32)
            weights_mu.append(W)
        W=scale*tf.Variable(tf.random.uniform([N_terms+1+N_layers],dtype=tf.float32),dtype=tf.float32)
        weights_mu.append(W)
        
        weights_sigma=[]
        for I in range(0,N_layers):
            W=scale*tf.Variable(tf.random.uniform([2,N_terms+1+I],dtype=tf.float32),dtype=tf.float32)
            weights_sigma.append(W)
        W=scale*tf.Variable(tf.random.uniform([N_terms+1+N_layers],dtype=tf.float32),dtype=tf.float32)
        weights_sigma.append(W)

        return weights_mu, weights_sigma


    def RNN_mu(self,y,t,w,N_terms,N_layers):
        if N_layers==0:
            W=w[0]
            F_rhs=W[0]*y+W[1]*t+W[2]
        else:
            tf_list=[]
            tf_list.append(y)
            tf_list.append(t)
            for I in range(0,N_layers):
                R=0.0*y
                L=0.0*y
                W=w[I]
                for J in range(0,N_terms+I):
                    R=W[1,J]*tf_list[J]+R
                    L=W[0,J]*tf_list[J]+L
                R=R+W[1,N_terms+I]
                L=L+W[0,N_terms+I]
                B=L*R
                tf_list.append(B)
            W=w[N_layers]
            s=0*y
            for I in range(0,N_layers+N_terms):
                s=s+tf_list[I]*W[I]
            F_rhs=s
        return F_rhs,w



    def RNN_sigma(self,y,t,w,N_terms,N_layers):
        if N_layers==0:
            W=w[0]
            F_rhs=W[0]*y+W[1]*t+W[2]
        else:
            tf_list=[]
            tf_list.append(y)
            tf_list.append(t)
            for I in range(0,N_layers):
                R=0.0*y
                L=0.0*y
                W=w[I]
                for J in range(0,N_terms+I):
                    R=W[1,J]*tf_list[J]+R
                    L=W[0,J]*tf_list[J]+L
                R=R+W[1,N_terms+I]
                L=L+W[0,N_terms+I]
                B=L*R
                tf_list.append(B)
            W=w[N_layers]
            s=0*y
            for I in range(0,N_layers+N_terms):
                s=s+tf_list[I]*W[I]
            F_rhs=s
        return F_rhs,w
    
    def net_uv(self,y_mean,B):
        temp=tf.ones([1,Num_sim],dtype=tf.float32)
        
        """
        #Computing Mean
        Sim=temp*y[0,:]
        for I in range(0,Nt-1):
            F_mu, weights_mu = self.RNN_mu(Sim[I,:],t[I],self.weights_mu,self.N_terms,self.N_layers)
            F_sigma, weights_sigma = self.RNN_sigma(Sim[I,:],t[I],self.weights_sigma,self.N_terms,self.N_layers)
#            Temp=[Sim[I,:]]+F_mu*dt+F_sigma*(np.sqrt(dt)*tf.convert_to_tensor(B[I,:], np.float32))
            Temp=[Sim[I,:]]+F_mu*dt+F_sigma*np.sqrt(dt)*B[I,:]
            Sim=tf.concat([Sim,Temp],0)
        reduce_mean=tf.math.reduce_sum(Sim,1)/Num_sim

        #Computing Variance
        temp=tf.zeros([1,Num_sim],dtype=tf.float32)
        Sim_var=temp*y[0,:]
        for I in range(1,Nt):
            Temp=(y_mean[I,0]-Sim[I,:])**2
            Sim_var=tf.concat([Sim_var,[Temp]],0)
        reduce_var=tf.math.reduce_sum(Sim_var,1)/(Num_sim*Nt*dt)
        """



        #Computing Mean
        Sim=temp*y[0,:]
        temp=tf.zeros([1,Num_sim],dtype=tf.float32)
        Sim_var=temp*y[0,:]
        for I in range(0,Nt-1):
            F_mu, weights_mu = self.RNN_mu(Sim[I,:],t[I],self.weights_mu,self.N_terms,self.N_layers)
            F_sigma, weights_sigma = self.RNN_sigma(Sim[I,:],t[I],self.weights_sigma,self.N_terms,self.N_layers)
            Temp=[Sim[I,:]]+F_mu*dt+F_sigma*np.sqrt(dt)*B[I,:]
#            Temp=[Sim[I,:]]+F_mu*dt+F_sigma*np.sqrt(dt)*B[I,:]
            Sim=tf.concat([Sim,Temp],0)
            Temp=(y_mean[I,0]-Sim[I,:])**2
            Sim_var=tf.concat([Sim_var,[Temp]],0)
        reduce_mean=tf.math.reduce_sum(Sim,1)/Num_sim
        reduce_var=tf.math.reduce_sum(Sim_var,1)/(Num_sim*Nt*dt)



        return reduce_mean,reduce_var,weights_mu,weights_sigma
    
    
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter):
        tf_dict = {self.y_tf: self.y_mean}
        
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

    def predictrho(self,y_mean):
        tf_dict = {self.y_tf: self.y_mean}
        y_pred=self.sess.run(self.mean_pred)
        y_pred_var=self.sess.run(self.var_pred)
        weights_mu=self.sess.run(self.weights_mu_tf)
        weights_sigma=self.sess.run(self.weights_sigma_tf)
        return y_pred, y_pred_var, weights_mu, weights_sigma

    
if __name__ == "__main__":
    
    TTdata = scipy.io.loadmat('data_SDE.mat') #This has simulations, mean, std, and other parameters.
#    Tdata=scipy.io.loadmat('data_SDE_density.mat') #This has the non smooth data recorded distribution
#    TTTdata=scipy.io.loadmat('data_SDE_density.mat') #This has the smooth Neural Net probability distribution

#    savemat('Prob_distribution.mat',{'V':V, 'Nx':Nx, 'Ny':Ny})


    
#    VV = TTTdata['V']
    B = TTdata['B']
    t = TTdata['t']
    y = TTdata['y']
    y_mean = TTdata['y_mean']
    y_var = TTdata['y_var']
    dt = TTdata['dt']
    Nt = TTdata['Nt']
    Num_sim = TTdata['Num_sim']
    a = TTdata['a']
    b=TTdata['b']


    dt=dt[0,0]
    Nt=Nt[0,0]
    Num_sim=Num_sim[0,0]
    a=a[0,0]
    b=b[0,0]


    
#    savemat('data_SDE_density.mat',{'rho':rho,'t':t,'dt':dt ,'Nt':Nt, 'Nx':Nx,'aa':aa, 'bb':bb,'t1':t1})
#    rho=Tdata['rho']
#    Nx=Tdata['Nx']

#    [Nx,Ny]=rho.shape


#    aa=Tdata['aa']
#    bb=Tdata['bb']
#    t1=Tdata['t1']
#    dx=Tdata['dx']


#    layers = [2, 5, 5, 5, 5, 5, 5, 1]#
#    ax=aa
#    bx=bb
    
#    Nx=Nx[0,0]
#    dx=dx[0,0]
#    ay=t1[0,0]
#    by=1



    """
    #Defining grid points (x,y)
    x=np.zeros([Nx,1], dtype=float)
    for I in range(0,Nx):
        x[I,0] = ax+(bx-ax)/(Nx-1)*I

    y=np.zeros([Ny,1], dtype=float)
    for I in range(0,Ny):
        y[I,0] = ay+(by-ay)/(Ny-1)*I        

    V=np.zeros([Nx,Ny],dtype=float)
    Vx=np.zeros([Nx,Ny],dtype=float)
    Vy=np.zeros([Nx,Ny],dtype=float)
    for I in range(0,Nx):
        for J in range(0,Ny):
#            V[I,J]=rho[I,J]
            V[I,J]=VV[I,J]
#            Vx[I,J]=np.roll(rho,1,0)-np.roll(rho,1,0)
#            Vy[I,J]=np.pi*np.sin(np.pi*x[I,0])*np.cos(np.pi*y[J,0])
            
    x_flat=np.zeros([Nx*Ny,1],dtype=float)
    y_flat=np.zeros([Nx*Ny,1],dtype=float)
    V_flat=np.zeros([Nx*Ny,1],dtype=float)
    L=0
    for I in range(0,Nx):
        for J in range(0,Ny):        
            x_flat[L,0]=x[I,0]
            y_flat[L,0]=y[J,0]
            V_flat[L,0]=V[I,J]
            L=L+1

    #This is what Nsteps does
    Nsteps=60000

    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I

    model = NeuralNetworkCode(x_flat,y_flat,V_flat,layers,Losss)
        
    start_time = time.time()
    model.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    V_pred, Vx_pred, Vy_pred = model.predict()


    LLosss=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        LLosss[I,0]=np.log(Losss[I,0])

    fig = plt.figure()
    plt.scatter(Losss_domain,LLosss, color='black', linewidth=2)
    

    V_pred_3d = np.zeros([Nx,Ny],dtype=float)
    Vx_pred_3d = np.zeros([Nx,Ny],dtype=float)
    Vy_pred_3d = np.zeros([Nx,Ny],dtype=float)

    L=0
    for I in range(0,Nx):
        for J in range(0,Ny):
            V_pred_3d[I,J]=V_pred[L,0]
            Vx_pred_3d[I,J]=Vx_pred[L,0]
            Vy_pred_3d[I,J]=Vy_pred[L,0]
            L=L+1

    x_mesh,y_mesh=np.meshgrid(y,x)
    
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(x_mesh,y_mesh, V_pred_3d, color='r')
    plt.title('Neural Network Solution',fontsize=14,fontweight='bold')
    
    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(x_mesh,y_mesh, V, color='r')
    plt.title('Probability Density Simulation',fontsize=14,fontweight='bold')




    partial_x=(np.roll(V_pred_3d,1,0)-np.roll(V_pred_3d,-1,0))/(2*dx)
    partial_t=(np.roll(V_pred_3d,1,1)-np.roll(V_pred_3d,-1,1))/(2*dt)
    
    
    [NNc,NNNc]=partial_t.shape
    partial_t[:,0]=0
    partial_t[:,NNNc-1]=0

    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(x_mesh,y_mesh, Vy_pred_3d, color='r')
    plt.title('partial in t using NN',fontsize=14,fontweight='bold')

    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(x_mesh,y_mesh, partial_t, color='r')
    plt.title('partial in t using FD',fontsize=14,fontweight='bold')

    fig = plt.figure()
    ax2 = plt.axes(projection='3d')
    ax2.plot_wireframe(x_mesh,y_mesh, partial_t-Vy_pred_3d, color='r')
    plt.title('partial in t using FD',fontsize=14,fontweight='bold')
    """







    
    
    N_layers=0
    N_terms=2


    Nsteps=50000
    Losss=np.zeros([Nsteps,1],dtype=float)
    Losss_domain=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        Losss_domain[I,0]=I
               

    #Computing time derivative using Legendre polynomials.
    L=np.zeros([Nt,10],dtype=float)
    L[:,0]=1+0*t[:,0]
    L[:,1]=t[:,0]
    L[:,2]=0.5*(3*t[:,0]*t[:,0]-1)
    L[:,3]=1.0/2.0*(5*t[:,0]**3-3*t[:,0])
    L[:,4]=1.0/8.0*(35*t[:,0]**4-30*t[:,0]**2+3)
    L[:,5]=1.0/8.0*(63*t[:,0]**5-70*t[:,0]**3+15*t[:,0])
    L[:,6]=1.0/16.0*(231*t[:,0]**6-315*t[:,0]**4+105*t[:,0]**2-5)
    L[:,7]=1.0/16.0*(429*t[:,0]**7-693*t[:,0]**5+315*t[:,0]**3-35*t[:,0])
    L[:,8]=1.0/128.0*(6435*t[:,0]**8-12012*t[:,0]**6+6930*t[:,0]**4-1260*t[:,0]**2+35)
    L[:,9]=1.0/128.0*(12155*t[:,0]**9-25740*t[:,0]**7+18018*t[:,0]**5-4620*t[:,0]**3+315*t[:,0])

    N_size=10
    A=np.zeros([Nt,N_size],dtype=float)
    for I in range(0,N_size):
        A[:,I]=L[:,I]
    
    #solving the normal equations:
    alpha = np.linalg.lstsq(A, y_mean, rcond=None)[0]
    
    y_fit_mean=np.zeros([Nt,1],dtype=float)
    for I in range(0,N_size):
        y_fit_mean[:,0]=y_fit_mean[:,0]+alpha[I]*L[:,I]

    alpha = np.linalg.lstsq(A, y_var, rcond=None)[0]
    
    y_fit_var=np.zeros([Nt,1],dtype=float)
    for I in range(0,N_size):
        y_fit_var[:,0]=y_fit_var[:,0]+alpha[I]*L[:,I]



    """
    y_mean_matrix=np.zeros([Nt,Num_sim],dtype=float)
    y_var_matrix=np.zeros([Nt,Num_sim],dtype=float)
    for J in range(0,Num_sim):
        for I in range(0,Nt):
#            y_mean_matrix[I,J]=y_mean[I,0]
            y_mean_matrix[I,J]=y_fit_mean[I,0]
            y_var_matrix[I,J]=y_fit_var[I,0]
#            y_var_matrix[I,J]=y_var[I,0]
#    skip
    """

    modelrho = SDE_CLASSIFY(t,np.float32(y_fit_mean),np.float32(y_fit_var),B,Nt,N_terms,N_layers,Losss)

    start_time = time.time()
    modelrho.train(Nsteps)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    y_pred, y_pred_var, weights_mu, weights_sigma = modelrho.predictrho(y)

    LLosss=np.zeros([Nsteps,1],dtype=float)
    for I in range(0,Nsteps):
        LLosss[I,0]=np.log(Losss[I,0])

    fig = plt.figure()
    plt.scatter(Losss_domain,LLosss, color='black', linewidth=2, label='Log of Loss vs Number of Iterations')
    plt.show()
    plt.legend()
#    fig.savefig('LossBDF2_example1.png')

#    savemat('Loss1.mat',{'L_domain':Losss_domain, 'Loss_y':LLosss})


#    eps=(np.tanh(w_eps) + 1)*(max_epsilon-min_epsilon)/2+min_epsilon


    Cons_mu,Lin_y_mu,Lin_t_mu,y_2_mu,ty_mu,t_2_mu=coeff_SDE(weights_mu,N_layers,N_terms)


    Cons_sigma,Lin_y_sigma,Lin_t_sigma,y_2_sigma,ty_sigma,t_2_sigma=coeff_SDE(weights_sigma,N_layers,N_terms)

    print(Cons_mu)
    print(Lin_y_mu)
    print(Lin_t_mu)
    
    print(Cons_sigma)
    print(Lin_y_sigma)
    print(Lin_t_sigma)
    
    fig = plt.figure()
    plt.scatter(t,y_pred, color='black', linewidth=2,label='Fitted Mean')
    plt.title('Fitted Mean')

    fig = plt.figure()
    plt.scatter(t,y_fit_mean, color='black', linewidth=2, label='Mean from Observations')
    plt.title('Mean from Observations')


    fig = plt.figure()
    plt.scatter(t,y_pred_var, color='black', linewidth=2)
    plt.title('Fitted Variance')

    fig = plt.figure()
    plt.scatter(t,y_fit_var, color='black', linewidth=2)
    plt.title('Variance from Observations')




