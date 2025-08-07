#import sys, os; sys.path.append(os.path.dirname(os.getcwd())) 

import matplotlib.pyplot as plt
from ipywidgets import *

import numpy as np
from pyfrechet_k.metric_spaces import MetricData
from pyfrechet_k.metric_spaces.sphere import Sphere, r2_to_angle
from pyfrechet_k.regression.frechet_regression import GlobalFrechet, LocalFrechet
from pyfrechet_k.metric_spaces import MetricData, Wasserstein1D

from pyfrechet.regression.kernels import NadarayaWatson, gaussian, epanechnikov




def gen_sphere(N, m_type='contant', eps=0.1):
    M = Sphere(1)
#    M = Sphere(2)
#    M = Sphere(3)
    
    m = lambda _: 1
    if m_type == 'constant':
        m = lambda _: 1
    elif m_type == 'linear':
        # m = lambda x:  np.pi*x[:,0]*1.5 #+0.5
        # m = lambda x:  (x[:,0]+x[:,1]+x[:,2])*1
        # m = lambda x:  (x[:,0]<0.3)*x[:,1] + (x[:,0]>0.3)*(x[:,2]+2)

        p = 0.3
        m = lambda x:  (x[:,0]<p)*x[:,0] + (x[:,0]>p)*(x[:,0]*np.pi+2)

        m = lambda x:  1.0 * np.pi*x[:,0] + np.pi/2


    elif m_type == 'nonlinear':
        m = lambda x: 5 + 5*x[:,0] - 10*x[:,0]**2

    x = np.random.rand(N*3).reshape((N,3))
    x = np.random.rand(N).reshape((N,1))

    theta = m(x) +  eps*np.random.randn(N)
    y = np.c_[np.cos(theta), np.sin(theta)]
    return x, MetricData(M, y)







    

class AdaBoost:
    def __init__(self, data, learner, omega=None, t=5, dimY=100): #,losstype?):
        self.data = data # class Data
        self.learner = learner
        self.t = t
        self.t_max = t
        self.omega = omega
        self.dimY = dimY
        self.indeces = None
        self.trained_learner = [None for i in range(self.t)]
        self.weights = None
        self.betas = None
        self.ts  = None
        self.w = None
        self.losses = None
        
                 
    
    def newtrain(self, w):
        """ Generates new training data based on the calculated weights
        """
        n = self.data.shape[0]
        f = self.data.shape[1]
        idx = np.arange(0, n)
        newdata = np.zeros((n,f))
        ws = w/np.sum(w)


       
        idx_ = np.random.choice(idx, size=n, p = ws)
        newdata = self.data[idx_]

        
        
        

      

        if self.learner == GlobalFrechet or isinstance(self.learner(), LocalFrechet):
            return newdata[:,:-self.dimY], MetricData(self.omega, newdata[:,-self.dimY:]) 
        else:
            return newdata[:,:-self.dimY], newdata[:,-self.dimY:]
    
    def linearloss(self, loss):
        n = len(loss)
        lin = np.zeros(n)
        for i in range(n):
            lin[i] = np.abs(loss[i])/max(np.abs(loss))
    
        return lin

    def squareloss(self, loss):
        sqr = self.linearloss(loss)**2
    
        return sqr

    def exponentialloss(self, loss):
        exp = 1 - np.exp(-self.linearloss(loss))
    
        return exp
    
    def losstypef(self, loss, losstype = "linear"):
        if losstype == "linear":
            losses = self.linearloss(loss)
        elif losstype == "square":
            losses = self.squareloss(loss)
        elif losstype == "exponential":
            losses = self.exponentialloss(loss)
        else:
            print("Provide loss to be calculated")
        
        return losses
    
    def fit(self, losstype = "linear"):
        """ Fitting
            Takes a learner method and a data set
            Trains t learners on newly generated datasets.
            calculates 
               - betas (scaled avg. loss for all learners)
               - ts (predictions for all learners) 
               - w (weights for all learners)
        """
        n = self.data.shape[0]
        w = np.ones(n)/n # assigning weights
        ts = np.zeros((n, self.dimY, self.t_max)) 
        betas = np.zeros(self.t_max)
        weights = np.zeros((n, self.t_max))
        self.losses = np.zeros((n, self.t_max))
        y_p = np.zeros((n, self.dimY))
        loss1s = np.zeros(n)
        avrlosss=np.zeros(self.t_max)
        

        for i in range(self.t_max):
        # 1 generating new data
            newdatax, newdatay = self.newtrain(w)
        
            losss = []
        
            # 2 train new learner on data
         
            self.trained_learner[i] = self.learner().fit(newdatax, newdatay)
        
            for j in range(n):
   
                y_p[j] = self.trained_learner[i].predict(np.expand_dims(self.data[j,:-self.dimY], axis=0)) # generalize to "learner"
         
                ts[j, :, i] = y_p[j]
                

                if self.omega != None:
                    losss.append(self.omega.d(y_p[j], self.data[j, -self.dimY:]))
               
                else:
                    losss.append(np.abs(y_p[j] - self.data[j, -self.dimY:]))
                
            loss1s = self.losstypef(losss, losstype)
            self.losses[:, i] = loss1s

            # 5 calculating average loss       
            avrlosss = 0
            for j in range(n):
                
                avrlosss += loss1s[j] * (w[j]/np.sum(w))
        
            if avrlosss > 0.5: #stop when average loss is above 0.5
                if i > 1: #can only stop 2nd iteration
                    self.t = i - 1
                    betas = betas[:self.t]
                    ts = ts[:, :, :self.t]
                    w # ok, doesn't depend on t
                    weights = weights[:, :self.t]
                    break

            # 6 calculating beta    
            betas[i] = avrlosss/(1-avrlosss)

            # 7 updating weights
            for k in range(n):
                w[k] = w[k]*betas[i]**(1-loss1s[k])
            weights[:, i] = w


            
                 
        self.betas, self.ts, self.w, self.weights = betas, ts, w, weights  
        return betas, ts, w 
    
                 
    def weighted_median(self, x):
        """ Prediction Method
            
            Assumes x has n observations 
        """
        
        def weighted_median_(x):
            y_p = np.zeros(self.t)
            for i in range(self.t):
                y_p[i] = self.trained_learner[i].predict(x)

            sortedidx = np.argsort(y_p)
            sorted_y_p = y_p[sortedidx]
            sortedbeta = self.betas[sortedidx]

            for i in range(self.t):
                betasum = np.sum(np.log(1/sortedbeta[:i]))
                if betasum >= 0.5 * np.sum(np.log(1/self.betas)):
                    break
            return sorted_y_p[i]
        
        n = x.shape[0]
        ensemble_prediction = np.zeros(n)
        for i in range(n):
            ensemble_prediction[i] = weighted_median_(x[i])
            
        return ensemble_prediction
        
        


        
 

    
    
    def weighted_mean(self, x):
        """ Prediction Method
        """
        fm = np.zeros((x.shape[0], self.dimY))
        expectation = np.zeros((self.t, x.shape[0], self.dimY))
        w = np.zeros((self.t))
        for i in range(self.t):
            preds = self.trained_learner[i].predict(x[:,:-self.dimY]).data
            expectation[i,:,:] = preds 
     
            w = np.log(1/self.betas)/np.sum(np.log(1/self.betas))

#  
        for j in range(x.shape[0]):
            expectmetric = MetricData(self.omega, expectation[:,j,:])
#          
            fm[j,:]= expectmetric.frechet_mean(w)
#           

        return MetricData(self.omega, fm), expectation    
      
    
    
    def mean_weight2(self, x):
        trained_weights = np.zeros((x.shape[0], self.t))
        for i in range(self.t):
            for j in range(x.shape[0]):
                print(self.trained_learner[i].weights_for(x[j, :-self.dimY]))
                trained_weights[j, i] = self.trained_learner[i].weights_for(x[j, :-self.dimY])
        avrweights = np.mean(trained_weights, axis=1)
        ytrain = MetricData(self.omega, self.data[:, -self.dimY:])
        return ytrain.frechet_mean(avrweights)
        

    def mean_weight(self, xs):
        n_xs = xs.shape[0]
        preds = np.zeros((n_xs, self.dimY))
  
        for i in range(n_xs):
            x = xs[i]
            trained_weights = np.zeros((self.data.shape[0], self.t)) 
            for j in range(self.t):
                trained_weights[:, j] = self.trained_learner[j].weights_for(x[:-self.dimY])
            
            avrweights = np.mean(trained_weights, axis=1) 
            
#          
            ytrain = MetricData(self.omega, self.data[:, -self.dimY:])
          
            preds[i,:] = ytrain.frechet_mean(avrweights)

        preds_metric = MetricData(self.omega, preds)
        return preds_metric, None
#    
        
        
        
    
    def predict(self, x, method="median"):
        """ Prediction
            Makes prediction based on the t learners
        """
        
        if method == "median":
            etellerandet = self.weighted_median(x)
        elif method == "mean_fm":
            etellerandet = self.weighted_mean(x)
        elif method == "mean_weight":
            etellerandet = self.mean_weight(x)   
        else:
            print("Provide a correct method")
            
        return etellerandet 
    #self.learner.predict()

        
# myBooster = AdaBoost(data=data, learner=bl.binarytreeReg, t=5) # example constructor
# myBooster.fit()
# myBooster.predict(data)




  


