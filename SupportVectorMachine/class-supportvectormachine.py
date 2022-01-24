import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs



class SupportVectorMachine():
    
    def __init__(self, learn_rate = 0.001):
        
        self.bias_weight = None
        self.l_rate = learn_rate
        
    def pegasos(self,X, y):
        for i, j in enumerate(y):
            if j == 0:
                y[i] =-1
            elif j ==1:
                y[i] =1
        X = np.c_[np.ones(X.shape[0]),X]

        self.bias_weight = np.zeros(X.shape[1])

        order = np.arange(0,X.shape[0],1)

        t = 0
        
        margin_current = 0
        margin_previous = -10
        
        pos_support_vectors = 0
        neg_support_vectors = 0
        
        not_converged = True
        while(not_converged):
            t +=1
            pos_support_vectors = 0
            neg_support_vectors = 0
            
            margin_previous = margin_current
            nu = 1/(self.l_rate * t)
            
            
            random.shuffle(order)
            for i in order:
                prediction = np.dot(X[i],self.bias_weight)
                
                if (round((prediction),1) == 1):
                    pos_support_vectors += 1
                    
                if (round((prediction),1) == -1):
                    neg_support_vectors += 1
                
                if (y[i] * prediction ) < 1:
                    self.bias_weight = (1 - (nu * self.l_rate)) * self.bias_weight + nu * y[i]*X[i]
                else:
                    self.bias_weight = (1 - (nu * self.l_rate)) * self.bias_weight
            
                
            if (t>10000):
                margin_current = np.linalg.norm(self.bias_weight)
                if ((margin_current - margin_previous) < 0.01) and ((pos_support_vectors > 0)and(neg_support_vectors > 0)):
                    not_converged = False
        return t, self.bias_weight
    
    
X, y = make_blobs(n_samples = 40, centers =2, cluster_std = 1.2, n_features= 2, random_state =42)
    


SVM = SupportVectorMachine()

t, bias_weights = SVM.pegasos(X,y)
#bias_weights = np.array([0.25970647,  0.69353358, -0.0854183])
x = np.linspace(-5,7,100)

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(1,1,1)

ax.scatter(X[:,0],X[:,1],s=10)

ax.plot(x, -(bias_weights[0] + bias_weights[1]*x)/bias_weights[2])
plt.show()


#advance kernel SVM https://www.adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-training-algorithms/