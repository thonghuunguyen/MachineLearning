import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 10.0)
from mpl_toolkits import mplot3d


from sklearn.model_selection import train_test_split
from datetime import datetime
start_time = datetime.now()

class MSE_Gradient_Descent():
    
    def __init__(self):
        
        self.bias_weights = None
        self.log_bias_weights = None
        self.log_mean_squared_error = None
        
    def predict(self, X, theta):
        y_predicted = np.dot(X, theta.T)
        
        return y_predicted
    
    #Cost/Loss function: Mean squared error
    def cost_function(self, actual, predicted, X):
        
        m = len(actual)
        
        mse = np.sum(np.square(predicted - actual))/(2*m)
        
        return mse

    #Partial derivative of Cost/loss function with respect to bias and wieghts
    
    
    #Gradent Descent 
    def gradient_descent(self, actual, X, n_iterations, learn_rate = 0.1):
        
        self.bias_weights = np.array(np.zeros(X.shape[1]))
        self.bias_weights = self.bias_weights.reshape(1,X.shape[1])
        
        self.log_bias_weights = []
        self.log_mean_squared_error = []
        
        m = len(actual)
        # predicted = np.dot(X, self.bias_weights.T)
        
        # error = predicted - actual
        # gradient = np.dot(X.T, error.reshape(-1,1)) / m
        for _ in range(n_iterations):

            predicted = np.dot(X, self.bias_weights.T)


            error = predicted - actual

            gradient = np.dot(error.T, X) / m


            
            self.bias_weights = self.bias_weights - learn_rate * gradient
            
        
            self.log_mean_squared_error.append(self.cost_function(actual, predicted, X) )
        
        return self.log_mean_squared_error


#load dataset of Students scores for Math, Reading, and Writing
dataset=pd.read_csv('student.csv')

#seperate the features
X = np.array(dataset.iloc[:,0:2])
X = np.pad(X, [(0,0),(1,0)], mode='constant', constant_values = 1)


#seperate the values
y = np.array(dataset.iloc[:,2])
y = y.reshape(-1,1)



gd = MSE_Gradient_Descent()

log_mse=gd.gradient_descent(y, X, n_iterations = 100000, learn_rate = 0.0001)


#plot the dataset

x_1 = np.linspace(0, 100, num=100)
x_2 = np.linspace(0,100, num=100)
ax = plt.axes(projection='3d')

ax.scatter3D(X[:,1], X[:,2], y, color = 'b')
ax.plot(x_1, x_2, gd.bias_weights[0][0]+gd.bias_weights[0][1]*x_1+ gd.bias_weights[0][2]*x_2, 'r')
plt.show()

end_time = datetime.now()

print("--- %s seconds ---" % (end_time - start_time))