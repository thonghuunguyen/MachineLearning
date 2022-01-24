import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

from datetime import datetime
start_time = datetime.now()

class Perceptron():
    
    def __init(self):
        
        self.bias_weights = None
        self.log_bias_weights = None
        
    def prediction(self, X):
        #print("bias_weights shape: ",self.bias_weights.shape)
        #print("X shape: ", X.shape)
        predict = np.dot(self.bias_weights, X.T)
        #print("predict shape: ", predict.shape)
        #predict = np.where(predict >= 0, 1, 0)
        #print(predict)
        return np.where(predict >= 0, 1, 0)

    
    
    def gradient_descent(self, actual, X, n_iterations = 1000, learn_rate = 0.1):
        
        self.bias_weights = np.array(np.zeros(X.shape[1]))
        #print(self.bias_weights)
        #self.bias_weights = self.bias_weights.reshape(1,X.shape[1])
        
        self.log_bias_weights = []
        
        
        for _ in range(n_iterations):
            
            linear_prediction = self.prediction(X)
            linear_prediction = linear_prediction.reshape(-1,1)

            error = (actual - linear_prediction)
            #print("linear shape: ",linear_prediction.shape)
            #print("actual y shape: ",actual.shape)
            #print("error shape: ",error.shape)
            gradient = np.dot(error.T, X)
            #print("gradient shape: ",gradient.shape)
            self.bias_weights = self.bias_weights + learn_rate * gradient
            #print("bias_weights shape: ",self.bias_weights.shape)
            self.log_bias_weights.append(self.bias_weights)
            
        return self.log_bias_weights
    
    
X, y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=7)
y = y.reshape(-1,1)

X = np.pad(X, [(0,0),(1,0)], mode='constant', constant_values = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


def accuracy(y_predict, y_actual):
    accuracy = np.sum(y_predict == y_actual) / len(y_actual)
    return accuracy


p = Perceptron()
log_bw = p.gradient_descent(y_train, X_train, n_iterations = 1000, learn_rate = .01)

predict = p.prediction(X_test)

print("accuracy: ", accuracy(predict.T,y_test))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X_train[:, 1], X_train[:, 2], marker="o", c=y_train)

x_1 = np.linspace(np.min(X_train[:, 1]),np.max(X_train[:, 1]),num=100)


ax.plot(x_1, -(p.bias_weights[0][1] * x_1 + p.bias_weights[0][0])/ p.bias_weights[0][2], "k")


plt.show()


end_time = datetime.now()
print("--- %s seconds ---" % (end_time - start_time))