import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from datetime import datetime
start_time = datetime.now()


class Logistics_Gradient_Descent():
    

    def __init__(self):
        
        self.bias_weights = None
        self.log_bias_weights = None
        self.log_error = None
        
    def prediction(self, X):
        linear_prediction = np.dot(X, self.bias_weights.T)
        predict = self.sigmoid_activation(linear_prediction)
        predict = np.round(predict)
        return predict
        
    def sigmoid_activation(self, predicted):

        sigmoid_function = np.array([])
        for i in range(len(predicted)):
            
            sigmoid_function = np.append(sigmoid_function, 1.0 / (1.0 + np.exp(-predicted[i])))
        
        
        #return 1.0 if sigmoid_function >= .5 else 0
        return sigmoid_function.reshape(-1,1)
    
    #Cost/Loss function: Logistic
    def cost_function(self, actual, predicted, X):
        
        m = len(actual)

        cost = - np.sum(actual * np.log(predicted) + (1-actual) * np.log(1-predicted)) / m 
        # for i in range(len(actual)):
        #     cost +=  actual[i] * np.log(self.sigmoid_activation(predicted[i])) + (1-actual[i]) * np.log(1-self.sigmoid_activation(predicted[i]))
        
        # return - cost / m
    
    def gradient_descent(self, actual, X, n_iterations = 1000 ,learn_rate = 0.1 ):
      
        
        self.bias_weights = np.array(np.zeros(X.shape[1]))

        self.bias_weights = self.bias_weights.reshape(1,X.shape[1])
        
        self.log_bias_weights = []
        self.log_error = []
        
        m = len(actual)
        for _ in range(n_iterations):
            
            linear_prediction = np.dot(X, self.bias_weights.T)
            sigmoid_prediction = self.sigmoid_activation(linear_prediction)
            error = sigmoid_prediction - actual

            gradient = np.dot(error.T, X) / m
            
            self.bias_weights = self.bias_weights - learn_rate * gradient
            
            self.log_bias_weights.append(self.bias_weights)
            self.log_error.append(error)

        return self.log_bias_weights, self.log_error

    
def normalize(X):
    X_normed = (X - X.min()) / (X.max() - X.min())
    
    return X_normed

#load dataset
dataset2 = pd.read_csv('pima-indians-diabetes.csv',header=None)
dataset = normalize(dataset2)

#seperate the features
X = np.array(dataset.iloc[:,0:dataset.shape[1]-1])

X = np.pad(X, [(0,0),(1,0)], mode='constant', constant_values = 1)


#seperate the values
y = np.array(dataset.iloc[:,-1])
y = y.reshape(-1,1)

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size = 0.1)


#initialize the Logistic Regression Class

LR = Logistics_Gradient_Descent()

log_bw, log_error = LR.gradient_descent(y_train, X_train, n_iterations=100, learn_rate = 0.1)


prediction = LR.prediction(X_test)

correct = 0
for i in range(len(y_test)):
    if prediction[i] == y_test[i]:
        correct += 1
print(correct / float(len(y_test)) * 100.0)



end_time = datetime.now()

print("--- %s seconds ---" % (end_time - start_time))