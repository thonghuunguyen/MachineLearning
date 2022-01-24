import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Simple_Linear_Regression():

    def __init__(self):

        self.weights = None
        self.bias = None

    def mean(self, values):

        return np.mean(values)

    def variance(self, values):

        return np.var(values, ddof = 1)

    def covariance(self, values1, values2):

        return np.cov(values1, values2)[0][1]


    def bias_weights(self, dataset):


        #x = [row[0] for row in dataset]
        #y = [row[1] for row in dataset]

        x = dataset[:,0]
        y = dataset[:,1]

        x_mean, y_mean = self.mean(x), self.mean(y)

        self.weights = self.covariance(x, y) / self.variance(x)
        self.bias = y_mean - self.weights * x_mean


    def prediction(self, X):

        predicted = self.weights * X + self.bias

        return np.array(predicted)

    def rmse(self,predictions,actual):

        rmse = np.sqrt((np.sum((predictions - actual)**2))/float(len(actual)))
        return rmse

## Swedish Insurance Data
#col_names = ["Claims", "Total Payment"]    
#dataset = pd.read_csv('swedinsurance.csv',header=None,skiprows=None,names=col_names)


## Year Experience vs Salary Data
# col_names = ["Years experience", "Salary"]
# dataset = pd.read_csv('salary_data.csv',header=None,skiprows=None,names=col_names)


## SAT vs GPA Data
#col_names = ["SAT", "GPA"]
#dataset = pd.read_csv('satgpa.csv',header=None,skiprows=None,names=col_names)

## Real Estate Prcie vs Size 
col_names = ["Price", "Size"]
dataset = pd.read_csv('realestateprice.csv',header=None,skiprows=None,names=col_names)

dataset = dataset[["Size", "Price"]]

data = np.array(dataset)
data_train, data_test= train_test_split(data, test_size = 0.33, random_state = 5)

slr = Simple_Linear_Regression()

slr.bias_weights(data_train)

print("bias: %.3f, weights: %.3f" % (slr.bias, slr.weights))

y_prediction = slr.prediction(data_test[:,0])

print("RMSE: %.3f" %(slr.rmse(y_prediction, data_test[:,1])))

plt.scatter(data_train[:,0], data_train[:,1], marker = '*')
plt.plot(data_test[:,0], y_prediction, 'r')
