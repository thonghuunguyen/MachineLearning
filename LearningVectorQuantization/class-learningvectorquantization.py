# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 19:09:55 2021

@author: tnguyen
"""

import numpy as np
import pandas as pd
from random import randrange

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

from datetime import datetime
start_time = datetime.now()



class Learning_Vector_Quantization():
    
    def __init__(self):
        
        self.predictions = None
        self.codebooks = None
        
    def euclidean_dictance(self, v1, v2):
        distance = 0.0
        for i in range(len(v1)-1):
            distance += np.sum(np.square(v1[i] - v2[i]))
        return np.sqrt(distance)
    
    
    
    def closest_distance(self, codebooks, row):
        distance = []
        
        for codebook in codebooks:
            dist = self.euclidean_dictance(codebook, row)
            distance.append((codebook, dist))
        distance.sort(key = lambda x : x[1])
        return distance[0][0]
    
    
    def random_codebook(self, train_dataset):
        n_records = len(train_dataset)
        n_features = len(train_dataset[0])
        codebook = [train_dataset[randrange(n_records)][i] for i in range(n_features)]
        return codebook


    def train_codebooks(self, X_train, Y_train, n_codebooks, lrate, epochs):
        train_dataset = np.concatenate((X_train, Y_train), axis = 1)
        codebooks = [self.random_codebook(train_dataset) for i in range(n_codebooks)]
        #train_dataset = np.concatenate((X_train, Y_train), axis = 1)
        
        #Learning Vector Quantization Algorithm
        for epoch in range(epochs):
            rate = lrate * (1.0 - (epoch/float(epochs)))
            
            sum_error = 0.0
            for row in train_dataset:
                w = self.closest_distance(codebooks, row)
                
                for i in range(len(row)-1):
                    error = row[i] - w[i]
                    sum_error += np.square(error)
                    
                    if w[-1] == row[-1]:
                        w[i] += rate * error
                    else:
                        w[i] -= rate * error
        return codebooks
        
    
    def predict(self, X_train, Y_train, X_test, Y_test, n_codebooks, lrate, epochs):
        self.codebooks = self.train_codebooks(X_train, Y_train, n_codebooks, lrate, epochs)
        
        self.predictions = []
        test_dataset = np.concatenate((X_test, Y_test), axis = 1)
        
        for row in test_dataset:
            output = self.closest_distance(self.codebooks, row)[-1]
            self.predictions.append(output)
        return self.predictions
        


#label the column of the data
col_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', 'result']

#load data
data = pd.read_csv('ionosphere.data', skiprows=0, header = None, names = col_names)

#make a copy of the data
dataCopy = data.copy()

X = dataCopy.iloc[:, :-1].values
#Y = dataCopy.iloc[:, -1:].values
Y = dataCopy.iloc[:, -1].values.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .3)

LVQ = Learning_Vector_Quantization()

lvq_predict = LVQ.predict(X_train, Y_train, X_test, Y_test, n_codebooks = 20, lrate =  0.3, epochs = 50)

score = 0
n = len(Y_test)
for i in range(n):
    if lvq_predict[i] == Y_test[i].tolist()[0]:
        score += 1
        
print("Accuracy: ", score/n)
end_time = datetime.now()


print("--- %s seconds ---" % (end_time - start_time))