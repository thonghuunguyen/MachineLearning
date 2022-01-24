import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

from datetime import datetime
start_time = datetime.now()


class K_Nearest_Neighbors():
    
    def __init__(self):
        
        self.classification_prediction = None
        self.rmse_prediction = None
        self.neighbors = None
        
    def euclidean_distance(self, v1, v2):
        distance = 0.0
        for i in range(len(v1)):
            distance += np.sum(np.square(v1[i] - v2[i]))
        return np.sqrt(distance)
    
    def get_neighbors(self, train_dataset, test_row, num_neighbors):
        distance = []
        
        for train_row in train_dataset:
            dist = self.euclidean_distance(test_row, train_row)
            distance.append((train_row, dist))
        distance.sort(key = lambda x: x[1])
        
        self.neighbors = []
        for  i in range(num_neighbors):
            self.neighbors.append(distance[i][0])
        return self.neighbors
    
    def predict_classification(self, X_train, Y_train, X_test, num_neighbors):
        
        train_dataset = np.concatenate((X_train, Y_train), axis = 1)
        self.classification_prediction = []
        
        for row in X_test:
            
            neighbors = self.get_neighbors(train_dataset, row, num_neighbors)
            output_values = [row[-1] for row in neighbors]
            self.classification_prediction.append(max(set(output_values), key=output_values.count))
        return self.classification_prediction
            
    def predict_regression(self, X_train, Y_train, X_test, num_neighbors):
        
        train_dataset = np.concatenate((X_train, Y_train), axis = 1)
        self.rmse_prediction = []
        for row in X_test:
            neighbors = self.get_neighbors(train_dataset, row, num_neighbors)
            output_values = [row[-1] for row in neighbors]
            self.rmse_prediction.append(sum(output_values) / float(len(output_values)))
        return self.rmse_prediction
        


#label the column of the data
col_names = ['sex', 'length', 'diameter', 'height', 'whole weight', 'shucked weight', 'viscera weight', 'shell weight', 'rings']

#load data
data = pd.read_csv('abalone.data', skiprows=0, header = None, names = col_names)
mapping = {'M': 0, 'F': 1, 'I': 2}
data = data.replace({'sex': mapping})
#make a copy of the data
dataCopy = data.copy()

X = dataCopy.iloc[:, :-1].values
#Y = dataCopy.iloc[:, -1:].values
Y = dataCopy.iloc[:, -1].values.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .5)

KNN = K_Nearest_Neighbors()

#KNN.predict_classification(X_train, Y_train, X_test, num_neighbors = 5)
output = KNN.predict_regression(X_train, Y_train, X_test, num_neighbors = 5)

end_time = datetime.now()
print("--- %s seconds ---" % (end_time - start_time))