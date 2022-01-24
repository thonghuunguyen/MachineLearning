import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

from datetime import datetime
start_time = datetime.now()


class Naive_Bayes():
    
    def __init__(self):
        
        self.probabilities = None
        self.summaries = None
        self.prediction = None
        
        
    def separate_feature_by_class(self,X,Y):
        
        separated = {}
        
        for i in range(len(Y)):
            class_value = Y[i]
            if (class_value not in separated):
                separated[class_value] = []
                
            separated[class_value].append(X[i])
        return separated
    
    def mean(self, numbers):
        return np.mean(numbers)
    
    def stdev(self, numbers):
        return np.std(numbers, ddof = 1)
    
    
    def summarize_feature(self, X):
        summaries = [(self.mean(column), self.stdev(column), len(column)) for column in zip(*X)]
    
        return summaries


    def summarize_class(self, X, Y):
        separated = self.separate_feature_by_class(X,Y)
        
        self.summaries = {}
        
        for class_value, rows in separated.items():
            self.summaries[class_value] = self.summarize_feature(rows)
        return self.summaries

    def calculate_probability(self, x, mean, stdev):
        return np.exp(-((x-mean)**2/(2*stdev**2)))/(np.sqrt(2*np.pi)*stdev)
    
    def calculate_class_probabilities(self,summaries, X_test):
        
        total_rows = sum([summaries[class_value][0][2] for class_value in summaries])
        probabilities = {}
        
        for class_value, class_summaries in summaries.items():
            
            probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
            
            for i in range(len(class_summaries)):
                mean, stdev, count = class_summaries[i]
                
                probabilities[class_value] *=  self.calculate_probability(X_test[i], mean, stdev)
        return probabilities


    def predict(self, X_test):
        self.prediction = []
        self.probabilities = {}
        for row in X_test:
            self.probabilities = self.calculate_class_probabilities(self.summaries, row)
            best_label, best_probability = None, -1
            
            for class_value, probability in self.probabilities.items():
                
                if best_label is None or probability > best_probability:
                    best_probability = probability
                    best_label = class_value
                    
            self.prediction.append(best_label)
        return self.prediction

#Label the column of the data
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'type']


#load the data
data = pd.read_csv("iris.data",skiprows=0, header = None, names=col_names)

#make a copy of the data
dataCopy = data.copy()
        
#replace the string values to be numbers Iris-setosa = 0 etc...        
#dataCopy['type'] = dataCopy['type'].replace(['Iris-setosa'], 0)
#dataCopy['type'] = dataCopy['type'].replace(['Iris-versicolor'], 1)
#dataCopy['type'] = dataCopy['type'].replace(['Iris-virginica'], 2)

X = dataCopy.iloc[:, :-1].values
#Y = dataCopy.iloc[:, -1:].values
Y = dataCopy.iloc[:, -1].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .3)


NB = Naive_Bayes()
NB.summarize_class(X_train, Y_train)

y_predict = NB.predict(X_test)

count = 0

for i in range(len(Y_test)):
    if y_predict[i] == Y_test[i]:
        count += 1
        
print("Accuracy: ", count/float(len(Y_test)) * 100.0)




end_time = datetime.now()
print("--- %s seconds ---" % (end_time - start_time))