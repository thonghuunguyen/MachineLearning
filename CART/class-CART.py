import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

from datetime import datetime
start_time = datetime.now()


class Node():
    
    def __init__(self, feature_index = None, threshold_value = None, left = None, right = None, info_gain = None, value = None):
        
        self.feature_index = feature_index
        self.threshold_value = threshold_value
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value


class DecisionTree():
    
    
    def __init__(self, max_depth = 2, min_samples = 5):
        
        self.root = None
        self.max_depth = max_depth
        self.min_samples = min_samples
        
    
    def build_tree(self, dataset, curr_depth = 0):
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        
        num_samples, num_features = np.shape(X)
        
        if num_samples >= self.min_samples and curr_depth <= self.max_depth:
            
            best_split = self.best_fit(dataset, num_samples, num_features)
            
            if best_split["info_gain"] > 0:
                
                left_subtree = self.build_tree(best_split['dataset_left'], curr_depth + 1)
                right_subtree = self.build_tree(best_split['dataset_right'], curr_depth + 1)
                
                return Node(best_split['feature_index'], best_split["threshold_value"], left_subtree, right_subtree, best_split["info_gain"])
            
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value = leaf_value)
    
    def best_fit(self, dataset, num_samples, num_features):
       
        best_split = {}
        
        max_info_gain = -float("inf")
        
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            unique_values = np.unique(feature_values)
            
            for value in unique_values:
                dataset_left, dataset_right = self.split(dataset, feature_index, value)
                
                
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    
                    y, y_left, y_right = dataset[:,-1], dataset_left[:,-1], dataset_right[:,-1]
                    
                    curr_info_gain = self.information_gain(y, y_left, y_right)
                    
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold_value"]  = value
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
        return best_split
    
    def split(self, dataset, feature_index, value):
        
        dataset_left = np.array([row for row in dataset if row[feature_index] <= value])
        dataset_right = np.array([row for row in dataset if row[feature_index] > value])
        return dataset_left, dataset_right
    
    def information_gain(self, parent_value, left_child_value, right_child_value):
        
        weight_left = len(left_child_value) / len(parent_value)
        weight_right = len(right_child_value) / len(parent_value)
        
        gain = self.gini_index(parent_value) - (weight_left * self.gini_index(left_child_value) + weight_right * self.gini_index(right_child_value))
        
        return gain
    
    def gini_index(self, y_values):
        class_values = np.unique(y_values)
        gini = 0
        for i in class_values:
            p = len(y_values[y_values == i]) / len(y_values)
            gini += p**2
            
        return 1 - gini
    
    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
    
    def fit(self, X,Y):
        dataset = np.concatenate((X,Y), axis = 1)
        self.root = self.build_tree(dataset)
        
    def print_tree(self, tree=None, indent = " "):
        if not tree:
            tree = self.root
            
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_"+str(tree.feature_index), "<=", tree.threshold_value, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
            
    def predict(self, X):
        
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions
    
    def make_prediction(self, x, tree):
        
        if tree.value != None: return tree.value
        
        feature_value = x[tree.feature_index]
        if feature_value <= tree.threshold_value:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
            
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
Y = dataCopy.iloc[:, -1].values.reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .5)


classifier = DecisionTree(max_depth = 7, min_samples = 3)
classifier.fit(X_train, Y_train)
classifier.print_tree()    

y_predict = classifier.predict(X_test)

count = 0
for i in range(len(Y_test)):
    if y_predict[i] == Y_test[i]:
        count += 1

print("Accuracy: ", count/len(Y_test))
        
end_time = datetime.now()
print("--- %s seconds ---" % (end_time - start_time))