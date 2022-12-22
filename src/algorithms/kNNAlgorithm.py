import pandas as pd
import numpy as np
import numpy as np
import math
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
import time as t

class kNNAlgorithm:
    def __init__(self,):
        self.TrainData = None
        self.TrainLabels = None
        self.WeightingMethod = None
        self.FeaturesWeights = None
        self.Voting = None
        self.TestPredictions = []

    def fit(self, Dataset: pd.DataFrame, weighting = 'equal'):
        if 'y_true' in Dataset.columns:
            self.TrainData = Dataset.iloc[:,:-1]
            self.TrainLabels = Dataset['y_true']
        else:
            raise RuntimeError("Data without labels. Please inlcude the labels or rename the column as 'y_true'")
        if weighting in ['equal', 'information_gain', 'correlation']:
            self.WeightingMethod = weighting
            self.FeaturesWeights = self.compute_weights(self.TrainData, self.TrainLabels, self.WeightingMethod)
        else:
            raise RuntimeError("Invalid weighting method! Posible values: ['equal', 'information_gain', 'correlation']")

    def kNearestNeighbors(self, datapoint, k, metric='euclidean'):
        neighbors = {}
        for idx, x_i in enumerate(self.TrainData.values):
            neighbors[idx] = (x_i, self.compute_distance(datapoint, x_i, self.FeaturesWeights, metric))
        neighbors = pd.DataFrame(neighbors).T
        neighbors.columns = ['x_i', 'distance']
        neighbors = neighbors.sort_values(by=['distance'])
        return neighbors.iloc[:k]

    def predict(self, datapoint, k, metric, policy):
        neighbors = self.kNearestNeighbors(datapoint, k, metric)
        return self.compute_label(neighbors, self.TrainLabels,policy)

    def predict_test(self, TestDataset, k, metric, policy, time = False):
        self.TestPredictions = []
        start = t.time()
        for datapoint in TestDataset.values:
            self.TestPredictions.append(self.predict(datapoint, k, metric, policy))
        end = t.time()
        if time:
            return self.TestPredictions, end - start
        return self.TestPredictions

    def evaluate(self, TestLabels):
        if not self.TestPredictions:
            print('Run the predict_test method before doing an evaluation')
            return None
        correct, incorrect = 0,0
        for idx in range(len(self.TestPredictions)):
            if self.TestPredictions[idx] == TestLabels.values[idx]:
                correct += 1
            else:
                incorrect += 1
        return correct/(correct+incorrect), correct, incorrect

    @staticmethod
    def compute_distance(datapoint, x_i, weights, metric='euclidean'):
        w = weights
        if metric == 'cosine':
            return 1-(np.sum(w*datapoint*x_i)/(np.sqrt(np.sum(w*(datapoint**2))* np.sum(w*(x_i**2)))))
        elif metric in ['euclidean','manhattan']:
            p = 2 if metric == 'euclidean' else 1
            return np.sum(w * abs(datapoint - x_i) ** p) ** (1/p)
        else:
            raise RuntimeError("Please introduce an available metric ['euclidean','cosine','manhattan']")

    @staticmethod
    def compute_label(neighbors, labels, policy, p=1):
        if policy == 'idw':
            policy = lambda index: (1 / (neighbors.loc[index][1]**p)) if not label in classes.keys() else classes[label] + (1 / (neighbors.loc[index][1]**p))
        elif policy == 'sheppard':
            policy = lambda index: math.e**(-neighbors.loc[index][1]) if not label in classes.keys() else classes[label] + math.e**(-neighbors.loc[index][1])
        elif policy == 'majority':
            policy = lambda _: 1 if not label in classes.keys() else classes[label] + 1

        classes = {}
        for i,idx in enumerate(neighbors.index):
            label = labels.iloc[idx]
            classes[label] = policy(idx)
        return max(classes, key=classes.get)

    @staticmethod
    def compute_weights(data, labels, method):
        if method == 'equal':
            return np.ones_like(data.iloc[0])
        elif method == 'correlation':
            return abs(pd.concat([data,labels], axis=1).corr()['y_true'][:-1].values) 
        elif method == 'information_gain':
            return mutual_info_classif(data, np.squeeze(labels))