from sys import maxsize
import random
import numpy as np
import pandas as pd
from src.algorithms.kNNAlgorithm import kNNAlgorithm


class reductionkNNAlgorithm:
    def __init__(self,Data, Weights = 'equal'):
        self.Data = Data
        self.Weights = Weights

    def CNN_reduction(self, initialization = 'sequence', k=1, metric= 'euclidean', policy='majority'):
        if initialization == 'sequence':
            ReducedData = pd.DataFrame(self.Data.iloc[0]).T
        elif initialization == 'random':
            ReducedData = self.Data.iloc[[random.choice([i for i, e in enumerate(self.Data['y_true']) if e == n]) for n in set(self.Data['y_true'])]]
        knn = kNNAlgorithm().fit(ReducedData, self.Weights)
        completed = False
        cont = 0
        while completed == False:
          for idx in self.Data.index:
              cont += 1
              PredLab = knn.predict(self.Data.iloc[idx,:-1], k, metric, policy)
              if PredLab != self.Data.iloc[idx,-1]:
                  ReducedData = ReducedData.append(self.Data.iloc[idx])
                  knn.fit(ReducedData, self.Weights)
                  break
              if idx == len(self.Data.index)-1:
                  completed = True
        print(cont)
        return ReducedData

    def RNN_reduction(self, initialization = 'sequence', k=1, metric= 'euclidean', policy='majority'):
        CNN_RD = self.CNN_reduction(initialization)
        count_rem = 0
        for idx in CNN_RD.index:
            O_CNN_RD = CNN_RD.copy() 
            CNN_RD = CNN_RD.drop(idx-count_rem)
            knn = kNNAlgorithm().fit(CNN_RD, self.Weights)
            removed = False
            for idx2 in self.Data.index:
                PredLab = knn.predict(self.Data.iloc[idx2,:-1], k, metric, policy)
                if PredLab != self.Data.iloc[idx2,-1]:
                    CNN_RD = O_CNN_RD.copy()
                    break
                if idx == len(self.Data.index)-1:
                  removed = True
            if removed:
                count_rem += 1
        return CNN_RD

    def ENNth_reduction(self, k=2, u=0.8, metric='euclidean'):
        T_ENNth_RD = self.Data.copy()
        ENNth_RD = self.Data.copy()
        for idx in self.Data.index:
            T_ENNth_RD = T_ENNth_RD.drop(idx)
            knn = kNNAlgorithm().fit(T_ENNth_RD, self.Weights).kNearestNeighbors(self.Data.loc[idx][:-1],k,metric)
            probs = {}
            for nnidx in knn.index:
                cls = int(T_ENNth_RD.iloc[nnidx][-1])
                probs[cls] = 1.0 / (1.0 + knn.loc[nnidx]['distance']) if not cls in probs.keys() else probs[cls] + 1.0 / (1.0 + knn.loc[nnidx]['distance'])
            norm_probs = {k:v/sum(probs.values()) for k,v in probs.items()}
            fin_class = max(norm_probs, key=norm_probs.get)
            if fin_class != self.Data.loc[idx][-1] or norm_probs[fin_class] <= u:
                ENNth_RD = ENNth_RD.drop(idx)
            T_ENNth_RD = self.Data.copy()
        return ENNth_RD

    def DROP3_reduction(self, k=3, metric='euclidean'):

        # Variables initialization
        columns_names = self.Data.iloc[:,:-1].keys()
        init_data, init_y = self.Data.iloc[:,:-1].values, self.Data.iloc[:,-1].values[:,np.newaxis]
        init_data, samples_index = np.unique(ar=init_data,return_index=True, axis=0)
        init_y = init_y[samples_index]
        points_metadata = {tuple(x): [[], [], y] for x, y in zip(init_data,init_y)}
        init_dists = []

        # Neighbors & Associates search
        self.compute_neighbors_and_associates(self, init_dists, init_data, init_y, k, metric, points_metadata)

        removed = 0
        
        for idx in range(len(init_dists)):
            datapoint = tuple(init_dists[idx - removed][0])
           
            w, wt = self.compute_with_without(datapoint, points_metadata)
            
            if wt >= w:
                init_dists = init_dists[:idx - removed] + init_dists[idx - removed + 1:]

                for associate in points_metadata[(datapoint)][1]:
                    a_neighs, remaining_samples = self.del_from_neighs(datapoint, associate, init_dists, points_metadata)
                    knn = kNNAlgorithm().fit(pd.DataFrame(remaining_samples), self.Weights, y=False)
                    neighbors = knn.kNearestNeighbors(associate, k + 2, metric)
                    possible_neighs = [init_dists[x][0] for x in neighbors.index]

                    self.compute_new_neighbors(associate, a_neighs, possible_neighs, points_metadata)

                    new_neighbor = a_neighs[-1]
                    points_metadata[tuple(new_neighbor)][1].append(associate)
                removed += 1
                
        Drop3_RD = pd.DataFrame([x for x, _, _ in init_dists],
                               columns=columns_names)
        Drop3_RD['y_true'] = pd.DataFrame([x for _, x, _ in init_dists])

        return Drop3_RD

    @staticmethod
    def compute_neighbors_and_associates(self, init_dists, init_data, init_y,
                         k, metric, points_metadata):
        for datapoint, label in zip(init_data, init_y):
            # Compute initial distances
            min_distance = maxsize
            for datapoint2, label2 in zip(init_data, init_y):
                if label != label2:
                    xy_distance = np.linalg.norm(datapoint - datapoint2)
                    if xy_distance < min_distance:
                        min_distance = xy_distance
            init_dists.append([datapoint, label, min_distance])

            knn = kNNAlgorithm().fit(pd.DataFrame(init_data), self.Weights, y=False)
            neighbors = knn.kNearestNeighbors(datapoint, k + 2, metric)
            # Add neighbors of datapoint
            neighs = [init_data[x] for x in neighbors.index[1:]]
            points_metadata[tuple(datapoint)][0] = neighs
            # Add datapoint as an associate of is neighbors
            for neigh in neighs[:-1]: points_metadata[tuple(neigh)][1].append(datapoint)
            init_dists.sort(key=lambda x: x[2], reverse=True)

    @staticmethod
    def compute_with_without(datapoint, points_metadata):
        w,wt = 0,0

        datapoint_associates = points_metadata[datapoint][1]
        associates_labels = [points_metadata[tuple(x)][2] for x in datapoint_associates]
        associates_neighs = [points_metadata[tuple(x)][0] for x in datapoint_associates]

        for a_label, a_neighs in zip(associates_labels, associates_neighs):
            neighs_labels = [int(points_metadata[tuple(x)][2][0]) for x in a_neighs]
            # number of associates of P classified correctly with P as a neighbor.
            count = np.bincount(neighs_labels[:-1])
            max_class = np.where(count == np.amax(count))[0][0]
            if max_class == a_label: w += 1
            # number of associates of P classified correctly without P.
            idx_a = [e.tolist() for e in a_neighs].index(list(datapoint))
            count = np.bincount(neighs_labels[:idx_a] + neighs_labels[idx_a + 1:])
            max_class = np.where(count == np.amax(count))[0][0]
            if max_class == a_label: wt += 1

        return w, wt

    @staticmethod
    def del_from_neighs(datapoint, associate, init_dists, points_metadata):
        a_neighs = points_metadata[tuple(associate)][0]
        idx_a = [e.tolist() for e in a_neighs].index(list(datapoint))
        a_neighs = a_neighs[:idx_a] + a_neighs[idx_a + 1:]
        remaining_samples = [x for x, _, _ in init_dists]
        return a_neighs, remaining_samples

    @staticmethod
    def compute_new_neighbors(a_associate_of_x, a_neighs, possible_neighs, points_metadata):
        for pos_neigh in possible_neighs[1:]:
            if not list(pos_neigh) in [e.tolist() for e in a_neighs]:
                a_neighs.append(pos_neigh)
                break
        points_metadata[tuple(a_associate_of_x)][0] = a_neighs