import sys
sys.path.append('../../')
import os
from src.datapipeline import MLDataset
from src.algorithms.kNNAlgorithm import kNNAlgorithm
from src.algorithms.reductionkNNAlgorithm import reductionkNNAlgorithm
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import argparse

# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-exp", "--Experiment", help = "['PC-KNN':'Parameter Combinations kNN', 'IS-KNN':'Instance Selection kNN']", default='PC-KNN', type=str)
parser.add_argument("-ds", "--DataSet", help = "['cmc', 'pen-based', 'vowel']", default='vowel', type=str)
parser.add_argument("-k", "--K", help = "Number of K (int) nearest neighbors to retreive", default=3, type=int)
parser.add_argument("-dist", "--Distance", help = "['euclidean','manhattan', 'cosine']", default='euclidean', type=str)
parser.add_argument("-votes", "--Votes", help = "['majority', 'sheppard', 'idw']", default='majority', type=str)
parser.add_argument("-weights", "--Weights", help = "['equal','correlation','information_gain']", default='equal', type=str)
parser.add_argument("-u", "--U", help = "Probability treshold for ENNth reduction", default=0.8, type=float)
parser.add_argument("-init", "--Init", help = "Initialization for the RNN reduction", default='random', type=str)
parser.add_argument("-reduction", "--RedTech", help = "['rnn','ennth','drop3']", default='drop3', type=str)
parser.add_argument("-save", "--Save", help = "Directory to save figs", default=False, type=str)

args = parser.parse_args()

if args.Save:
    if not os.path.exists(args.Save):
        os.makedirs(args.Save)
    save_path = Path(args.Save)

data_path = Path('../../data/folded_datasets/raw/' + args.DataSet)
ds = MLDataset(data_path)

if args.Experiment == 'PC-KNN':
########################### Experiment 1 ###################################
    params = {'k':[1,3,5,7],
            'distance': ['euclidean', 'manhattan', 'cosine'],
            'votes': ['majority', 'sheppard', 'idw'],
            'weights': ['equal','correlation','information_gain']}
    keys, values = zip(*params.items())

    params_combs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    results, fold_means = {},{}
    for n, comb in enumerate(params_combs):
        fold_res = []
        for fold, (TrainMatrix, TestMatrix) in enumerate(ds):
            knnalg = kNNAlgorithm()
            knnalg.fit(TrainMatrix, comb['weights'])
            _, time = knnalg.predict_test(TestMatrix.iloc[:,:-1], comb['k'], comb['distance'], comb['votes'], True)
            acc, corr, incorr = knnalg.evaluate(TestMatrix['y_true'])
            results[n*10+(fold)], fold_res  = [fold, *comb.values(), acc,corr,incorr,time], fold_res + [[acc,corr,incorr,time]]
        fold_means[n] = np.array(fold_res).mean(0)
    results, fold_means = pd.DataFrame(results).T, pd.DataFrame(fold_means).T
    results.columns, fold_means.columns = ['fold', 'k', 'distance', 'votes', 'weighting', 'acc', 'corr', 'incorr', 'time'], ['acc', 'corr', 'incorr', 'time']
    if args.Save:
        results.to_csv(save_path.as_posix() + '/fold_res.csv'), fold_means.to_csv(save_path.as_posix() + '/fold_mean_res.csv')
elif args.Experiment == 'IS-KNN':
########################### Experiment 2 ###################################

    results,fold_res = {},[]
    for fold, (TrainMatrix, TestMatrix) in enumerate(ds):
        rknnalg = reductionkNNAlgorithm(TrainMatrix, args.Weights)
        if args.RedTech == 'drop3': RTrainMatrix = rknnalg.DROP3_reduction()#args.K, args.Distance)
        elif args.RedTech == 'ennth': RTrainMatrix = rknnalg.ENNth_reduction()#args.K, args.U, args.Distance)
        elif args.RedTech == 'rnn': RTrainMatrix = rknnalg.RNN_reduction(args.Init)#,args.K, args.Distance, args.Votes)
        knnalg = kNNAlgorithm().fit(RTrainMatrix, args.Weights)
        _, time = knnalg.predict_test(TestMatrix.iloc[:,:-1], args.K, args.Distance, args.Votes, True)
        acc, corr, incorr = knnalg.evaluate(TestMatrix['y_true'])
        results[fold] = [fold, args.K, args.Distance, args.Votes, args.Weights, acc,corr,incorr,time, len(RTrainMatrix)/len(TrainMatrix)]
    
    results = pd.DataFrame(results).T
    results.columns = ['fold', 'k', 'distance', 'votes', 'weighting', 'acc', 'corr', 'incorr', 'time', 'storage']
    if args.Save:
        results.to_csv(save_path.as_posix() + '/{}_folds_res.csv'.format(args.RedTech))