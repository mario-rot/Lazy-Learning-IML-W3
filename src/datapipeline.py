from typing import Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from scipy.io import arff
import os

class MLDataset:
    def __init__(self,
                 data_path: Union[Path, str],
                 verbose=False,
                 save_dir=False):
        self.data_path = data_path
        self.verbose = verbose
        self.save_dir = save_dir
        self.raw_data = None
        self.meta = None
        self.df = None
        self.y_true = None
        self.processed_data = None
        self.classes_relation = None
        self.TrainMatrix = None
        self.TestMatrix = None
        self.train = None
        self.train_raw = None
        self.test = None
        self.test_raw = None

        self.preprocess_folds()

        # self.preprocessing()

    def __len__(self):
        return len(self.TrainMatrix)

    def __getitem__(self, idx):
        return self.TrainMatrix[idx], self.TestMatrix[idx]

    def preprocess_folds(self):
        self.TrainMatrix = []
        self.TestMatrix = []
        # os.chdir(self.data_path)
        for file in os.listdir(self.data_path):
            file_path = f"{self.data_path}\{file}"
            self.processed_data, self.raw_data = self.preprocessing(Path(file_path))
            if file.endswith("train.arff"):
                self.TrainMatrix.append(self.processed_data)
                self.train = self.processed_data
                self.train_raw = self.raw_data
                if self.save_dir:
                    self.save('processed_' + file, self.processed_data, self.save_dir)
            elif file.endswith("test.arff"):
                if self.processed_data.iloc[:,:-1].shape[1] < 24 and 'cmc' in file: 
                    self.processed_data.insert(17, 'hoccupation_4', np.zeros([len(self.processed_data)]))
                self.TestMatrix.append(self.processed_data)
                self.test = self.processed_data
                self.test_raw = self.raw_data
                if self.save_dir:
                    self.save('processed_' + file, self.processed_data, self.save_dir)

        return self.TrainMatrix, self.TestMatrix

    def save(self, filename, matrix, dir=''):
        matrix.to_csv(dir + filename + '.csv')

    def statistics(self, data_type):
        if data_type == 'raw':
            data = pd.concat([self.train_raw.iloc[:, :-1], self.test_raw.iloc[:, :-1]])
            labels = pd.concat([self.train_raw.iloc[:, -1], self.test_raw.iloc[:, -1]])
        else:
            data = pd.concat([self.train.iloc[:, :-1], self.test.iloc[:, :-1]])
            labels = pd.concat([self.train.iloc[:, -1], self.test.iloc[:, -1]])
        gen_stats = {'n_classes': len(set(labels)),
                     'n_features': len(data.columns),
                     'n_instances': len(data)}
        stats = {'Nulls': data.isnull().sum(0).values,
                 'Mins': data.min().values,
                 'Max': data.max().values,
                 'Means': data.mean().values,
                 'StD': data.std().values,
                 'Variance': data.var().values}
        stats = pd.DataFrame.from_dict(stats, orient='index',
                                       columns=data.columns)
        return gen_stats, stats

    def preprocessing(self, fold_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Preprocessing
        if self.verbose:
            print(f'---Preprocessing {fold_path.name} dataset fold---')
        df, meta = self.import_raw_dataset(fold_path)
        y_true = self.get_predicted_value(df)
        classes_relation =  {k:v for v,k in enumerate(sorted(set(y_true)))}
        df = self.remove_predicted_value(df)
        nulls = self.check_null_values(df)
        if self.verbose:
            if nulls.sum() != 0:
                print(f'There is nulls values: {nulls}')
            else:
                print(f'Nan values: 0')
        processed_data = self.standardization(df)
        processed_data['y_true'] = self.encode_labels(y_true, classes_relation)
        return processed_data, df

    def import_raw_dataset(self, fold_path):
        data, meta = arff.loadarff(fold_path)
        data = pd.DataFrame(data)
        raw_data = self.string_decode(data)
        return raw_data, meta

    @staticmethod
    def string_decode(df: pd.DataFrame):
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.decode('utf-8')
        return df

    @staticmethod
    def remove_predicted_value(df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[:, :-1]

    @staticmethod
    def get_predicted_value(df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[:, -1]

    @staticmethod
    def check_null_values(df: pd.DataFrame) -> pd.Series:
        return df.isnull().sum()

    @staticmethod
    def encode_labels(labels, classes_relation):
        num_classes = [classes_relation[item] for item in labels]
        return num_classes

    @staticmethod
    def standardization(df: pd.DataFrame, columns=None) -> pd.DataFrame:
        # numerical features
        num_features = df.select_dtypes(include=np.number).columns
        num_transformer = Pipeline(steps=[
            ('replace_nan', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())])
        # categorical features
        cat_features = df.select_dtypes(exclude=np.number).columns
        cat_transformer = Pipeline(steps=[
            ('replace_nan', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder())])

        # transform columns
        ct = ColumnTransformer(
            transformers=[
                ('num', num_transformer, num_features),
                ('cat', cat_transformer, cat_features)])
        X_trans = ct.fit_transform(df)

        # dataset cases
        # case 1: categorical and numerical features
        if len(cat_features) != 0 and len(num_features) != 0:
            columns_encoder = ct.transformers_[1][1]['encoder']. \
                get_feature_names_out(cat_features)
            columns = num_features.union(pd.Index(columns_encoder), sort=False)

        # case 2: only categorical features
        elif len(cat_features) != 0 and len(num_features) == 0:
            columns = ct.transformers_[1][1]['encoder']. \
                get_feature_names_out(cat_features)
            columns = pd.Index(columns)
            X_trans = X_trans.toarray()

        # case 3: only numerical features
        elif len(cat_features) == 0 and len(num_features) != 0:
            columns = num_features

        # catch an error
        else:
            print('There is a problem with features')

        # processed dataset
        processed_df = pd.DataFrame(X_trans, columns=columns)
        return processed_df


if __name__ == '__main__':
    data_path = Path('../data/raw/iris.csv')
    dataset = MLDataset(data_path)
