import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

class Dataset():
    def __init__(self, df, features, target, split=0.8, normalize=None, random_state=42):
        self.split = split
        self.normalize = normalize
        self.features = features
        self.target = target
        self.df = df[features + [target]]

        self.train_df = self.df.sample(frac=split, random_state=random_state)
        self.test_df = self.df.drop(self.train_df.index)

        if self.normalize:
            self.normalize(method=self.normalize, features=self.features)

    def normalize_minimax(self, features=None):
        if features is None:
            features = self.features
        minmax_scale = MinMaxScaler()
        self.train_df[features] = minmax_scale.fit_transform(self.train_df[features])
        self.test_df[features] = minmax_scale.transform(self.test_df[features])
        print("Data normalized with minmax Normalization")

    def normalize_zscore(self, features=None):
        if features is None:
            features = self.features

        scaler = StandardScaler()
        self.train_df[features] = scaler.fit_transform(self.train_df[features])
        self.test_df[features] = scaler.transform(self.test_df[features])
        print("Data normalized with z-score Normalization")

    def df_selector(self, dataset=None):
        if dataset not in (None, "all", "train", "test"):
            print("Invalid dataset use train or test or all(default)")
            return
        
        if dataset in (None, "all"):
            df = self.df
        elif dataset == "train":
            df = self.train_df
        elif dataset == "test":
            df = self.test_df

        return df

    def skewness(self, features=None, dataset=None):
        if features is None:
            features = self.features

        df = self.df_selector(dataset=dataset)
        if df is None:
            return

        skewness = df[features].skew()
        return skewness
    
    def balance(self, features=None, dataset=None):
        if features is None:
            features = self.features

        df = self.df_selector(dataset=dataset)
        if df is None:
            return

        balance = df[self.target].value_counts()
        return balance
    
    def outliers(self, feature, dataset=None,threshold=3,condition = ">"):
        if self.normalize:
            warnings.warn(f"Warning data is already normalized with {self.normalize}. Outliers might be inaccurate")

        df = self.df_selector(dataset=dataset)
        if df is None:
            return

        if feature not in df.columns:
            warnings.warn(f"Feature '{feature}' not found in the dataset.", UserWarning)
            return


        values = df[feature].values.reshape(-1, 1)
        z_scores = np.abs(StandardScaler().fit_transform(values)).flatten()
        mask = z_scores > threshold
        

        outliers_df = df[mask]
        return outliers_df
