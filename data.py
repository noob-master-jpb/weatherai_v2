import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Dataset():
    def __init__(self,df,features,target,split=0.8,normalize=False):
        self.split = split
        self.normalize = normalize
        self.features = features
        self.target = target
        self.df = df[features + [target]]

        self.train_df = self.df.sample(frac=split, random_state=42)
        self.test_df = self.df.drop(self.train_df.index)

        if normalize:
            if normalize == "minimax":
                self.min_vals = self.train_df[features].min()
                self.max_vals = self.train_df[features].max()

                self.train_df[features] = (self.train_df[features] - self.min_vals) / (self.max_vals - self.min_vals)
                self.test_df[features] = (self.test_df[features] - self.min_vals) / (self.max_vals - self.min_vals)

            elif (normalize == "standard") or (normalize == "zscore") or (normalize == "z-score"):
                self.mean = self.train_df[features].mean()
                self.std = self.train_df[features].std()

                self.train_df[features] = (self.train_df[features] - self.mean) / self.std
                self.test_df[features] = (self.test_df[features] - self.mean) / self.std
            else:
                raise ValueError("Invalid normalization method. Use 'minimax' or 'zscore'")


    def skewness(self,features=None,dataset=None):
        if features is None:
            features = self.features

        if (dataset is None) or (dataset == "all"):
            df = self.df
        elif dataset == "train":
            df = self.train_df
        elif dataset == "test":
            df = self.test_df
        else:
            print("Invalid dataset use train or test or all(default)")
            return

        skewness = df[features].skew()
        return skewness
    
    def balance(self,feature = None, dataset = None):
        if features is None:
            features = self.features

        if (dataset is None) or (dataset == "all"):
            df = self.df
        elif dataset == "train":
            df = self.train_df
        elif dataset == "test":
            df = self.test_df
        else:
            print("Invalid dataset use train or test or all(default)")
            return
        
        balance = df[self.target].value_counts()
        return balance