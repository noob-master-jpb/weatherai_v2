import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Dataset():
    def __init__(self,df,features,target,split=0.8,normalize=None,random_state=42):
        self.split = split
        self.normalize = normalize
        self.features = features
        self.target = target
        self.df = df[features + [target]]

        self.train_df = self.df.sample(frac=split, random_state=random_state)
        self.test_df = self.df.drop(self.train_df.index)

        if self.normalize:
            self.normalize(method=self.normalize,features=self.features)


    def normalize(self,method,features=None):

        if features is None:
            features = self.features

        if method not in ("minimax","zscore","z-score","standered"):
            raise ValueError("Invalid normalization method. Use 'minimax' or 'zscore'")

        if self.normalize == None:
            self.normalize=method

        if method == "minimax":
            self.min_vals = self.train_df[features].min()
            self.max_vals = self.train_df[features].max()

            self.train_df[features] = (self.train_df[features] - self.min_vals) / (self.max_vals - self.min_vals)
            self.test_df[features] = (self.test_df[features] - self.min_vals) / (self.max_vals - self.min_vals)
            print(f"Data normalized with {method} Normalization")

        elif method in ("zscore","z-score","standered"):
            self.mean = self.train_df[features].mean()
            self.std = self.train_df[features].std()

            self.train_df[features] = (self.train_df[features] - self.mean) / self.std
            self.test_df[features] = (self.test_df[features] - self.mean) / self.std
            print(f"Data normalized with {method} Normalization")

    def df_selector(self,dataset=None):
        if dataset not in (None,"all","train","test"):
            print("Invalid dataset use train or test or all(default)")
            return
        
        if dataset in (None, "all"):
            df = self.df
        elif dataset == "train":
            df = self.train_df
        elif dataset == "test":
            df = self.test_df

        return df
            

    def skewness(self,features=None,dataset=None):
        if features is None:
            features = self.features

        df = self.df_selector(dataset=dataset)
        if df == None:
            return

        skewness = df[features].skew()
        return skewness
    
    def balance(self,features = None, dataset = None):
        if features is None:
            features = self.features

        df = self.df_selector(dataset=dataset)
        if df == None:
            return


        balance = df[self.target].value_counts()
        return balance
    
    def outliers(self,features=None,dataset=None):
        if self.normalize:
            print(f"Warning data is already normalised with {self.normalize}. outliers might be in accurate")

        if features is None:
            features = self.features

        df = self.df_selector(dataset=dataset)
        if df == None:
            return
        
        

