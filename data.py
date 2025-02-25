import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
import pandas as pd

df = pd.read_csv('data.csv')
class Dataset():
    def __init__(self, df, features, target, final_target=None, binary_out = True, split=0.8, normalize=None,
                drop_outliers=False, drop_duplicate=False, random_state=42):

        df = df[features + [target]]
        if drop_duplicate:
            df.drop_duplicates(inplace=True)

        if final_target:
            df = df.rename(columns={target: final_target})
            target = final_target
        
        if binary_out:
            df[target] = (df[target] != 0).astype(int)

        self.split = split
        self.features = features
        self.normalize = normalize
        self.target = target
        self.df = df

        if drop_outliers:
            self.drop_outliers(features=self.features,verbose=False)

        self.train_df = self.df.sample(frac=split, random_state=random_state)
        self.test_df = self.df.drop(self.train_df.index)
        if self.normalize:
            if self.normalize == "minmax":
                self.normalize_minimax()
            elif self.normalize == "zscore":
                self.normalize_zscore()
            warnings.warn(f"Unknown normalization method: {self.normalize}", UserWarning)

    def normalize_minimax(self, features=None):
        if features is None:
            features = self.features
        minmax_scale = MinMaxScaler()
        self.df[features] = minmax_scale.fit_transform(self.df[features])
        self.train_df[features] = minmax_scale.transform(self.train_df[features])
        self.test_df[features] = minmax_scale.transform(self.test_df[features])
        print("Data normalized with minmax Normalization")

    def normalize_zscore(self, features=None):
        if features is None:
            features = self.features

        scaler = StandardScaler()
        self.df[features] = scaler.fit_transform(self.df[features])
        self.train_df[features] = scaler.transform(self.train_df[features])
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
    
    def outliers(self, feature, dataset=None,threshold=3,verbose = True):
        if verbose:
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

    def drop_outliers(self, features=None, dataset=None,verbose = False):
        if features is None:
            features = self.features

        df = self.df_selector(dataset=dataset)
        if df is None:
            return

        temp = {}

        for i in tuple(features):
            out = self.outliers(i,verbose=verbose)
            temp[i] = out
            df.drop(out.index,inplace = True)
            if verbose:
                print(f"dropped indices {out.index} from {i}")

        return temp