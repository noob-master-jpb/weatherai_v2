import pandas as pd
import numpy as np
import warnings
class Distance():
    def __init__(self, datapoints, features, target, method, k, test=None, p=None,verbose=False):
        if method not in ["euclidean","manhattan","minkowski","chebyshev"]:
            raise ValueError("Method_Not_Found")
        
        if (method == "minkowski") and (p is None):
            raise ValueError("p must be defined for minkowski distance")

        if k <= 0:
            raise ValueError("k must be greater than 0")

        if test is None:
            if verbose:
                warnings.warn("""Test point not defined during initialization. This may result in errors when calculating distances.""", UserWarning)
        self.method = method
        self.k = k
        self.p = p
        self.train_df = pd.DataFrame(datapoints)
        self.features = features
        self.target = target
        self.test = test
        self.closest = None
        self.calulated  = False


    def euclidean(self, point1, point2,p=2):
        if len(point1) != len(point2):
            raise ValueError("Points must have the same dimensions")

        distance = np.sum(np.absolute(np.array(point1) - np.array(point2)) ** p) ** (1/p)
        return distance

    def manhattan(self,point1, point2):
        if len(point1) != len(point2):
            raise ValueError("Points must have the same dimensions")

        distance = np.sum(np.absolute(np.array(point1) - np.array(point2)))
        return distance
    
    def chebyshev(self,point1, point2):
        if len(point1) != len(point2):
            raise ValueError("Points must have the same dimensions")
        
        distance = np.max(np.absolute(np.array(point1) - np.array(point2)))
        return distance
    
    def calculate_distances(self,test=None,method=None):
        if (test is None) and (self.test is not None):
            test = self.test
        elif(test is not None):
            self.test = test
        else:
            raise ValueError("Test point not defined. Define a test point before calculating distances")


        if method is None:
            method = self.method

        self.calulated = True

        if method == "euclidean":
            self.train_df[method] = self.train_df[self.features].apply(
                lambda row: self.euclidean(row.values, test), 
                axis=1)
            
        elif method == "manhattan":
            self.train_df[method] = self.train_df[self.features].apply(
                lambda row: self.manhattan(row.values, test),
                axis=1)
        
        elif method == "minkowski":
            self.train_df[method] = self.train_df[self.features].apply(
                lambda row: self.euclidean(row.values, test, self.p),
                axis=1)

        elif method == "chebyshev":
            self.train_df[method] = self.train_df[self.features].apply(
                lambda row: self.chebyshev(row.values, test),
                axis=1)
            
        else:
            # self.calulated = False
            raise ValueError("Method_Not_Found")
        
    def get_closest(self):
        if not self.calulated:
            
            raise ValueError("Distances not calculated. Calculate the distance before Ranking")

        self.calculate_distances()
        self.train_df = self.train_df.sort_values(by=self.method)
        self.closest = self.train_df.head(self.k)
        return self.closest