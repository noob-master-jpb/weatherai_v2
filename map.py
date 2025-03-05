import pandas as pd
import numpy as np

class Distance():
    def __init__(self, datapoints, features, target, method, k, test, p=None):
        if method not in ["euclidean","manhattan","minkowski","chebyshev","cosine"]:
            raise ValueError("Method_Not_Found")
        
        if (method == "minkowski") and (p is None):
            raise ValueError("p must be defined for minkowski distance")

        if k <= 0:
            raise ValueError("k must be greater than 0")

        self.method = method
        self.k = k
        self.p = p
        self.train_df = pd.DataFrame(datapoints)
        self.features = features
        self.target = target
        self.test = test
        self.closest = None

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
    
    def cosine(self,point1, point2):
        if len(point1) != len(point2):
            raise ValueError("Points must have the same dimensions")
        
        distance = np.dot(point1, point2) / (np.linalg.norm(point1) * np.linalg.norm(point2))
        return distance

    def calculate_distances(self):
        method = self.method

        if method == "euclidean":
            self.train_df[method] = self.train_df[self.features].apply(
                lambda row: self.euclidean(row.values, self.test), 
                axis=1)
            
        elif method == "manhattan":
            self.train_df[method] = self.train_df[self.features].apply(
                lambda row: self.manhattan(row.values, self.test),
                axis=1)
        
        elif method == "minkowski":
            self.train_df[method] = self.train_df[self.features].apply(
                lambda row: self.euclidean(row.values, self.test, self.p),
                axis=1)

        elif method == "chebyshev":
            self.train_df[method] = self.train_df[self.features].apply(
                lambda row: self.chebyshev(row.values, self.test),
                axis=1)
            
        elif method == "cosine":
            self.train_df[method] = self.train_df[self.features].apply(
                lambda row: self.cosine(row.values, self.test),
                axis=1)
            
        else:
            raise ValueError("Method_Not_Found")