from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np

class DAU:

  def __init__(self,n_neighbors:int=3,min_samples:int=5,eps:float=0.05,percentile:int=25):
    self.n_neighbors = n_neighbors
    self.min_samples = min_samples
    self.eps = eps
    self.percentile = percentile

  def fit_transform(self,X:pd.DataFrame,y:pd.Series):
    nearest_neighbors = NearestNeighbors(n_neighbors=self.n_neighbors)
    nearest_neighbors.fit(X)
    distances,indices = nearest_neighbors.kneighbors(X)
    avg_distances = np.mean(distances,axis=1)
    threshold = np.percentile(avg_distances, self.percentile)

    sparse_idx = np.where(avg_distances > threshold)[0]
    dense_idx = np.where(avg_distances < threshold)[0]
    dense_data = X.iloc[dense_idx]

    dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
    cluster_labels = dbscan.fit_predict(dense_data)
    dense_representatives = []
    unique_labels = np.unique(cluster_labels)

    for label in unique_labels:
      if label == -1: continue
      cluster_local_indices = np.where(cluster_labels == label)[0]
      representative_original_idx = dense_idx[cluster_local_indices[0]]
      dense_representatives.append(representative_original_idx)

    noise_indices = dense_idx[cluster_labels == -1]
    final_keep_idx = np.concatenate( [sparse_idx, np.array(dense_representatives), noise_indices])
    final_keep_idx = np.unique(final_keep_idx)
    X_majority_reduced = X.iloc[final_keep_idx]
    y_majority_reduced = y.iloc[final_keep_idx]

    return X_majority_reduced, y_majority_reduced