import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class TimeSeriesHierarchicalClustering:
    """
    Hierarchical Clustering of time series.

    Parameters
    ----------

    n_clusters : int, default = 3
        The number of clusters.
    
    method : str, default = 'complete'
        The linkage criterion.
        Options: {single, complete, average, weighted}.

    model : sklearn object
        An sklearn agglomerative clustering.
    """
    def __init__(self, n_clusters=2, method='complete'):

        self.n_clusters = n_clusters
        self.method = method
        self.model = None

    def _create_linkage_matrix(self):
        counts = np.zeros(self.model.children_.shape[0])
        n_samples = len(self.model.labels_)

        for i, merge in enumerate(self.model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([self.model.children_, self.model.distances_, counts]).astype(float)

        return linkage_matrix

    def fit(self, distance_matrix):
 
        self.model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.method)

        self.model.fit(distance_matrix)
            
        return self

    def _draw_timeseries_allclust(self, dx, labels, leaves, gs, ts_hspace):

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        margin = 7

        max_cluster = len(leaves)
        # flip leaves, as gridspec iterates from top down
        leaves = leaves[::-1]

        for cnt in range(len(leaves)):
            plt.subplot(gs[cnt:cnt+1, max_cluster-ts_hspace:max_cluster])
            plt.axis("off")

            leafnode = leaves[cnt]
            ts = dx[leafnode]
            ts_len = ts.shape[0] - 1

            label = int(labels[leafnode])
            color_ts = colors[label]

            plt.plot(ts, color=color_ts)
            plt.text(ts_len+margin, 0, f'class = {label}')

    def plot_dendrogram(self, df, labels, ts_hspace=12, title='Dendrogram'):

        max_cluster = len(self.linkage_matrix) + 1

        plt.figure(figsize=(12, 9))

        gs = gridspec.GridSpec(max_cluster, max_cluster)

        plt.subplot(gs[:, 0 : max_cluster - ts_hspace - 1])
        plt.xlabel("Distance")
        plt.ylabel("Cluster")
        plt.title(title, fontsize=16, weight='bold')

        ddata = dendrogram(self.linkage_matrix, orientation="left", color_threshold=sorted(self.model.distances_)[-2], show_leaf_counts=True)

        self._draw_timeseries_allclust(df, labels, ddata["leaves"], gs, ts_hspace)        
        