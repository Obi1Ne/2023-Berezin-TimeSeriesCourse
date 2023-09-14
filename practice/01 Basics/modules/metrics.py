import numpy as np


def ED_distance(ts1, ts2):
    squared_diff = (ts1 - ts2) ** 2
    sum_squared_diff = np.sum(squared_diff)
    ed_dist = np.sqrt(sum_squared_diff) 
    return ed_dist


def norm_ED_distance(ts1, ts2):
    """
    Calculate the normalized Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    norm_ed_dist : float
        The normalized Euclidean distance between ts1 and ts2.
    """

    norm_ed_dist = 0

    # INSERT YOUR CODE 

    return norm_ed_dist


def DTW_distance(ts1, ts2, r=None):
    
    n = len(ts1)

    cost_matrix = np.full((n, n), np.inf)
    cost_matrix[0, 0] = (ts1[0] - ts2[0]) ** 2
    
    if r is None:
        r = max(n, n)

    for i in range(1, n):
        for j in range(max(1, i - r), min(n, i + r)):
            cost = (ts1[i] - ts2[j]) ** 2
            cost_matrix[i, j] = cost + min(cost_matrix[i - 1, j], cost_matrix[i, j - 1], cost_matrix[i - 1, j - 1])
 
    dtw_dist = cost_matrix[n - 1, n - 1]
    
    return dtw_dist
