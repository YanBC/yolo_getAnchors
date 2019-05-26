import numpy as np 
from scipy.spatial import distance


def _kmeans(X, k, metric, average_fn, max_iter):
    """
    Runs the kmeans algorithm once to cluster X into
    k clusters.

    INPUT:
        X <2d array>: The data to be clustered, each row
                        represents one data point, and each
                        column represents a feature.

        k <int>: Number of clusters.

        metric <fn>: A function that takes in two arrays and 
                        returns a number. This is to be used
                        as the distance function. 

        average_fn <fn>: A function that takes in a matrix X
                            and returns the center of the points.

        max_iter <int>: Max number of iterations to run before
                        stopping the algorithm.

    OUTPUT:
        inertia <float>: Sum of distances for each point to its centroid
                            Where distance is taken as the metric argument.

        centroids <2d array>: Each row represents the coordinates of one centroid

        assignments <array>: A list of assignments.
    """
    centroids = X[np.random.choice(X.shape[0], size=k, replace=False), :]
    inertia, delta = None,  100
    for _ in range(max_iter):
        distances = np.array([np.apply_along_axis(lambda c: metric(x, c), 1, centroids) for x in X])
        assignments = np.argmin(distances, 1)
        for i in range(k):
            points = X[np.where(assignments == i)]
            if len(points) > 0:
                centroids[i] = average_fn(points)

        new_inertia = sum([metric(X[row], centroids[clust]) for row, clust in enumerate(assignments)])
        if inertia is None: delta = new_inertia
        else: delta = new_inertia - inertia
        inertia = new_inertia

        if np.abs(delta) < 0.0001:
            break

    return inertia, centroids, assignments


class KMeans():
    """
    KMeans algorithm. Variable and function names are modelled
    after scikit's implementation. This one allows for arbitrary
    distance metrics (you can define your own or use one from 
    scipy.spatial.distance)

    PARAMETERS:
        k <int>: Number of clusters

        n_init <int>: Number of initializations to try
        
        metric <fn>: Function that takes in two arrays and returns a number
                        (used as the distance metric)

        average_fn <fn>: A function that takes in a matrix X
                            and returns the center of the points.

        max_iter <int>: Maximum number of iterations to run before the 
                        algorithm terminates

    ATTRIBUTES:
        inertia_ <float>: Sum of distances for each point to its centroid
                            Where distance is taken as the metric argument.

        cluster_centers_ <2d array>: Each row represents the coordinates of one centroid

        labels_ <array>: A list of assignments.

    """

    def __init__(self, k, n_init=10, metric=distance.euclidean, 
                average_fn=lambda x: np.mean(x, axis=0), max_iter=100):
        self._k = k
        self._n_init = n_init
        self._metric = metric
        self._max_iter = max_iter
        self.inertia_ = np.Inf
        self._average_fn = average_fn
        self.cluster_centers_ = None

    def fit(self, X):
        """
        Fits the KMeans object to data in X. Populates 
        self.inertia_, self.cluster_centers_, and self.labels_

        INPUT:
            X <2d array>:   Matrix of data points to cluster, where each row
                            is a data point, and each column is a feature
        OUTPUT:
            self
        """
        for _ in range(self._n_init):
            inertia, centers, labels = _kmeans(X, self._k, self._metric, self._average_fn, self._max_iter)
            if inertia < self.inertia_:
                self.inertia_ = inertia
                self.cluster_centers_ = centers
                self.labels_ = labels
        return self

    def fit_predict(self, X):
        """
        Fits the KMeans object to the data in X. Returns the resulting labels

        INPUT:
            X <2d array>:   Matrix of data points to cluster, where each row
                            is a data point, and each column is a feature
        
        OUTPUT:
            self.labels <list>: The cluster number assigned to each data point.

        """
        self.fit(X)
        return self.labels_

    def predict(self, X):
        """
        Predicts which clusters the rows of X belong to 

        INPUT:
            X <2d array>:   Matrix of data points to cluster, where each row
                            is a data point, and each column is a feature

        OUTPUT:
            predictions <list int>: A list of predicted cluster assignments of each point
        """
        if self.cluster_centers_ is None:
            raise ValueError('Run fit first!')

        distances = np.array([np.apply_along_axis(lambda c: self._metric(x, c), 1, self.cluster_centers_) for x in X])
        return np.argmin(distances, 1)


