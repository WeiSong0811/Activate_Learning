import numpy as np

class KMeans:
    def __init__(self,
                 n_clusters=3,
                 max_iter=100,
                 tol=1e-4,
                 random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids = None
        self.labsels_ = None
        self.inertia_ = None

    def _initialize_centroids(self, X):
        '''
        从原始样本里随机选取K个点，并把他们当作初始中心
        '''
        rng = np.random.default_rng(self.random_state)
        indics = rng.choice(X.shape[0],
                            size=self.n_clusters,
                            replace=False)
        return X[indics].copy()
    
    def _compute_distances(self, X, centroids):
        '''
        计算每个样本到各个聚类中心的距离
        '''
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        return distances

    def _assign_labels(self, X, centroids):
        '''
        根据距离将每个样本分配到最近的聚类中心
        '''
        distances = self._compute_distances(X, centroids)
        labels = np.argmin(distances, axis=1)
        return labels
    
    def _update_centroids(self, X, labels):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]

            if len(cluster_points) > 0:
                random_idx = np.random.randint(0, X.shape[0])
                new_centroids[k] = X[random_idx]

            else:
                new_centroids[k] = cluster_points.mean(axis=0)
        return new_centroids
    
    def _compute_inertia(self, X, labels, centroids):
        total = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                total += ((cluster_points - centroids[k]) ** 2).sum()
        return total
    
    def fit(self, X):
        X = np.asarray(X, dtype=float)

        centroids = self._initialize_centroids(X)

        for _ in range(self.max_iter):
            labels = self._assign_labels(X, centroids)
            new_centroids = self._update_centroids(X, labels)

            shift = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max()

            centroids = new_centroids
            if shift < self.tol:
                break

        self.centroids = centroids
        self.labels_ = labels
        self.inertia_ = self._compute_inertia(X, labels, centroids)

        return self
    
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return self._assign_labels(X, self.centroids)
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_