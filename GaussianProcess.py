import numpy as np

class GP:
    def __init__(self, length_scale=1.0, sigma_f=1.0, sigma_n=1e-6, agg_weights=None):

        '''
        length_scale: RBF核的长度尺度，控制函数变化快慢
        sigma_f: RBF核的信号方差，控制函数的幅度
        sigma_n: 噪声方差，控制观测数据的噪声
        agg_weights: 聚合权重，用于多任务学习
        '''

        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.agg_weights = agg_weights

        self.X_train = None
        self.y_train = None
        self.K = None
        self.is_multi_output = False
        self.n_outputs = 1

        self.alpha = None
        self.alphas = None

    def rbf_kernel(self, X1, X2):
        '''
        计算RBF kernel matrix
        X1: shape (n1, d)
        X2: shape (n2, d)
        return: shape (n1, n2)
        '''

        X1 = np.asarray(X1, dtype=float)
        X2 = np.asarray(X2, dtype=float)

        sq_dist = (
            np.sum(X1**2, axis=1).reshape(-1, 1) + 
            np.sum(X2**2, axis=1).reshape(1, -1) -
            2 * np.dot(X1, X2.T)
        )

        K = self.sigma_f**2 * np.exp(-0.5 / self.length_scale**2 * sq_dist)

        return K
    
    def _get_weights(self):
        if self.n_outputs == 1:
            return np.array([1])
        
        if self.agg_weights is None:
            weights = np.ones(self.n_outputs) / self.n_outputs

        else:
            weights = np.array(self.agg_weights, dtype=float)
            if len(weights) != self.n_outputs:
                raise ValueError("agg_weights length must match number of outputs")
            if np.any(weights < 0):
                raise ValueError("agg_weights must be non-negative")
            s = np.sum(weights)
            if s <= 0:
                raise ValueError("agg_weights must sum to a positive value")
            if s != 1:
                raise ValueError("agg_weights must sum to 1")
        
        return weights
    
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train, dtype=float)
        y_train = np.array(y_train, dtype=float)

        if y_train.ndim == 1:
            self.is_multi_output = False
            self.n_outputs = 1
            self.y_train = y_train.reshape(-1, 1)
        elif y_train.ndim == 2:
            if y_train.shape[1] == 1:
                self.is_multi_output = False
                self.n_outputs = 1
            else:
                self.is_multi_output = True
                self.n_outputs = y_train.shape[1]
            self.y_train = y_train
        else:
            raise ValueError("y_train must be 1D or 2D array")
        
        self.K = self.rbf_kernel(self.X_train, self.X_train)

        n = self.X_train.shape[0]
        self.K += (self.sigma_n**2) * np.eye(n)

        if not self.is_multi_output:
            self.alpha = np.linalg.solve(self.K, self.y_train.reshape(-1, 1))
            self.alphas = None
        else:
            self.alphas = np.linalg.solve(self.K, self.y_train)
            self.alpha = None


    def predict(self, X_test, return_std=True):
        X_test = np.array(X_test, dtype=float)

        K_star = self.rbf_kernel(X_test, self.X_train)

        if not self.is_multi_output:
            mean = (K_star @ self.alpha).ravel()
            if not return_std:
                return {
                    'mean': mean
                }
            K_star_star = self.rbf_kernel(X_test, X_test)
            v = np.linalg.solve(self.K, K_star.T)
            cov = K_star_star - K_star @ v
            var = np.maximum(np.diag(cov), 0.0)
            std = np.sqrt(var)
            return {
                'mean': mean,
                'std': std
            }
        
        means_per_label = K_star @ self.alphas
        
        if not return_std:
            weights = self._get_weights()
            mean_agg = means_per_label @ weights

            return {
                'mean_per_label': means_per_label,
                'mean_agg': mean_agg
            }
        K_star_star = self.rbf_kernel(X_test, X_test)
        v = np.linalg.solve(self.K, K_star.T)
        cov = K_star_star - K_star @ v

        var_shared = np.maximum(np.diag(cov), 0.0)
        std_shared = np.sqrt(var_shared)

        stds_per_label = np.tile(std_shared.reshape(-1, 1), (1, self.n_outputs))
        weights = self._get_weights()
        mean_agg = means_per_label @ weights

        vars_per_label = stds_per_label**2
        var_agg = vars_per_label @ weights
        std_agg = np.sqrt(var_agg)

        return {
            'mean_per_label': means_per_label,
            'std_per_label': stds_per_label,
            'mean_agg': mean_agg,
            'std_agg': std_agg
        }
    
