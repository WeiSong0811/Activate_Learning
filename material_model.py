import numpy as np

class MaterialSimulator:
    def __init__(self, noise_std=0.5, random_state=None):
        self.noise_std = noise_std
        self.rng = np.random.default_rng(random_state)

    def __call__(self, X):
        X = np.asarray(X)
        assert X.shape[1] == 10, "Input must have 10 features"

        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = [X[:, i] for i in range(10)]

        # --- label 1: 强度 ---
        y1 = (
            2.0 * x1 +
            1.5 * x2 -
            3.0 * (x3 - 0.5)**2 +
            2.0 * np.sin(np.pi * x4) +
            1.2 * x5 * x6
        )

        # --- label 2: 刚度 ---
        y2 = (
            5.0 * np.exp(-x1) +
            2.0 * x2**2 +
            1.5 * x7 +
            np.cos(2 * np.pi * x8) +
            x9 * x10
        )

        # --- label 3: 延性 ---
        y3 = (
            3.0 * np.exp(-(x1 - 0.5)**2) +
            2.0 * np.sin(2 * np.pi * x3 * x4) +
            1.5 * x6 -
            x7**2 +
            0.5 * x8 * x9
        )
        
        # 加噪声（模拟实验/仿真误差）
        noise = self.noise_std * self.rng.normal(size=(X.shape[0], 3))

        Y = np.stack([y1, y2, y3], axis=1) + noise
        return Y