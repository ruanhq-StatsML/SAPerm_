"""Online permutation feature importance demo, ported from notebook cells 37-41.

The sage-importance and ixai installs in cell 37 are not needed by this file and are not reproduced.
Only numpy, pandas and scikit-learn are required.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class OnlinePFI:
    def __init__(self, model, feature_names, buffer_size=100, VIMP_buffer_size=20, smoothing=0.9,
                 gamma=0.85, loss_func=None):
        self.model = model
        self.feature_names = feature_names
        self.buffer_size = buffer_size
        self.smoothing = smoothing
        self.loss_func = loss_func
        self.n_features = len(feature_names)
        self.buffer = []
        self.importance = np.zeros(self.n_features)
        self.gamma = gamma
        self.importance_buffer = np.zeros((self.n_features, VIMP_buffer_size))
        if loss_func is None:
            self.loss_func = lambda y_true, y_pred: (y_true - y_pred) ** 2
        else:
            self.loss_func = loss_func
        self.n_updates = 0

    def _predict(self, X):
        if len(X.shape) == 1:
            return self.model.predict(X.reshape(-1, 1))
        else:
            return self.model.predict(X)

    def update_batch(self, X_batch, y_batch):
        for x in X_batch:
            self.buffer.append(x.copy())
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)
        if len(self.buffer) < 10:
            return
        y_pred_orig = self._predict(X_batch)
        loss_orig = np.mean([self.loss_func(yt, yp) for yt, yp in zip(y_batch, y_pred_orig)])
        for j in range(self.n_features):
            X_perm = X_batch.copy()
            for i in range(len(X_perm)):
                random_idx = np.random.randint(0, len(self.buffer))
                X_perm[i, j] = self.buffer[random_idx][j]
            y_pred_perm = self._predict(X_perm)
            loss_perm = np.mean([self.loss_func(yt, yp) for yt, yp in
                                 zip(y_batch, y_pred_perm)])
            delta_loss = loss_perm - loss_orig
            self.importance[j] = self.smoothing * self.importance[j] + (1 - self.smoothing) * delta_loss
        self.n_updates += 1

    def return_importance(self):
        importance = self.importance / np.sum(self.importance)
        return dict(zip(self.feature_names, importance))


def main():
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = 5 * X[:, 0] + 2 * X[:, 1] + 0 * X[:, 2] + 0.1 * np.random.randn(1000)

    model = RandomForestRegressor()
    model.fit(X[:500], y[:500])

    pfi = OnlinePFI(
        model=model,
        feature_names=["Feature" + str(i) for i in range(20)],
        buffer_size=25,
        smoothing=0.9
    )

    batch_size = 25
    vimp_dict = []
    for i in range(500, 1000, batch_size):
        X_batch = X[i:(i + batch_size)]
        y_batch = y[i:(i + batch_size)]
        model.fit(X_batch, y_batch)
        pfi.update_batch(X_batch, y_batch)
        vimp_dict.append(pfi.return_importance())

    print(pd.DataFrame(vimp_dict).round(4).to_string())


if __name__ == '__main__':
    main()
