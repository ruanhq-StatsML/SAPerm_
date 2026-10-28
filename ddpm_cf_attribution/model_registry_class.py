#Model registry class:
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import warnings
import inspect 
import xgboost as xgb
from typing import Any, Callable, Mapping, MutableMapping, Optional, Union 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor

def _rng_seed(seed):
    return None if seed is None else int(seed)

def _as_1d(y):
    return np.asarray(y, dtype = float).ravel()

def _as_1d_float(y):
    return _as_1d(y)

def _as_2d(X):
    Xa = np.asarray(X, dtype = float)
    if Xa.ndim == 1:
        return Xa.reshape(-1, 1) #(n, ) -> (n, 1)
    return Xa

def _as_2d_float(X):
    return _as_2d(X)

def _standardize_cols(X):
    cen = np.mean(X, axis = 0)
    sc = np.std(X, axis = 0, ddof = 0)
    sc = np.where(sc <= 1e-12, 1.0, sc)
    return (X - cen)/sc, cen, sc

FitFn = Callable[..., Any]
PredictFn = Callable[..., np.ndarray]

#The model adapter class -- Config!
@dataclass(frozen = True)
class ModelAdapter:
    """ 
    The model adapter contract aligned with: name, fit, predict,
    For the entire model factory, return {'name', 'fit', 'predict'} enhanced together here.
    """
    name: str
    fit: FitFn
    predict: PredictFn

class ModelRegistry:
    #The factory for incorporating all of the models with both classification and regression tasks.
    def __init__(
        self, *,
        ntree = 150,
        ridge_alpha = 0.25,
        nthread = 1,
        maxit = 200,
        max_depth = 5,
        gamma = 0.25,
        eta = 0.15,
        mlp_hidden_size = 7,
        mlp_decay = 1e-5,
        mlp_max_iter = 500,
        mlp_trace = False,
        mlp_max_coef_reg = 10000,
        mlp_max_coef_clf = 10000,
        warn_xgb_labels = True,
        positive_class = 1,
        activation = 'relu'
    ):
        self.ntree = int(ntree)
        self.positive_class = positive_class
        self.ridge_alpha = float(ridge_alpha)
        self.nthread = int(nthread)
        self.maxit = int(maxit)
        self.max_depth = int(max_depth)
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.mlp_hidden_size = int(mlp_hidden_size)
        self.mlp_decay = float(mlp_decay)
        self.mlp_max_iter = int(mlp_max_iter)
        self.mlp_trace = bool(mlp_trace)
        self.warn_xgb_labels = bool(warn_xgb_labels)
        self.ridge_alpha = float(ridge_alpha)
        self.activation = activation
        self.mlp_max_coef_reg = float(mlp_max_coef_reg)
        self.mlp_max_coef_clf = float(mlp_max_coef_clf)
    def make_rf_regression(self):
        def fit(X, Y, seed = None):
            s = _rng_seed(seed)
            Xa = np.asarray(X, dtype = float)
            Ya = np.asarray(Y, dtype = float).ravel()
            n, p = Xa.shape
            return RandomForestRegressor(
                n_estimators = int(self.ntree),
                max_features = max(1, round(p/3)),
                min_samples_leaf = max(1, int(round(np.sqrt(n)/2))),
                random_state = s,
                n_jobs = (self.nthread if self.nthread > 1 else None)
            ).fit(Xa, Ya)
        def predict(fit_obj, X_new):
            prediction = np.asarray(fit_obj.predict(np.asarray(X_new, dtype = float)),
                dtype = float).ravel()
            return prediction
        return ModelAdapter(name = 'rf_regressor', fit = fit, predict = predict)
    #Binary:
    def make_rf_classification(self):
        def fit(X, Y, seed = None):
            s = _rng_seed(seed)
            Xa = np.asarray(X, dtype = float)
            Ya = np.asarray(Y, dtype = float).ravel()
            n, p = Xa.shape
            return RandomForestClassifier(
                n_estimators = int(self.ntree),
                max_features = float(min(1.0, np.sqrt(p)/p)),
                min_samples_leaf = max(1, int(round(np.sqrt(n)/2))),
                random_state = s,
                n_jobs = (self.nthread if self.nthread > 1 else None)
            ).fit(Xa, Ya)
        def predict(fit_obj, X_new):
            probs = np.asarray(
                fit_obj.predict_proba(_as_2d_float(X_new)),
                dtype = float
            )
            classes = np.asarray(fit_obj.classes_)
            col = np.where(classes == self.positive_class)[0]
            return probs[:, col].ravel()
        return ModelAdapter(name = 'rf_classifier', fit = fit, predict = predict)
    #MultiNomial Classifier:
    def make_xgboost_regressor(self):
        n_rounds = int(self.ntree)
        def fit(X, y, seed):
            s = _rng_seed(seed)
            dtrain = xgb.DMatrix(_as_2d_float(X), label = _as_1d_float(y))
            params = dict(
                max_depth = int(self.max_depth),
                gamma = float(self.gamma),
                eta = float(self.eta),
                objective = 'reg:squarederror',
                nthread = int(self.nthread)
            )
            if s is not None:
                params['seed'] = int(s) % (2 ** 32)
            return xgb.train(params, dtrain, num_boost_round = int(n_rounds))
        def predict(fit_obj, X_new):
            d = xgb.DMatrix(_as_2d_float(X_new))
            out = np.asarray(fit_obj.predict(d), dtype = float).ravel()
            return out
        return ModelAdapter(name = 'xgb_regressor', fit = fit, predict = predict)
    #Binary: 
    def make_xgb_classifier(self):
        def fit(X, y, seed):
            y01 = np.asarray(y, dtype = int).ravel()
            labels = np.unique(y01)
            dtrain = xgb.DMatrix(_as_2d_float(X), label = (y01 == self.positive_class).astype(float))
            params = dict(
                max_depth = int(self.max_depth),
                gamma = float(self.gamma),
                eta = float(self.eta),
                objective = 'binary:logistic',
                nthread = int(self.nthread),
                eval_metric = 'logloss'
            )
            if seed is not None:
                params['seed'] = int(seed) % (2 ** 32)
            booster = xgb.train(params, dtrain, num_boost_round = int(self.ntree))
            return booster, labels.astype(int)
        def predict(fit_obj: Any, X_new):
            booster, labels = fit_obj
            d = xgb.DMatrix(_as_2d_float(X_new))
            p_pos = np.asarray(booster.predict(d), dtype = float).ravel()
            return p_pos
        #It's another config.
        return ModelAdapter(name = 'xgb_classifier', fit = fit, predict = predict) 
    #Specify the ridge regression:
    def make_ridge_regression(self):
        def fit(X, y, seed):
            _ = _rng_seed(seed)
            return Ridge(alpha = float(self.ridge_alpha),
                random_state = _rng_seed(seed)).fit(_as_2d_float(X), _as_1d_float(y))
        def predict(fit_obj: Ridge, X_new):
            return np.asarray(fit_obj.predict(_as_2d_float(X_new)), dtype = float).ravel()
        return ModelAdapter(name = 'ridge_regressor', fit = fit, predict = predict)
    #You may need to specify the positive class here:
    def make_ridge_classification(self):
        pc = self.positive_class
        def fit(X, y, seed = None):
            s = _rng_seed(seed)
            return LogisticRegression(
                penalty = 'l2',
                C = 1.0/float(self.ridge_alpha) if self.ridge_alpha > 0 else 1e12,
                solver = 'lbfgs',
                max_iter = int(self.maxit),
                random_state = s
            ).fit(_as_2d_float(X), np.asarray(y).ravel())
        def predict(fit_obj, X_new):
            probs = np.asarray(
                fit_obj.predict_proba(_as_2d_float(X_new)),
                dtype = float
            ) 
            classes = np.asarray(fit_obj.classes_)
            col = int(np.where(classes == pc)[0][0])
            return probs[:, col].ravel()
        return ModelAdapter(name = 'ridge_classifier', fit = fit, predict = predict)
    #How many levels in the response here?
    def make_logistic_classification(self):
        positive_class = self.positive_class
        maxit = self.maxit
        def fit(X, y, seed = None):
            s = _rng_seed(seed)
            return LogisticRegression(
                solver = 'lbfgs',
                max_iter = maxit,
                random_state = s
            ).fit(_as_2d_float(X), np.asarray(y).ravel())
        def predict(fit_obj, X_new):
            probs = np.asarray(
                fit_obj.predict_proba(_as_2d_float(X_new)),
                dtype = float
            )
            classes = np.asarray(fit_obj.classes_)
            col = np.where(classes == positive_class)[0]
            return probs[:, int(col[0])].ravel()#identify the position of the positive class.
        return ModelAdapter(name = 'logistic_classifier', fit = fit, predict = predict)
    def make_multinomial_classifier(self):
        max_iter = self.maxit
        def fit(X, y, seed = None):
            s = _rng_seed(seed)
            # sklearn>=1.5 dropped multi_class; multinomial is selected by solver/data
            return LogisticRegression(
                solver = 'lbfgs', max_iter = max_iter
            ).fit(_as_2d_float(X), np.asarray(y).ravel())
        def predict(fit_obj, X_new):
            probs = np.asarray(fit_obj.predict_proba(_as_2d_float(X_new)),
                dtype = float)
            classes_outputs = fit_obj.classes_[np.argmax(probs, axis = 1)].ravel()
            return classes_outputs
        return ModelAdapter(name = 'multinomial_classifier', fit = fit, predict = predict)
    #making the linear model:
    def make_lm_regression(self):
        def fit(X, y, seed):
            s = _rng_seed(seed)
            return LinearRegression().fit(np.asarray(X, dtype = float), _as_1d_float(y))
        def predict(fit_obj, X_new):
            return np.asarray(fit_obj.predict(_as_2d_float(X_new)),
              dtype = float).ravel()
        return ModelAdapter(name = 'linear_regressor', fit = fit, predict = predict)
    def make_mlp_regression(self):
        def fit(X: Any, y: Any, seed):
            s = _rng_seed(seed)
            Xs, center, scale = _standardize_cols(_as_2d_float(X))
            coef_lim = None
            m = MLPRegressor(
                hidden_layer_sizes = (int(self.mlp_hidden_size), ),
                activation = self.activation,
                alpha = float(self.mlp_decay),
                max_iter = int(self.mlp_max_iter),
                random_state = s,
                early_stopping = True,
                verbose = int(self.mlp_trace)
            ).fit(Xs, _as_1d_float(y))
            return {'model': m, 'center': center, 'scale': scale}
        def predict(fit_bundle, X_new):
            m = fit_bundle['model']
            cen = np.asarray(fit_bundle['center'], dtype = float)
            sc = np.asarray(fit_bundle['scale'], dtype = float)
            Xn = (_as_2d_float(X_new) - cen)/sc#you only need a 2-dimensional array here.
            return np.asarray(m.predict(Xn), dtype = float).ravel()
        return ModelAdapter(name = 'mlp_regressor', fit = fit, predict = predict)
    def make_mlp_classifier(self):
        def fit(X, y, seed):
            s = _rng_seed(seed)
            y_raw = np.asarray(y).ravel()
            classes = np.unique(y_raw)
            if classes.size != 2:
                raise ValueError("MLP Classifier Currently Support Binary Labels only.")
            Xs, center, scale = _standardize_cols(_as_2d_float(X))
            bin_idx = (y_raw == self.positive_class).astype(int)
            m = MLPClassifier(
                hidden_layer_sizes = (int(self.mlp_hidden_size), ),
                activation = 'relu',
                alpha = float(self.mlp_decay),
                max_iter = int(self.mlp_max_iter),
                random_state = s,
                early_stopping = True,
                verbose = bool(self.mlp_trace)
            ).fit(Xs, bin_idx)
            return {'model': m, 'center': center, 'scale': scale}
        #Scaling for the new batch of data is rather crucial here. (X_new - center)/sc here
        def predict(fit_obj, X_new):
            m = fit_obj['model']
            center = np.asarray(fit_obj['center'], dtype = float)
            sc = np.asarray(fit_obj['scale'], dtype = float)
            Xn = (_as_2d_float(X_new) - center)/sc
            probs = np.asarray(m.predict_proba(Xn), dtype = float)
            return probs[:, 1].ravel()
        return ModelAdapter(name = 'mlp_classifier', fit = fit, predict = predict)
    def make_mlp_multi_classifier(self):
        def fit(X, y, seed = 2026):
            s = _rng_seed(seed)
            classes, y_enc = np.unique(np.asarray(y).ravel(), return_inverse = True)
            Xs, center, scale = _standardize_cols(_as_2d_float(X))
            m = MLPClassifier(hidden_layer_sizes = (int(self.mlp_hidden_size), ),
                activation = 'relu', alpha = float(self.mlp_decay), max_iter = int(self.mlp_max_iter),
                random_state = s, early_stopping = True, verbose = bool(self.mlp_trace)).fit(Xs, y_enc.astype(int))
            return {'model': m, 'center': center, 'scale': scale}
        def predict(fit_obj, X_new):
            m = fit_obj['model']
            center = np.asarray(fit_obj['center'], dtype = float)
            sc = np.asarray(fit_obj['scale'], dtype = float)
            Xn = (_as_2d_float(X_new) - center)/sc
            probs = np.asarray(m.predict_proba(Xn), dtype = float)
            return probs
        return ModelAdapter(name = 'mlp_multi_classifier', fit = fit, predict = predict)
    #putting the models into one individual class:
    def adapters(self):
        return {
            'rf_regressor': self.make_rf_regression(),
            'rf_classifier': self.make_rf_classification(),
            'lm_regressor': self.make_lm_regression(),
            'logistic_classifier': self.make_logistic_classification(),
            'multinomial_classifier': self.make_multinomial_classifier(),
            'xgb_regressor': self.make_xgboost_regressor(),
            'xgb_classifier': self.make_xgb_classifier(),
            'ridge_regressor': self.make_ridge_regression(),
            'ridge_classifier': self.make_ridge_classification(),
            'mlp_regressor': self.make_mlp_regression(),
            'mlp_classifier': self.make_mlp_classifier(),
            'mlp_multi_classifier': self.make_mlp_multi_classifier()
        }
    def as_r_style_dict(self):
        mapping = self.adapters()
        registry_map = {
          k: {'name': v.name, 'fit': v.fit, 'predict': v.predict}
          for k, v in mapping.items()
          }
        return registry_map

#Initiate this ModelRegistryFactory to transform them to another dictionary:
def default_model_registry(**registry_kwargs):
    mr = ModelRegistry(**registry_kwargs)
    return mr.as_r_style_dict()
#Specify the list of the parameters here:
#registry_kwargs = {}
#mr = ModelRegistry(**registry_kwargs)




















