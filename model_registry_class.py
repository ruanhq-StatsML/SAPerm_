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
    if seed is None:
        return None
    return int(seed)


def _as_1d_float(values):
    return np.asarray(values, dtype=float).ravel()


def _as_2d_float(values):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    return array


def _standardize_cols(X):
    center = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale = np.where(scale == 0, 1.0, scale)
    return (X - center) / scale, center, scale



#Model Registry Class:
FitFn = Callable[..., Any]
PredictFn = Callable[..., np.ndarray]
VIMPFn = Callable[..., np.ndarray]

#The model adapter class:
@dataclass(frozen = True)
class ModelAdapter:
    """ 
    The model adapter contract aligned with: name, fit, predict,
    For the entire model factory, return {'name', 'fit', 'predict'} enhanced together here.
    """
    name: str
    fit: FitFn
    predict: PredictFn
    VIMP: VIMPFn

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
                min_samples_leaf = round(np.sqrt(n)/2),
                random_state = s,
                n_jobs = (self.nthread if self.nthread > 1 else None)
            ).fit(Xa, Ya)
        def predict(fit, X_new):
            prediction = np.asarray(fit.predict(np.asarray(X_new, dtype = float)),
                dtype = float).ravel()
            return prediction
        #Here we also leverage the Brier Score here:
        def VIMP(fit_obj, X_new, Y_new, VIMP_type = 'LOCO'):
            k = X_new.shape[1]
            orig_pred = (Y_new - fit_obj.predict(X_new)) ** 2
            VIMP_ls = np.zeros(k)
            if VIMP_type == 'LOCO':
                for j in range(k):
                    vimp_ls[j] = MSE_orig - np.mean((Y_new - fit_obj.predict(X_new[:, np.setdiff1d(np.arange(k), j)])) ** 2)
            elif VIMP_type == 'permutation':
                for j in range(k):
                    X_new_perm = X_new.copy()
                    X_new_perm[:, k] = np.random.permutation(X_new_perm[:, k])
                    vimp_ls[j] = MSE_orig - np.mean((Y_new - fit_obj.predict(X_new_perm)) ** 2)
            #permutation variable importance:
            else:
                raise ValueError('Only Support LOCO and PermutationVIMP, input one of these two notions')
            return vimp_ls, np.argsort(-np.asarray(vimp_ls, type = np.float32))
        return ModelAdapter(name = 'rf_regressor', fit = fit, predict = predict, VIMP = VIMP)
    #Binary:
    def make_rf_classification(self):
        def fit(X, Y, seed = None):
            s = _rng_seed(seed)
            Xa = np.asarray(X, dtype = float)
            Ya = np.asarray(Y, dtype = float).ravel()
            n, p = Xa.shape
            return RandomForestClassifier(
                n_estimators = int(self.ntree),
                max_features = min(1, np.sqrt(p)/p),
                min_samples_leaf = round(np.sqrt(n)/2),
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
        #Here we also leverage the Brier Score here:
        def VIMP(fit_obj, X_new, Y_new, VIMP_type = 'LOCO'):
            k = X_new.shape[1]
            orig_pred = (Y_new - fit_obj.predict(X_new)) ** 2
            VIMP_ls = np.zeros(k)
            if VIMP_type == 'LOCO':
                for j in range(k):
                    vimp_ls[j] = MSE_orig - np.mean((Y_new - fit_obj.predict(X_new[:, np.setdiff1d(np.arange(k), j)])) ** 2)
            elif VIMP_type == 'permutation':
                for j in range(k):
                    X_new_perm = X_new.copy()
                    X_new_perm[:, k] = np.random.permutation(X_new_perm[:, k])
                    vimp_ls[j] = MSE_orig - np.mean((Y_new - fit_obj.predict(X_new_perm)) ** 2)
            #permutation variable importance:
            else:
                raise ValueError('Only Support LOCO and PermutationVIMP, input one of these two notions')
            return vimp_ls, np.argsort(-np.asarray(vimp_ls, type = np.float32))
        #In the model here via the MSE values or the multinomial classifier here right? Very concise and very helpful, significant improvement can be leveraged.
        return ModelAdapter(name = 'rf_classifier', fit = fit, predict = predict, VIMP = VIMP)
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
            return xgb.train(params, dtrain, num_boost_round = int(n_rounds))
        def predict(fit_obj, X_new):
            d = xgb.DMatrix(_as_2d_float(X_new))
            out = np.asarray(fit_obj.predict(d), dtype = float).ravel()
            return out
        #Here we also leverage the Brier Score here:
        def VIMP(fit_obj, X_new, Y_new, VIMP_type = 'LOCO'):
            k = X_new.shape[1]
            orig_pred = (Y_new - fit_obj.predict(X_new)) ** 2
            VIMP_ls = np.zeros(k)
            if VIMP_type == 'LOCO':
                for j in range(k):
                    vimp_ls[j] = MSE_orig - np.mean((Y_new - fit_obj.predict(X_new[:, np.setdiff1d(np.arange(k), j)])) ** 2)
            elif VIMP_type == 'permutation':
                for j in range(k):
                    X_new_perm = X_new.copy()
                    X_new_perm[:, k] = np.random.permutation(X_new_perm[:, k])
                    vimp_ls[j] = MSE_orig - np.mean((Y_new - fit_obj.predict(X_new_perm)) ** 2)
            #permutation variable importance:
            else:
                raise ValueError('Only Support LOCO and PermutationVIMP, input one of these two notions')
            return vimp_ls, np.argsort(-np.asarray(vimp_ls, dtype = np.float32))
        return ModelAdapter(name = 'xgb_regressor', fit = fit, predict = predict, VIMP = VIMP)
    #df.reshape(-1, 1):
    def make_xgb_multiclassifier(self):
        def fit(X, y, seed):
            n_rounds = int(self.ntree)
            s = _rng_seed(seed)
            _, y_enc = np.unique(np.asarray(y).ravel(), return_inverse = True)
            dtrain = xgb.DMatrix(_as_2d_float(X), label = y_enc.astype(int))
            params = dict(
                max_depth = int(self.max_depth),
                gamma = float(self.gamma),
                eta = float(self.eta),
                objective = 'multi:softprob',
                nthread = int(self.nthread),
                eval_metric = 'mlogloss'
            )
            booster = xgb.train(params, dtrain, num_boost_round = n_rounds)
            return booster, y_enc.astype(int)
        def predict(fit_obj, X_new):
            booster, label = fit_obj
            d = xgb.DMatrix(_as_2d_float(X_new))
            #output all of the probability here:
            out = np.asarray(booster.predict_proba(d), dtype = float)
            return out
        #Here we also leverage the Brier Score here:
        def VIMP(fit_obj, X_new, Y_new, VIMP_type = 'LOCO'):
            k = X_new.shape[1]
            orig_pred = (Y_new - fit_obj.predict(X_new)) ** 2
            VIMP_ls = np.zeros(k)
            if VIMP_type == 'LOCO':
                for j in range(k):
                    vimp_ls[j] = np.mean((Y_new - fit_obj.predict(X_new[:, np.setdiff1d(np.arange(k), j)])) ** 2) - MSE_orig
            elif VIMP_type == 'permutation':
                for j in range(k):
                    X_new_perm = X_new.copy()
                    X_new_perm[:, k] = np.random.permutation(X_new_perm[:, k])
                    vimp_ls[j] = np.mean((Y_new - fit_obj.predict(X_new_perm)) ** 2) - MSE_orig
            #permutation variable importance:
            else:
                raise ValueError('Only Support LOCO and PermutationVIMP, input one of these two notions')
            return vimp_ls, np.argsort(-np.asarray(vimp_ls, dtype = np.float32))
        return ModelAdapter(name = 'xgb_multiclassifier', fit = fit, predict = predict, VIMP = VIMP)
    #Binary: 
    def make_xgb_classifier(self):
        def fit(X, y, seed):
            ntree = self.ntree
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
            booster = xgb.train(params, dtrain, num_boost_round = int(ntree))
            return booster, labels.astype(int)
        def predict(fit_obj: Any, X_new):
            booster, labels = fit_obj
            d = xgb.DMatrix(_as_2d_float(X_new))
            p_pos = np.asarray(booster.predict(d), dtype = float).ravel()
            return p_pos
        #Here we also leverage the Brier Score here:
        def VIMP(fit_obj, X_new, Y_new, VIMP_type = 'LOCO'):
            k = X_new.shape[1]
            orig_pred = (Y_new - fit_obj.predict(X_new)) ** 2
            VIMP_ls = np.zeros(k)
            if VIMP_type == 'LOCO':
                for j in range(k):
                    vimp_ls[j] = np.mean((Y_new - fit_obj.predict(X_new[:, np.setdiff1d(np.arange(k), j)])) ** 2) - orig_pred
            elif VIMP_type == 'permutation':
                for j in range(k):
                    X_new_perm = X_new.copy()
                    X_new_perm[:, k] = np.random.permutation(X_new_perm[:, k])
                    vimp_ls[j] = np.mean((Y_new - fit_obj.predict(X_new_perm)) ** 2) - orig_pred
            #permutation variable importance:
            else:
                raise ValueError('Only Support LOCO and PermutationVIMP, input one of these two notions')
            return vimp_ls, np.argsort(-np.asarray(vimp_ls, dtype = np.float32))
        return ModelAdapter(name = 'xgb_classifier', fit = fit, predict = predict, VIMP = VIMP) 
    #Specify the ridge regression:
    def make_ridge_regression(self):
        def fit(X, y, seed):
            _ = _rng_seed(seed)
            return Ridge(alpha = float(self.ridge_alpha),
                random_state = _rng_seed(seed)).fit(_as_2d_float(X), _as_1d_float(y))
        def predict(fit_obj: Ridge, X_new):
            return np.asarray(fit_obj.predict(_as_2d_float(X_new)), dtype = float)
        def VIMP(fit_obj, X_new, Y_new, VIMP_type = 'LOCO'):
            k = X_new.shape[1]
            orig_pred = (Y_new - fit_obj.predict(X_new)) ** 2
            VIMP_ls = np.zeros(k)
            if VIMP_type == 'LOCO':
                for j in range(k):
                    vimp_ls[j] = np.mean((Y_new - fit_obj.predict(X_new[:, np.setdiff1d(np.arange(k), j)])) ** 2) - MSE_orig
            elif VIMP_type == 'permutation':
                for j in range(k):
                    X_new_perm = X_new.copy()
                    X_new_perm[:, k] = np.random.permutation(X_new_perm[:, k])
                    vimp_ls[j] = MSE_orig - np.mean((Y_new - fit_obj.predict(X_new_perm)) ** 2) - MSE_orig
            #permutation variable importance:
            else:
                raise ValueError('Only Support LOCO and PermutationVIMP, input one of these two notions')
            return vimp_ls, np.argsort(-np.asarray(vimp_ls, dtype = np.float32))
        return ModelAdapter(name = 'ridge_regressor', fit = fit, predict = predict, VIMP = VIMP)
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
        #Here we also leverage the Brier Score here:
        def VIMP(fit_obj, X_new, Y_new, VIMP_type = 'LOCO'):
            k = X_new.shape[1]
            MSE_orig = (Y_new - fit_obj.predict(X_new)) ** 2
            VIMP_ls = np.zeros(k)
            if VIMP_type == 'LOCO':
                for j in range(k):
                    vimp_ls[j] = np.mean((Y_new - fit_obj.predict(X_new[:, np.setdiff1d(np.arange(k), j)])) ** 2) - MSE_orig
            elif VIMP_type == 'permutation':
                for j in range(k):
                    X_new_perm = X_new.copy()
                    X_new_perm[:, k] = np.random.permutation(X_new_perm[:, k])
                    vimp_ls[j] = np.mean((Y_new - fit_obj.predict(X_new_perm)) ** 2) - MSE_orig
            #permutation variable importance:
            else:
                raise ValueError('Only Support LOCO and PermutationVIMP, input one of these two notions')
            return vimp_ls, np.argsort(-np.asarray(vimp_ls, dtype = np.float32))
        return ModelAdapter(name = 'ridge_classifier', fit = fit, predict = predict, VIMP = VIMP)
    #How many levels in the response herem,the binary classification procedure can be leveraged the Brier Score!        
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
        #Here we also leverage the Brier Score here:
        def VIMP(fit_obj, X_new, Y_new, VIMP_type = 'LOCO'):
            k = X_new.shape[1]
            orig_pred = (Y_new - fit_obj.predict(X_new)) ** 2
            VIMP_ls = np.zeros(k)
            if VIMP_type == 'LOCO':
                for j in range(k):
                    VIMP_ls[j] = np.mean((Y_new - fit_obj.predict(X_new[:, np.setdiff1d(np.arange(k), j)])) ** 2) - orig_pred
            elif VIMP_type == 'permutation':
                for j in range(k):
                    X_new_perm = X_new.copy()
                    X_new_perm[:, k] = np.random.permutation(X_new_perm[:, k])
                    VIMP_ls[j] = np.mean((Y_new - fit_obj.predict(X_new[:, np.setdiff1d(np.arange(k), j)])) ** 2) - orig_pred
            #permutation variable importance:
            else:
                raise ValueError('Only Support LOCO and PermutationVIMP, input one of these two notions')
            return VIMP_ls, np.argsort(-np.asarray(VIMP_ls, dtype = np.float32))
        return ModelAdapter(name = 'logistic_classifier', fit = fit, predict = predict, VIMP = VIMP)
    def make_multinomial_classifier(self):
        max_iter = self.maxit
        def fit(X, y, seed = None):
            s = _rng_seed(seed)
            return LogisticRegression(
                multi_class = 'multinomial',
                solver = 'lbfgs', max_iter = max_iter
            ).fit(_as_2d_float(X), np.asarray(y).ravel())
        def predict(fit_obj, X_new, returnL2 = True):
            probs = np.asarray(fit_obj.predict_proba(_as_2d_float(X_new)),
                dtype = float)
            class_outputs = np.argmax(probs, axis = 1) #Also output the MSE here for the later evaluations.
            n_classes = probs.shape[1]
            probs_outputs = np.zeros([X_new.shape[0], n_classes])
            probs_outputs[:, class_outputs] = 1
            #classes_outputs = fit_obj.classes_[np.argmax(probs, axis = 1)].ravel()
            #Return the L2-loss for the prediction here:
            if returnL2:
                return np.mean((probs - probs_outputs) ** 2)
            else:
                return probs#output all of the probabilities here. -> as you return the 2d dimensions right, the VIMP should be adapted.
        def VIMP(fit_obj, X_new, Y_new, VIMP_type = 'LOCO'):
            k = X_new.shape[1]
            n_classes = len(np.asarray(fit_obj.classes_))
            gt_output =  np.zeros([X_new.shape[0], n_classes])
            probs_orig = fit_obj.predict(X_new, returnL2 = False)
            gt_output[:, Y_new] = 1
            orig_Brier = np.mean((gt_output - probs_orig) ** 2)
            VIMP_ls = np.zeros(k)
            if VIMP_type == 'LOCO':
                for j in range(k):
                    probs_new = fit_obj.predict(X_new[:, np.setdiff1d(np.arange(k), j)], returnL2 = False)
                    VIMP_ls[j] = np.mean((gt_output - probs_new) ** 2) - orig_Brier
            else:
                for j in range(k):
                    X_new_copy = X_new.copy()
                    X_new_copy[:, j] = np.random.permutation(X_new_copy[:, j])
                    probs_new = fit_obj.predict(X_new_copy, returnL2 = False)
                    VIMP_ls[j] = np.mean((gt_output - probs_new) ** 2) - orig_Brier
            return VIMP_ls, np.argsort(-np.asarray(VIMP_ls, dtype = np.float32))
        return ModelAdapter(name = 'multinomial_classifier', fit = fit, predict = predict, VIMP = VIMP)
    #making the linear model:
    def make_lm_regression(self):
        def fit(X, y, seed):
            s = _rng_seed(seed)
            return LinearRegression().fit(np.asarray(X, dtype = float), _as_1d_float(y))
        def predict(fit_obj, X_new):
            return np.asarray(fit_obj.predict(_as_2d_float(X_new)),
              dtype = float).ravel()
        def VIMP(fit_obj, X_new, Y_new, VIMP_type = 'LOCO'):
            k = X_new.shape[1]
            MSE_orig = np.mean((Y - fit_obj(X_new)) ** 2)
            VIMP_ls = np.zeros(k)
            if VIMP_type == 'LOCO':
                for j in range(k):
                    VIMP_ls[j] = np.mean((Y_new - fit_obj.predict(X_new[:, np.setdiff1d(np.arange(k), j)]))**2) - MSE_orig
            else:
                for j in range(k):
                    X_new_copy = X_new.copy()
                    X_new_copy[:, j] = np.random.permutation(X_new_copy[:, j])
                    probs_new = fit_obj.predict(X_new_copy)
                    VIMP_ls[j] = np.mean((Y_new - probs_news) ** 2) - MSE_orig
        return ModelAdapter(name = 'linear_regressor', fit = fit, predict = predict, VIMP = VIMP)
    def make_mlp_regression(self):
        def fit(X: Any, y: Any, seed):
            s = _rng_seed(seed)
            Xs, center, scale = _standardize_cols(_as_2d_float(X))
            coef_lim = None
            m = MLPRegressor(
                hidden_layer_sizes = (int(self.mlp_hidden_size), ),
                activation = self.activation,
                alpha = float(self.mlp_decay),
                max_iter = int(self.maxit),
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
        def VIMP(fit_obj, X_new, Y_new, VIMP_type = 'LOCO'):
            raise NotImplementedError
        return ModelAdapter(name = 'mlp_regressor', fit = fit, predict = predict, VIMP = VIMP)
    #How many classesyou have. 
    def make_mlp_classifier(self):
        positive_class = self.positive_class
        def fit(X, y, seed):
            s = _rng_seed(seed)
            y_raw = np.asarray(y).ravel()
            classes = np.unique(y_raw)
            if classes.size != 2:
                raise ValueError("MLP Classifier Currently Support Binary Labels only.")
            Xs, center, scale = _standardize_cols(_as_2d_float(X))
            bin_idx = (y_raw == positive_class).astype(int)
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
        def VIMP(fit_obj, X_new, Y_new, VIMP_type = 'LOCO'):
            raise NotImplementedError
        return ModelAdapter(name = 'mlp_classifier', fit = fit, predict = predict, VIMP = VIMP)
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
        def VIMP(fit_obj, X_new, Y_new, VIMP_type = 'LOCO'):
            raise NotImplementedError
        return ModelAdapter(name = 'mlp_multi_classifier', fit = fit, predict = predict, VIMP = VIMP)
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
            'xgb_multiclassifier': self.make_xgb_multiclassifier(),
            'ridge_regressor': self.make_ridge_regression(),
            'ridge_classifier': self.make_ridge_classification(),
            'mlp_regressor': self.make_mlp_regression(),
            'mlp_classifier': self.make_mlp_classifier(),
            'mlp_multi_classifier': self.make_mlp_multi_classifier()
        }
    def as_r_style_dict(self):
        mapping = self.adapters()
        registry_map = {
          k: {'name': v.name, 'fit': v.fit, 'predict': v.predict, 'vimp': v.VIMP}
          for k, v in mapping.items()
          }
        return registry_map



