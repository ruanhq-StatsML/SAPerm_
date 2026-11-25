import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.special import kl_div
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler




"""
#R-learner weighted regression:
"""
def _fit_tau_rlearner_weighted(X, Y_tilde, W_tilde, seed: int = 0, clip_wtilde: float = 1e-3):
    X = np.asarray(X)
    Y_tilde = np.asarray(Y_tilde).reshape(-1)
    W_tilde = np.asarray(W_tilde).reshape(-1)
    mask = (np.abs(W_tilde) > clip_wtilde)
    z = Y_tilde[mask]/W_tilde[mask]
    weights = (W_tilde[mask] ** 2)
    tau_model = Pipeline(
      steps = [
      ('scaler', StandardScaler(with_mean = True, with_std = True)),
      ('ridge', Ridge(alpha = 1.0, random_state = seed))])
    tau_model.fit(X[mask], z, ridge__sample_weight = weights)
    return tau_model

'''
Making the folds here via the array_split:
'''
def make_folds(n, n_folds = 5, seed = 2026):
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    return folds


'''
Cross-fitting for the outcome model and the propensity score model:
'''
def cross_nuisance_fit(X, Y, W, n_folds = 5,
  n_estimators = 100, binary_outcome = True,
  model_registry = MODEL_REGISTRY,
  model_m = 'rf_regressor', model_e = 'rf_classifier',
  clip_e = 1e-3, seed = 2026):
    X = np.asarray(X)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    folds = make_folds(n, n_folds = n_folds, seed = seed)
    mu_hat = np.zeros(n, dtype = float)
    e_hat = np.zeros(n, dtype = float)
    for k, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train = Y[train_idx]
        W_train = W[train_idx]
        model_propensity_score = model_registry[model_e]
        model_outcome = model_registry[model_m]
        #Fit the outcome model and propensity score models:
        fit_mu = model_outcome['fit'](
          X_train, Y_train, seed = seed + k
          )
        mu_hat[test_idx] = model_outcome['predict'](
          fit_obj = fit_mu, X_new = X_test
          )
        fit_e = model_propensity_score['fit'](
          X_train, W_train, seed = seed + k * 3
          )
        e_hat[test_idx] = model_outcome['predict'](
          fit_obj = fit_e, X_new = X_test
          )
    e_hat = np.clip(e_hat, clip_e, 1 - clip_e)
    return mu_hat, e_hat

'''
Permutation Test for Distribution Shift via R-risk followed by the permute-then-refit procedure:
Hyperparameters:
- X:               Covariates array - (n_exist + n_new, p)
- Y:               Responses - (n_exist + n_new, )
- W:               Treatment(Batch) Assignment - (n_exist + n_new, )
- seed:            Random State Seeds
- n_split:         Number of splits
- clip_e:          Clipping value for the propensity score
- n_perm:          Number of permutations for the test
- alpha:           Significance Level for the test
- model_m:         Outcome Model
- model_e:         Propensity Score Model
Returns the cross-fitted value (m_hat, e_hat) evaluated on the whole data
as well as the fitted outcome model and propensity score models.
'''
#Specify the model registry factory so that we can fetch the model with flexibility afterwards:

def RRPerm(
    X: np.ndarray, Y: np.ndarray, W: np.ndarray, *,
    model_registry = MODEL_REGISTRY,
    model_m = 'rf_regressor', model_e = 'rf_classifier',
    seed: int = 0, n_folds: int = 5, binary_outcome = False,
    clip_e = 0.01, n_perm = 150, alpha = 0.05, return_detail = True):
    X = np.asarray(X)
    Y = _as_1d(Y)
    W = _as_1d(W).astype(int)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    #---------------------------------------------------------------------
    # Cross-fitted nuisance estimation, via the pseudo-outcome regression
    #---------------------------------------------------------------------
    folds = make_folds(n, n_folds = n_folds, seed = seed)
    mu_hat = np.zeros(n, dtype = float)
    e_hat  = np.zeros(n, dtype = float)
    mu_hat, e_hat = cross_nuisance_fit(
      X, Y, W, n_folds = n_folds, model_registry = model_registry,
      model_m = model_m, model_e = model_e
    )
    #---------------------------------------------------------------------
    # Calculate the R-risk followed by the weighted regression
    #---------------------------------------------------------------------
    Y_tilde = Y - mu_hat
    W_tilde = W - e_hat
    tau_model = _fit_tau_rlearner_weighted(X, Y_tilde, W_tilde, seed = seed)
    tau_hat = tau_model.predict(X)
    observed_r_risk = np.mean((Y_tilde - tau_hat * W_tilde) ** 2)
    #---------------------------------------------------------------------
    # Permute-then-refit for re-estimation of the R-risk
    #---------------------------------------------------------------------
    permuted_r_risk = np.zeros(n_perm, dtype = float)
    for b in range(n_perm):
        W_perm = rng.permutation(W)
        #Refit the model:
        mu_hat_b, e_hat_b = cross_nuisance_fit(
          X, Y, W_perm, n_folds = n_folds, model_registry = model_registry,
          model_m = model_m, model_e = model_e
          )
        Y_tilde_b = Y - mu_hat_b
        W_tilde_b = W - e_hat_b
        tau_model_b = _fit_tau_rlearner_weighted(X, Y_tilde_b, W_tilde_b, seed = seed)
        tau_hat_b = tau_model_b.predict(X)
        permuted_r_risk[b] = np.mean((Y_tilde_b - tau_hat_b * W_tilde_b) ** 2)
    p_value = (1.0 + np.sum(permuted_r_risk >= observed_r_risk))/(1.0 + n_perm)
    out = {
      'statistic': float(observed_r_risk),
      'p_value': float(p_value),
      'reject': np.bool_(p_value < alpha),
      'alpha': float(alpha),
      'model_outcome': model_outcome['name'],
      'model_propensity_score': model_propensity_score['name']
    }
    return out



#Test cases: H_{0}
#rng = np.random.default_rng(2023)
#n = 400
#p = 16
#X = rng.normal(size = (n, p))
#Y = 10.0 * X[:, 0] + 2.0 * X[:, 1] + 1.0 * X[:, 2] + rng.normal(scale = 1.0, size = n)
#W = rng.binomial(1, 0.5, size = n)
#output = RRPerm(X, Y, W, n_folds = 5,
# model_registry = MODEL_REGISTRY, model_m = 'rf_regressor', model_e = 'rf_classifier',
# n_perm = 5)
#output



