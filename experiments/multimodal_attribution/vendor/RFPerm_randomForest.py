#RFPerm Software Packages:
import os
import statistics
import sklearn
import numpy as np
import pandas as pd
import xgboost
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from typing import Iterable, Optional, Tuple, Union
from scipy.stats import multivariate_normal
from model_registry_class import ModelRegistry

#Generalize this to a general purposed permutation test where the mean of the validation error and the
#prediction error are compared.
def LM_generation(n, beta_hat, cor, n_nuisance, eps, mean_X = 0, var_X=1):
    beta_hat = np.asarray(beta_hat)
    p = len(beta_hat)
    corr_matrix = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            corr_matrix[i, j] = (cor ** abs(i - j)) * var_X
    X_design = multivariate_normal.rvs(
        mean=[mean_X] * p,
        cov=corr_matrix,
        size=n
    )
    Y1 = X_design @ beta_hat
    random_error = np.random.normal(0, eps, n)
    X_nuiss = np.random.normal(0, 1, size=(n, n_nuisance))
    Y = Y1 + random_error
    df_return = pd.DataFrame(
        np.column_stack([Y1, X_design, X_nuiss, Y])
    )
    colnames = (
        ["Y1"] +
        [f"X{i+1}" for i in range(p)] +
        [f"X_nuis{i+1}" for i in range(n_nuisance)] +
        ["Y"]
    )
    df_return.columns = colnames
    X_return = df_return.iloc[:, 1:-1]
    return {
        "df_return": df_return,
        "X_return": X_return
    }




def permTest(mse_list_val, mse_list_pred, n_perm = 5000, seed = 2026):
    '''
    Permutation test for one-sided or two-sided p-values to compare the mean of the 
    two sets of MSE
    '''
    rng = np.random.default_rng(seed)
    val = np.asarray(mse_list_val, dtype = float)
    pred = np.asarray(mse_list_pred, dtype = float)
    mean_pred = float(np.mean(val))
    mean_exist = float(np.mean(pred))
    mean_dif_original = mean_pred - mean_exist
    n_val = int(val.shape[0])
    n_pred = int(pred.shape[0])
    n = int(n_val + n_pred)
    gen = np.random.default_rng(seed = seed) if seed else np.random.default_rng()
    mean_dif_seq = np.zeros(n_perm)
    pooled_mse = np.empty(n, dtype = float)
    pooled_mse[:n_val] = val
    pooled_mse[n_val:] = pred
    pooled = pooled_mse.copy()
    for i in range(n_perm):
        pooled_ = rng.permutation(pooled)
        mean_dif_seq = np.mean(pooled[n_val:]) - np.mean(pooled[:n_val])
    #One-sided p-value:
    p_val_onesided = (1 + np.sum(mean_dif_original > mean_dif_seq))/(n_perm + 1)
    #Two-sided p-value:
    p_val_twosided = (1 + np.sum(np.abs(mean_dif_original) > np.abs(mean_dif_seq)))/(n_perm + 1)
    return p_val_onesided, p_val_twosided


def infer_response_type(Y):
    """
    Infer the types of response in Y:
    Y should either be Numeric or Object
    Return one of the "binary", "continuous" for the downstream modeling procedure.
    """
    if Y is None:
        raise ValueError("Y must not be None")
    Y_ = np.array(Y).reshape(-1)
    if Y_.size == 0:
        raise ValueError("Y must contain at least one observation")
    if pd.isna(Y_).any():
        raise ValueError("Y contains missing values. Please remove or impute them first.")
    dtype = Y_.dtype
    #Checking whether it's object or numeric, otherwise, ValueError
    is_numeric = np.issubdtype(dtype, np.number)
    is_object_like = dtype == object or np.issubdtype(dtype, np.str_)
    if not (is_numeric or is_object_like):
        raise TypeError(
            'Y must be numeric or categorical/object-like.'
            f"Got dtype={dtype}."
        )
    len_class = np.unique(Y_).size
    if len_class == 2:
        return 'binary'
    elif len_class > 6:
        return 'continuous'
    else:
        return 'multinomial'
    #checking the forms and conditions of this:


def RFPerm(df_exist, df_new, loss,
           alpha = 0.05, B = 200,
           n_perm = 5000, seed = 2026):
    """
    Random Forest Permutation Test for Distribution Shift
    Random Forest has nice properties of OOB(out-of-bag) error
    The inner permutation test supplies one-sided p-value here.
    Parameters:
    - df_exist:         Existing Batch of data, (n_exist, p), dataframe
    - df_new:           New Batch of data, (n_new, p), dataframe
    - B:                Number of repeatances for the power evaluation
    - loss:             Notion of the loss function
    - alpha:            Significance Level
    - n_perm:           Number of the permutation
    - seed:             Random Seed
    Return:
    - p_value:          Results of Significance.
    """
    X_exist = df_exist.iloc[:, :-1]
    Y_exist = df_exist.iloc[:, -1]
    X_new = df_new.iloc[:, :-1]
    Y_new = df_new.iloc[:, -1]
    n_exist, p = X_exist.shape
    n_new = X_new.shape[0]
    response_type = infer_response_type(Y_exist)
    pval_list = np.zeros(B)
    for i in range(B):
        df_exist_B = df_exist.sample(frac = 1, random_state = seed + i).reset_index(drop = True)
        df_new_B = df_new.sample(frac = 1, random_state = seed + i).reset_index(drop = True)
        X_exist_B = df_exist_B.iloc[:, :-1]
        Y_exist_B = df_exist_B.iloc[:, -1]
        X_new_B = df_new_B.iloc[:, :-1]
        Y_new_B = df_new_B.iloc[:, -1]
        if response_type == 'continuous':
            pipeline = Pipeline(
                [('scaler', StandardScaler()),
                  ('RF', RandomForestRegressor(
                        min_samples_leaf = round(np.sqrt(n_exist)/2),
                        max_features = 0.33,
                        n_estimators = 150,
                        oob_score = True,
                        bootstrap = True,
                        random_state = seed
                   ))]
                )
            fitted_model = pipeline.fit(X_exist_B, Y_exist_B)
            rf_model = fitted_model.named_steps['RF']
            #_, rf_model = fitted_model.name_steps['RF']
            Y_hat_oob = np.asarray(rf_model.oob_prediction_, dtype = float)
            Y_hat_test = pipeline.predict(X_new_B)
        elif response_type == 'binary':
            pipeline = Pipeline(
                [('scaler', StandardScaler()),
                  ('RF', RandomForestClassifier(
                        min_samples_leaf = round(np.sqrt(n_exist)/2),
                        max_features = round(np.sqrt(p)),
                        n_estimators = 150,
                        oob_score = True,
                        bootstrap = True,
                        random_state = seed + 1
                    ))]
                )
            fitted_model = pipeline.fit(X_exist_B, Y_exist_B)
            rf_model = fitted_model.named_steps['RF']
            Y_hat_oob = rf_model.oob_decision_function_[:, 1]
            Y_hat_test = pipeline.predict(X_new_B)
        else:
            pipeline = Pipeline(
                [('scaler', StandardScaler()),
                  ('RF', RandomForestClassifier(
                        min_samples_leaf = round(np.sqrt(n_exist)/2),
                        max_features = round(np.sqrt(p)),
                        n_estimators = 150,
                        oob_score = True,
                        bootstrap = True,
                        random_state = seed + 1
                    ))]
                )
            fitted_model = pipeline.fit(X_exist_B, Y_exist_B)
            rf_model = fitted_model.named_steps['RF']
            Y_hat_oob = rf_model.oob_decision_function_
            Y_hat_test = pipeline.predict_proba(X_new_B)
        p_one, _ = permTest(
            loss(Y_hat_oob, Y_exist_B),
            loss(Y_hat_test, Y_new_B),
            n_perm = n_perm
        )
        pval_list[i] = p_one
    power = float(np.mean([1 if float(p) < alpha else 0 for p in pval_list]))
    return power

'''
Bidirectional Adjustment for the permutation test:
- 
'''
def bidir(df_exist, df_new, method = 'stouffer'):
    p_value_for = RFPerm(df_exist, df_new, method = method)
    p_value_bac = RFPerm(df_exist, df_new, method = method)
    p_value_bidir = combine_pvalues([p_value_for, p_value_bac], method = method)
    return p_value_bidir
