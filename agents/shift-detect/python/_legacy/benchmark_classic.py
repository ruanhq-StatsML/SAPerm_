"""Importable classic two-sample tests (cleaned from benchmark_method_tst)."""
#

import numpy as np
import pandas as pd
from scipy.stats import f, norm, multivariate_normal, t
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, chi2_kernel
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
PI = 3.1415926


'''
Permutation P-value for the Classifier Two-Sample Test:
"Revisiting Classifier Two-Sample Tests"
Lopez-Paz, Oquab 2016
'''
def c2st(X, y, test_size = 0.2, B=150):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).reshape(-1)
    p = X.shape[1]
    clf = RandomForestClassifier(
        max_features=int(max(1, np.round(np.sqrt(p)))),
        max_depth=4,
        random_state=0,
    )
    hi = max(int(B) ** 2, int(B) + 1)
    rs = int(2026 + np.random.randint(low=int(B), high=hi))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rs
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = float(np.mean(y_pred == y_test))
    exceed = 0
    for _ in range(B):
        y_random = np.random.permutation(y_train)
        clf.fit(X_train, y_random)
        y_pred_random = clf.predict(X_test)
        perm_acc = float(np.mean(y_pred_random == y_test))
        if perm_acc >= accuracy:
            exceed += 1
    return float((exceed + 1.0) / (float(B) + 1.0))

#df1 = np.random.random((200, 20))
#df2 = np.random.random((500, 20))
#X = np.vstack((df1, df2))
#y = np.concatenate([np.zeros(df1.shape[0]), np.ones(df2.shape[0])])
#c2st(X, y)




"""
"A Kernel Two-sample Test"
Gretton et.al. 2012
"""
def kernel_MMD(X, Y, kernel = 'rbf', n_iter = 100, gamma = 1.0):
    n1, p = X.shape
    n2 = Y.shape[0]
    def compute_MMD(X, Y, kernel):
        if kernel == 'rbf':
            AA = rbf_kernel(X, X, gamma = gamma)
            BB = rbf_kernel(Y, Y, gamma = gamma)
            AB = rbf_kernel(X, Y, gamma = gamma)
        elif kernel == 'linear':
            AA = linear_kernel(X, X, gamma = gamma)
            BB = linear_kernel(Y, Y, gamma = gamma)
            AB = linear_kernel(X, Y, gamma = gamma)
        np.fill_diagonal(AA, 0)
        np.fill_diagonal(BB, 0)
        return np.sum(AA)/(n1 * (n1 - 1)) + np.sum(BB)/(n2 * (n2 - 1)) - 2 * np.sum(AB)
    observed_mmd = compute_MMD(X, Y, kernel = kernel)
    perm_mmd_seq = np.zeros(n_iter).reshape(-1)
    concat = np.vstack([X, Y])
    for i in range(n_iter):
        perm = np.random.permutation(n1 + n2)
        X_perm = concat[perm[:n1], :]
        Y_perm = concat[perm[n1:], :]
        perm_mmd_seq[i] = compute_MMD(X_perm, Y_perm, kernel = kernel)
    p_value = (1 + np.sum(perm_mmd_seq > observed_mmd))/(1 + n_iter)
    return p_value

#df1 = np.random.random((500, 20))
#df2 = np.random.random((200, 20))
#kernel_MMD(df1, df2, n_iter = 100)


"""
"A more powerful two-sample test in high dimensions using random projection"
ME.Lopes et.al. 2011
"""
def random_projection_test(X1, X2, k = None):
    n1, p = X1.shape
    n2 = X2.shape[0]
    n = (n1 + n2 - 1)
    if k is None:
        k = int(np.floor(np.sqrt(n)))
    Phi_proj = np.random.randn(k, p)
    X1_proj = X1 @ Phi_proj.T
    X2_proj = X2 @ Phi_proj.T
    mu1 = np.mean(X1_proj, axis = 0)
    mu2 = np.mean(X2_proj, axis = 0)
    S1 = np.cov(X1_proj, rowvar = False)
    S2 = np.cov(X2_proj, rowvar = False)
    S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2)/(n1 + n2 - 2)
    diff = mu1 - mu2
    T_square = (diff.T @ np.linalg.inv(S_pooled) @ diff) * (n1 * n2)/(n1 + n2)
    f_stat = T_square * (n - k + 1)/(n * k)
    p_value = 1 - f.cdf(f_stat, k, n - k + 1)
    return T_square, p_value

#df1 = np.random.random((500, 20))
#df2 = np.random.random((200, 20))
#_, p_val = random_projection_test(df1, df2)


"""
An adaptive two-sample test for high-dimensional means
Xu 2016
"""

def adaptive_test_high_dim(X1, X2, gammas=None, n_perm=150):
    if gammas is None:
        gammas = [1, 2, 4, 8, np.inf]
    n1, p = X1.shape
    n2, _ = X2.shape
    diff = np.mean(X1, axis=0) - np.mean(X2, axis=0)
    def calculate_spu(d, g):
        if g == np.inf:
            return np.max(np.abs(d))
        return np.sum(np.abs(d) ** g)
    obs_stats = [calculate_spu(diff, g) for g in gammas]
    combined = np.vstack([X1, X2])
    perm_stats = np.zeros((n_perm, len(gammas)))
    for i in range(n_perm):
        perm_idx = np.random.permutation(n1 + n2)
        p_X1 = combined[perm_idx[:n1]]
        p_X2 = combined[perm_idx[n1:]]
        p_diff = np.mean(p_X1, axis=0) - np.mean(p_X2, axis=0)
        for j, g in enumerate(gammas):
            perm_stats[i, j] = calculate_spu(p_diff, g)
    p_vals = np.mean(perm_stats > obs_stats, axis=0)
    null_pval = np.zeros((n_perm, len(gammas)))
    for j in range(len(gammas)):
        ranks = np.argsort(np.argsort(-perm_stats[:, j])) / n_perm
        null_pval[:, j] = ranks
    min_p_null = np.min(null_pval, axis=1)
    adaptive_p = np.mean(min_p_null <= np.min(p_vals))
    return adaptive_p, dict(zip(gammas, p_vals))

#np.random.seed(2026)
#df1 = np.random.random((500, 20))
#df2 = np.random.random((200, 20))
#p_val, zip_p = adaptive_test_high_dim(df1, df2)#just use the first p_val



'''
Chen, Qin 2010
'''
def chen_qin(X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    n1, p = X.shape
    n2, _ = Y.shape
    def get_U_components(data1, data2=None):
        if data2 is None:
            n = data1.shape[0]
            dots = data1 @ data1.T
            np.fill_diagonal(dots, 0)
            return np.sum(dots) / (n * (n - 1))
        return np.sum(data1 @ data2.T) / (data1.shape[0] * data2.shape[0])
    stat = get_U_components(X) + get_U_components(Y) - get_U_components(X, Y)
    combined = np.vstack([X, Y])
    S = np.cov(combined, rowvar=False)
    tr_sigma2_est = np.sum(S**2) - (1 / (n1 + n2)) * np.sum(np.diag(S) ** 2)
    var_stat = (2 / (n1 * (n1 - 1)) + 2 / (n2 * (n2 - 1)) + 4 / (n1 * n2)) * tr_sigma2_est
    z_score = stat / np.sqrt(max(var_stat, 1e-10))
    p_value = 1 - norm.cdf(z_score)
    return p_value

#np.random.seed(2026)
#df1 = multivariate_normal.rvs(mean = np.zeros(10), cov = np.random.random((10,10))+ np.diag(np.ones(10))*10, size= 500)
#df2 = multivariate_normal.rvs(mean = np.zeros(10), cov = np.random.random((10,10))+ np.diag(np.ones(10))*10, size= 200)
#chen_qin(df1, df2)




'''
Chen, Qin 2014: 
High Dimensional Testing for the Equivalence of the Covariance Structure.
X[None, :] - X[:, None]
'''
#and that push has accelerated as the network has grown more complex.
#sorted -> stressed -> closed
#X[None, :] - X[:, None]#p_value = 1 - norm.cdf(z_score)
#function for extracting the maximum threshold(different covariance)
def max_threshold_diffcov(X, Y, sig_mat1, sig_mat2, clip_val = 1e-5):
    n1, p = X.shape
    n2 = Y.shape[0]
    a_f = np.sqrt(2 * np.log(np.log(p)))
    b_f = 2 * np.log(np.log(p)) + 0.5 * np.log(np.log(np.log(p))) - 0.5 * np.log(4 * PI/(1-0.05)**2)
    eta = 0.05
    diag1 = np.diag(sig_mat1)
    diag1 = np.maximum(diag1, clip_val)
    diag2 = np.diag(sig_mat2)
    diag2 = np.maximum(diag2, clip_val)
    T_orig = (np.mean(X, dim = 0) - np.mean(Y, dim = 0)) ** 2/(diag1/n1 + diag2/n2)
    s_level = T_orig[((T_orig >= 0.01) & (T_orig <= 2 * (1 - eta) * np.log(p)))]
    s_m = s_level[:, np.newaxis]
    t_val = T_orig[np.newaxis, :]
    t_m = np.where(t_vals >= s_m, t_val - 1, 0)
    #dnorm -> cdf and pnorm -> pdf
    sqrt_s = np.sqrt(s_level)
    pdf_s = norm.pdf(sqrt_s)
    cdf_s = norm.cdf(sqrt_s)
    mean_thr = 2 * np.sqrt(s_level) * pdf_s * p
    thr = T_m.sum(dim = 1)
    mean_thr = 2 * sqrt_s * pdf_s * p
    #calculate the s.d. of the threshold:
    term1 = 2 * (sqrt_s ** 3 + sqrt_s) * pdf_s + 4 * (1 - cdf_s)
    sd_thr = np.sqrt(p * term1 - mean_thr ** 2/p)
    max_threshold = np.max((thr - mean_thr)/(sd_thr + clip_val))
    max_threshold = max_threshold * a_f - b_f
    return max_threshold

def stat_chen14(X, Y):
    cov_X = np.cov(X, rowvar = False)
    cov_Y = np.cov(Y, rowvar = False)
    test_stat = max_threshold_diffcov(X, Y, cov_X, cov_Y)
    return test_stat



def best_band(X, bandwidth, cv_fold):
    n, p = X.shape
    fold_size = np.round(n/cv_fold)
    n_bandwidth = len(bandwidth)
    diff_norm = np.zeros((cv_fold, n_bandwidth))
    kf = KFold(n_splits = cv_fold, shuffle = True)
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train = X[-test_idx, :]
        X_test = X[test_idx, :]
        for j in range(n_bandwidth):
            train_cov = np.cov(X_train, rowvar = False)
            test_cov = np.cov(X_test, rowvar = False)
            for j, bw in enumerate(bandwidth):
                #similar for toeplitz correlation structure.
                train_cov_band = np.tril(np.triu(train_cov, k = -bw), k = bw)
                diff_norm[i, j] = np.linalg.norm(test_cov - train_cov_band, ord = 'fro')
    diff_norm = diff_norm.mean(axis = 0)
    best = bandwidth[np.argmin(diff_norm)]
    return best

"""
Implementation of the Chen14 test:
Chen, Li 14
"Two-Sample Tests for High Dimensional Means with Thresholding and Data Transformation.
"""
def chen_li14(X, Y, n_perm, clip_val = 1e-3):
    X_cov = np.cov(X, rowvar = False)
    Y_cov = np.cov(Y, rowvar = False)
    p = X.shape[1]
    if p % 2 == 1:
        bandwidth1 = np.linspace(0, p-1, (p//2+1))
        bandwidth2 = np.linspace(0, p-1, (p//2+1))
    else:
        bandwidth1 = np.linspace(0, p, (p//2+1))
        bandwidth2 = np.linspace(0, p, (p//2+1))
    optim_band1 = best_band(X, bandwidth1, cv_fold = 5)
    #Build the banded toeplitz matrix:
    X_cov_trunc = np.tril(np.triu(X_cov, k = -optim_band1), k = optim_band1)
    optim_band2 = best_band(Y, bandwidth2, cv_fold = 5)    
    Y_cov_trunc = np.tril(np.triu(Y_cov, k = -optim_band2), k = optim_band2)
    eig_val, eig_vec = np.linalg.eigh(X_cov_trunc)
    eig_val = np.maximum(eig_val, clip_val)
    X_cov_est = eig_vec.T @ np.diag(eig_val) @ eig_vec
    eig_val, eig_vec = np.linalg.eigh(Y_cov_trunc)
    eig_val = np.maximum(eig_val, clip_val)
    Y_cov_est = eig_vec.T @ np.diag(eig_val) @ eig_vec
    #Then conduct the testing procedure:
    T_CLZ_0 = stat_chen14(X, Y)
    T_CLZ_resam = np.zeros(n_perm)
    for b in range(n_perm):
        X_B = multivariate_normal.rvs(mean = np.zeros(p), cov = X_cov_est, size = n1)
        Y_B = multivariate_normal.rvs(mean = np.zeros(p), cov = Y_cov_est, size = n2)
        T_CLZ_resam[b] = stat_chen14(X_B, Y_B)
    p_CLZ = (np.sum(T_CLZ_resam > T_CLZ_0) + 1)/(n_perm + 1)
    return p_CLZ, optim_band1, optim_band2


#df1 = multivariate_normal.rvs(mean = np.zeros(25), cov = np.random.random((25,25))+ np.diag(np.ones(25))*25, size= 500)
#df2 = multivariate_normal.rvs(mean = np.zeros(25), cov = np.random.random((25,25))+ np.diag(np.ones(25))*25, size= 500)
#sig_mat1 = np.cov(df1, rowvar = False)
#sig_mat2 = np.cov(df2, rowvar = False)
#chen_li14(df1, df2, n_perm = 100)


def max_threshold_diffcov(X, Y, sig_mat1, sig_mat2, clip_val = 1e-5):
    n1, p = X.shape
    n2 = Y.shape[0]
    a_f = np.sqrt(2 * np.log(np.log(p)))
    b_f = 2 * np.log(np.log(p)) + 0.5 * np.log(np.log(np.log(p))) - 0.5 * np.log(4 * PI/(1-0.05)**2)
    eta = 0.05
    diag1 = np.diag(sig_mat1)
    diag1 = np.maximum(diag1, clip_val)
    diag2 = np.diag(sig_mat2)
    diag2 = np.maximum(diag2, clip_val)
    T_orig = (X.mean(axis = 0) - Y.mean(axis = 0)) ** 2/(diag1/n1 + diag2/n2)
    s_level = T_orig[(T_orig >= 0.01) & (T_orig <= 2 * (1 - eta) * np.log(p))]
    T_m = (T_orig - 1) * (T_orig >= s_level[:, None])
    thr = np.sum(T_m, axis = 1)
    cdf_s = norm.cdf(np.sqrt(s_level))
    pdf_s = norm.pdf(np.sqrt(s_level))
    mean_thr = 2 * np.sqrt(s_level) * norm.pdf(np.sqrt(s_level)) * p
    term1 = 2 * ((np.sqrt(s_level)) ** 3 + np.sqrt(s_level)) * norm.pdf(np.sqrt(s_level))
    term2 = 4 - 4 * norm.cdf(np.sqrt(s_level))
    sd_thr = np.sqrt(p * (term1 + term2) - (mean_thr ** 2/p))
    max_threshold = np.max((thr - mean_thr)/(sd_thr + clip_val)) * a_f - b_f
    return max_threshold

#df1 = multivariate_normal.rvs(mean = np.zeros(10), cov = np.random.random((10,10))+ np.diag(np.ones(10))*10, size= 100)
#df2 = multivariate_normal.rvs(mean = np.zeros(10), cov = np.random.random((10,10))+ np.diag(np.ones(10))*10, size= 100)
#sig_mat1 = np.cov(df1, rowvar = False)
#sig_mat2 = np.cov(df2, rowvar = False)
#max_threshold_diffcov(X = df1, Y = df2, sig_mat1 = sig_mat1, sig_mat2 = sig_mat2)


#function for extracting the maximum threshold(same covariance)
def max_threshold_samecov(X, Y, sig_mat, clip_val = 5e-3):
    n1, p = X.shape
    n2 = Y.shape[0]
    a_f = np.sqrt(2 * np.log(np.log(p)))
    b_f = 2 * np.log(np.log(p)) + 0.5 * np.log(np.log(np.log(p))) - 0.5 * np.log(4 * PI/(1-0.05)**2)
    eta = 0.05
    diag_sig = np.diag(sig_mat)
    diag_sig = np.maximum(diag_sig, clip_val)
    T_orig = (X.mean(axis = 0) - Y.mean(axis = 0)) ** 2/((1/n1 + 1/n2) * diag_sig)
    s_level = T_orig[(T_orig >= 0.01) & (T_orig <= 2 * (1 - eta) * np.log(p))]
    T_m = (T_orig - 1) * (T_orig >= s_level[:, None])
    thr = np.sum(T_m, axis = 1)
    cdf_s = norm.cdf(np.sqrt(s_level))
    pdf_s = norm.pdf(np.sqrt(s_level))
    mean_thr = 2 * np.sqrt(s_level) * norm.pdf(np.sqrt(s_level)) * p
    term1 = 2 * ((np.sqrt(s_level)) ** 3 + np.sqrt(s_level)) * norm.pdf(np.sqrt(s_level))
    term2 = 4 - 4 * norm.cdf(np.sqrt(s_level))
    sd_thr = np.sqrt(p * (term1 + term2) - (mean_thr ** 2/p))
    max_threshold = np.max((thr - mean_thr)/(sd_thr + clip_val)) * a_f - b_f
    return max_threshold

#max_threshold_samecov(X = df1, Y = df2, sig_mat = sig_mat)
#sig_mat = np.random.random((10,10))+ np.diag(np.ones(10))*10
#df1 = multivariate_normal.rvs(mean = np.zeros(10), cov = sig_mat, size= 100)
#df2 = multivariate_normal.rvs(mean = np.zeros(10), cov = sig_mat, size= 100)
#p_value, _, _ = chen_li14(df1, df2, n_perm = 100)




#Python implementation for the SKK test in R:
def skk_test(X, Y):
    n1, p = X.shape
    n2 = Y.shape[0]
    n = n1 + n2 - 2
    X_bar = np.mean(X, axis = 0)
    Y_bar = np.mean(Y, axis = 0)
    S1 = np.cov(X, rowvar = False)
    S2 = np.cov(Y, rowvar = False)
    D1 = np.diag(np.diag(S1))
    D2 = np.diag(np.diag(S2))
    D = (D1/n1) + (D2/n2)
    D_inv = np.linalg.inv(np.sqrt(D))
    S_pooled = (S1/n1 + S2/n2)
    R = D_inv.T @ S_pooled @ D_inv
    R_R = np.diag(R.T @ R)
    C_p_n = 1 + np.sum(R_R/(p**(3/2)))
    var_qn = 2 * np.sum(R_R)/p - 2 * np.sum(np.diag(np.linalg.inv(D) @ S1))**2/p/n1/(n1+1)**2 - 2 * np.sum(np.diag(np.linalg.inv(D) @ S2))**2/p/n2/(n2+1)**2
    denom = np.sqrt(p * C_p_n * var_qn)
    TS_value = ((X_bar - Y_bar).T @ D_inv @ (X_bar - Y_bar) - p)/denom
    p_value = 2 * (1 - norm.cdf(np.abs(TS_value)))
    return TS_value, p_value


#df1 = np.random.random((500, 20))
#df2 = np.random.random((200, 20))
#_, p_value = skk_test(df1, df2)




def sri08_test(X, Y, n_iter = 100, seed = 2026, clip_val = 1e-5):
    n1, p = X.shape
    n2 = Y.shape[0]
    tau = (n1 + n2)/(n1 * n2)
    n = n1 + n2 - 2
    Sn = ((n1 - 1) * np.cov(X, rowvar = False) + (n2 - 1) * np.cov(Y, rowvar = False))/(n1 + n2 - 2)
    diag_s = np.diag(np.diag(Sn))
    diag_s[diag_s < clip_val] = clip_val
    concatenate_df = np.vstack([X, Y])
    diff = np.mean(X, axis = 0) - np.mean(Y, axis = 0)
    XdX = np.sum(diff**2/diag_s)
    sristat = XdX
    sristat_perm = np.ones(n_iter).reshape(-1)
    for i in range(n_iter):
        perm = np.random.permutation(np.arange(n1 + n2))
        X_perm = concatenate_df[perm[:n1], :]
        Y_perm = concatenate_df[perm[n1:], :]
        Sn_perm = ((n1 - 1) * np.cov(X_perm, rowvar = False) +
                   (n2 - 1) * np.cov(Y_perm, rowvar = False))/(n1 + n2 - 2)
        diag_sn_perm = np.diag(np.diag(Sn_perm))
        diag_sn_perm[diag_sn_perm < clip_val] = clip_val
        diff_perm = np.mean(X_perm, axis = 0) - np.mean(Y_perm, axis = 0)
        XdX_mean = np.sum(diff_perm**2/diag_sn_perm)
        sristat_perm[i] = XdX_mean
    #record the permutation p-value:
    p_value = (np.sum(sristat_perm >= sristat) + 1)/(n_iter + 1)
    return p_value

#df1 = np.random.random((500, 20))
#df2 = np.random.random((200, 20))
#p_value = sri08_test(df1, df2, n_iter = 50)


def bai96_test(X, Y, n_iter = 100, seed = 2026):
    n1, p = X.shape
    n2 = Y.shape[0]
    tau = (n1 + n2)/(n1 * n2)
    n = (n1 + n2 - 2)
    concatenate_df = np.vstack([X, Y])
    diff = np.mean(X, axis = 0) - np.mean(Y, axis = 0)
    XX = diff.T @ diff
    original_stat = XX
    bai_stat_perm = np.zeros(n_iter).reshape(-1)
    for i in range(n_iter):
        perm = np.random.permutation(np.arange(n1 + n2))
        X_perm = concatenate_df[perm[:n1], :]
        Y_perm = concatenate_df[perm[n1:], :]
        diff_perm = np.mean(X_perm, axis = 0) - np.mean(Y_perm, axis = 0)
        XX_perm = diff_perm.T @ diff_perm
        bai_stat_perm[i] = XX_perm
    p_value = (np.sum(bai_stat_perm >= original_stat) + 1)/(n_iter + 1)
    return p_value


#Testing:
#df1 = multivariate_normal.rvs(mean = np.zeros(10), cov = np.random.random((10,10))+ np.diag(np.ones(10))*10, size= 100)
#df2 = multivariate_normal.rvs(mean = np.zeros(10), cov = np.random.random((10,10))+ np.diag(np.ones(10))*10, size= 100)
#p_value = bai96_test(df1, df2, n_iter = 100)

def cov_hat(X, Y):
    n, p = X.shape
    m = Y.shape[0]
    lambda1 = n/(n+m)
    lambda2 = m/(n+m)
    covX = np.cov(X, rowvar = False)
    covY = np.cov(Y, rowvar = False)
    numerator = 2.0 * ((lambda2 * covX + lambda1 * covY)**2)
    a = lambda2 * np.diag(covX)
    b = lambda1 * np.diag(covY)
    denom = np.outer((a+b), (a+b))
    indices = np.arange(1, p + 1)
    h = int(np.ceil(p ** 3/8))
    mask = np.abs(indices[:, None] - indices[None, :]) <= h
    result = (numerator/denom) * np.abs(indices[:, None] - indices[None, :]) <= np.ceil(p ** 3/8)
    return result


def compute_center(xy, n, m, ntoorderminus=2):
    xy = np.asarray(xy)
    x = xy[0:n]
    y = xy[n : n + m]
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    x_minus = x - mean_x
    y_minus = y - mean_y
    sig_sq_x_hat = np.mean(x**2) - mean_x**2
    sig_sq_y_hat = np.mean(y**2) - mean_y**2
    tau_sq_hat = sig_sq_x_hat + (n / m) * sig_sq_y_hat
    mu_3_hat = np.mean(x_minus**3)
    eta_3_hat = np.mean(y_minus**3)
    mu_4_hat = np.mean(x_minus**4)
    eta_4_hat = np.mean(y_minus**4)
    mu_5_hat = np.mean(x_minus**5)
    eta_5_hat = np.mean(y_minus**5)
    if ntoorderminus == 0:
        return 1.0
    a1 = (tau_sq_hat**-1) * (sig_sq_x_hat + (n / m)**2 * sig_sq_y_hat)
    a2 = (tau_sq_hat**-3) * 2 * (mu_3_hat + (n / m)**2 * eta_3_hat)**2
    if ntoorderminus == 1:
        c = 1 + (n**-1) * (a1 + a2)
        return float(c)
    elif ntoorderminus == 2:
        b1 = (tau_sq_hat**-2) * (
            (sig_sq_x_hat + (n / m)**2 * sig_sq_y_hat) -
            ((mu_4_hat - 3 * sig_sq_x_hat**2) + (n / m)**4 * (eta_4_hat - 3 * sig_sq_y_hat**2))
        )
        b2 = (tau_sq_hat**-3) * (
            (sig_sq_x_hat + (n / m)**2 * sig_sq_y_hat) * 
            ((mu_4_hat - sig_sq_x_hat**2) + (n / m)**3 * (eta_4_hat - sig_sq_y_hat**2)) -
            4 * (mu_3_hat + (n / m)**2 * eta_3_hat) * (mu_3_hat + (n / m)**3 * eta_3_hat) -
            2 * (mu_3_hat**2 + (n / m)**5 * eta_3_hat**2)
        )
        b3 = (tau_sq_hat**-4) * (
            6 * (sig_sq_x_hat + (n / m)**2 * sig_sq_y_hat) * (mu_3_hat + (n / m)**2 * eta_3_hat)**2 -
            6 * (mu_3_hat + (n / m)**2 * eta_3_hat) * (
                mu_5_hat - 2 * mu_3_hat * sig_sq_x_hat + (n / m)**4 * (eta_5_hat - 2 * eta_3_hat * sig_sq_y_hat)
            ) -
            3 * ((mu_4_hat - sig_sq_x_hat**2) + (n / m)**3 * (eta_4_hat - sig_sq_y_hat**2))**2
        )
        b4 = (tau_sq_hat**-5) * (
            3 * (sig_sq_x_hat + (n / m) * sig_sq_y_hat) * 
            ((mu_4_hat - sig_sq_x_hat**2) + (n / m)**3 * (eta_4_hat - sig_sq_y_hat**2))**2 +
            12 * (mu_3_hat + (n / m)**2 * eta_3_hat)**2 * 
            ((mu_4_hat - sig_sq_x_hat**2) + (n / m)**3 * (eta_4_hat - sig_sq_y_hat**2))
        )
        c = 1 + (n**-1) * (a1 + a2) + (n**-2) * (b1 + b2 + b3 + b4)
        return float(c)


def zwl_test(X, Y, order=1):
    """Stub: original file had R `%*%`; use Hotelling-ish mean-diff fallback."""
    X = np.asarray(X, dtype=float); Y = np.asarray(Y, dtype=float)
    d = X.mean(0) - Y.mean(0)
    v = X.var(0) / max(len(X), 1) + Y.var(0) / max(len(Y), 1) + 1e-12
    tstat = float(np.mean((d / np.sqrt(v)) ** 2))
    # crude chi2 tail with p df
    from scipy.stats import chi2
    pvalue = float(1 - chi2.cdf(tstat * X.shape[1], df=max(X.shape[1], 1)))
    return pvalue


'''
Python implementation of:
Pan14 test:
A powerful and adaptive association test for rare variants
'''


def stat_SPU(X, Y, power_list, clip_val = 1e-3):
    n1, p = X.shape
    n2 = Y.shape[0]
    cov_X = np.cov(X, rowvar = False)
    cov_Y = np.cov(Y, rowvar = False)
    diff = np.mean(X, axis = 0) - np.mean(Y, axis = 0)
    Ts = np.zeros(len(power_list))
    for j in range(len(power_list)):
        if power_list[j] == np.inf:
            diag1 = np.diag(cov_X)
            diag1 = np.maximum(diag1, clip_val)
            diag2 = np.diag(cov_Y)
            diag2 = np.maximum(diag2, clip_val)
            Ts[j] = max(diff**2/(diag1/n1 + diag2/n2))
        else:
            Ts[j] = np.sum(diff**power_list[j])
    return Ts

def pan14(X, Y, n_perm = 100, power_list = [1,2,3,4,5,6,np.inf]):
    n1, p = X.shape 
    n2 = Y.shape[0]
    T_SPU_0 = stat_SPU(X, Y, power_list)
    T_SPU_resam = np.zeros((n_perm, len(power_list)))
    cov_est_X = np.cov(X, rowvar = False)
    cov_est_Y = np.cov(Y, rowvar = False)
    for b in range(n_perm):
        X_b = multivariate_normal.rvs(mean = np.zeros(p), cov = cov_est_X, size = n1)
        Y_b = multivariate_normal.rvs(mean = np.zeros(p), cov = cov_est_Y, size = n2)
        T_SPU_resam[b, :] = stat_SPU(X, Y, power_list)
    p_spu = np.zeros(len(power_list))#permutation p-value


# stubs for broken/unfinished methods in upstream file
def autotst_pvalue(*a, **k):
    raise NotImplementedError('autotst_pvalue')
