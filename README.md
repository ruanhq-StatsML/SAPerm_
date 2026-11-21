# Hypothesis Testing Procedure for Distribution Shift with Application on Sentiment Analysis and Image Embedding

## Trustworthy Eval & Attribution Agent

See [`agents/shift-detect`](agents/shift-detect) for the Agent SDK agent that covers:

1. **Distribution-shift attribution** — detect → classify → pathway/LOCO attribute → Eval Gate  
2. **Trustworthy text/multimodal LLM evaluation** — versioned rubrics/datasets, cross-model对照, same-model对齐, eval-side behavior interpretability, Trust Eval Gate  

It wraps methods in this repo (and sibling RFPerm / CFPerm / OnlineMonitoring / FSDS work) and provides a portable offline suite for daily eval systematization.

---

Implementation for the Permutation Test for Distribution Shift with Application to Sentiment Analysis: Both Tree-Based Permutation Test and R-risk based Tests. We benchmark the following 15 methods with the 6 developed methods: 
- RFPerm: Random Forest Permutation Test for Distribution Shift 
- XGBPerm: XGBoost based Permutation Test for Distribution Shift \
---- Causal Inference Based Test followed by a Permute-then-Refit Procedure on the Variable Importance 
- CFPerm_loco:      Leave-one-Covariate-out Variable Importance 
- CFPerm_PermuCATE: Conditional Permutation Variable Importance 
- CFPerm_grf:       Causal Forest Split Variable Importance 
- DRPerm:           Hypothesis Test via the doubly robust pseudo-outcome learner 
- RRPerm:           Hypothesis Test via the R-learner 
-------- 
Benchmark Methods
We compare against the following state‑of‑the‑art and classical two‑sample testing methods via Python

SKK (2013) – A two‑sample test in high‑dimensional data
Srivastava, M. S., Katayama, S., & Kano, Y. (2013).
Journal of Multivariate Analysis. [A test for high‑dimensional mean vectors with unequal covariance matrices.]

Conformal (2024) – A two‑sample conditional distribution test using conformal prediction and weighted rank sum
Hu, X., et al. (2024).
arXiv preprint. [Conditional distribution test via conformal prediction and weighted rank sum.]

C2ST (2016) – Revisiting Classifier Two‑Sample Tests
Lopez‑Paz, D., & Oquab, M. (2016).
ICLR 2017. [Uses a classifier’s accuracy to test whether two samples come from the same distribution.]

KMMD (2012) – A Kernel Two‑Sample Test (Kernel MMD)
Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
Journal of Machine Learning Research. [Maximum Mean Discrepancy with a characteristic kernel.]

Miles (2011) – A more powerful two‑sample test in high dimensions using random projection
Lopes, M. E., Jacob, L., & Wainwright, M. J. (2011).
Advances in Neural Information Processing Systems. [Random projection based test for high‑dimensional means.]

Xu16 (2016) – An adaptive two‑sample test for high‑dimensional means
Xu, G., Lin, L., Wei, P., & Pan, W. (2016).
Biometrika. [Adaptive thresholding for sparse high‑dimensional mean differences.]

Chen10 (2010) – A two‑sample test for high‑dimensional data with applications to gene‑set testing
Chen, S. X., & Qin, Y. L. (2010).
The Annals of Statistics. [Uses a regularized Hotelling’s T² for gene‑set analysis.]

Chen14 (2014) – Two‑Sample Tests for High Dimensional Means with Thresholding and Data Transformation
Chen, S. X., Li, J., & Zhong, P. S. (2014).
Journal of the American Statistical Association. [Combines thresholding and transformation to handle sparse signals.]

Sri08 (2008) – A test for the mean vector with fewer observations than the dimension
Srivastava, M. S. (2008).
Journal of Multivariate Analysis. [Tests for the mean vector when p > n.]

Pan14 (2014) – A powerful and adaptive association test for rare variants
Pan, W. (2014).
Genetic Epidemiology. [Adaptive Two-sample Testing for Distribution Shift]

AutoTST (2020) – AutoML Two‑Sample Test
Kübler, J., et al. (2022).
[An automated machine learning approach to two‑sample testing. Please check the original source for exact year and citation.]

Bai96 (1996) – Effect of high dimension: by an example of a two sample problem
Bai, Z., & Saranadasa, H. (1996).
Statistica Sinica. [Demonstrates failure of classical Hotelling’s T² when dimension grows with sample size.]

ZWL (Zhang et al.) – Result consistency of high‑dimensional two‑sample tests applied to gene ontology terms with gene sets
Zhang, W., et al. (2020).
[A study on consistency of high‑dimensional tests when applied to gene ontology terms. Please provide the full citation if available.]

### Python Example

```python

'''
Illustration for the biasfinder dataset(with both concept drift and covariate shift exists)
'''
model_registry = default_model_registry(
  ntree = 150,
  ridge_alpha = 0.25,
  nthread = 1, maxit = 200, max_depth = 5,
  gamma = 0.25, eta = 0.15, mlp_hidden_size = 4,
  mlp_decay = 1e-5, mlp_max_iter = 500, mlp_trace = False,
  mlp_max_coef_reg = 10000, mlp_max_coef_clf = 10000,
  warn_xgb_labels = True, positive_class = 1
)
func_config = {
    'n_perm_cf': 150,
    'n_perm_rf': 150,
    'n_perm_drperm': 150,
    'n_perm_rrperm': 150,
    'outcome_model_causal': 'rf_regressor',
    'ps_model': '',
    'kernel': 'rbf',
    'n_conformal': 150,
    'ps_model': 'logistic_classifier'
}
df1 = pd.read_csv("datasets/train_biasfinder_bert_embedding.csv").drop(['Unnamed: 0'], axis = 1)
df2 = pd.read_csv("datasets/test_biasfinder_bert_embedding.csv").drop(['Unnamed: 0'], axis = 1)
df1 = df1.sample(500)
df2 = df2.sample(500)
df1_X = df1.iloc[:,:(df1.shape[1]-1)]
df2_X = df2.iloc[:,:(df2.shape[1]-1)]
df1['sentiment'] = df1['class'].apply(lambda X: 1 if X == 'sarc' else 0)
Y1 = np.asarray(df1['sentiment']).astype(float)
Y2 = np.asarray(df2['sentiment']).astype(float)
df_exist = np.column_stack([df1_X, Y1])
df_new = np.column_stack([df2_X, Y2])

result = benchmark_all_method(df1, df2, config = func_config)
```

### Illustration for Fine-Tuning & Word Embedding and Testing Pipeline:
 [https://colab.research.google.com/drive/15K7Y0g0ic4LlUon5enWHcEZPdd6vcKwR#scrollTo=VV46gNnXAuy_](https://colab.research.google.com/drive/1y-Hl654ASB48EmimkRdQ79XIKbEHWzSq#scrollTo=sWb7WEPD3w05)
### Embedding Extraction with recent Contextual Word Embeddings
https://colab.research.google.com/drive/1lYQFzql9wtXdPDvEh2QQ7q7shlFMjCbv#scrollTo=BeZ86cfSYAZu
### Embedding Extraction via Vision Transformers
https://colab.research.google.com/drive/1SyI5dn7d4WPZKPXVXMMxCVZo8PePXOF0

#### For the details of the dataset, please contact hank_rhq@outlook.com for more details

