# hsnpyr

Regularized horseshoe logistic regression in NumPyro, with tools for
evaluation and variable selection.

## Features

- **Regularized horseshoe prior** (Piironen & Vehtari, 2017) with separate
  unpenalized and penalized covariate groups
- **Samplers**: NUTS (via NumPyro) and MCLMC (via BlackJAX)
- **Evaluation**: C-statistic, logarithmic score, weight of evidence
  densities, and expected information for discrimination
- **K-fold cross-validation** with automatic memory-aware parallelism
- **Learning curves** with saturation model fit
- **Projection predictive forward search** (Piironen & Vehtari, 2020) for
  variable selection, with pre-screening and warm-starting for scalability
- Diagnostic plots: log(tau)--log(eta) pairs, weight of evidence densities,
  learning curves, projpred KL path

## Installation

Requires Python 3.10+. Install dependencies:

```
pip install numpy pandas matplotlib arviz scipy scikit-learn jax numpyro blackjax
```

Then place `hslogistic.py` on your Python path, or install in development mode
once a `pyproject.toml` is added.

## Quick start

```python
import hslogistic as hs
import jax.numpy as jnp

# fit the model
result = hs.fit(X_u, X, y, slab_scale=2.0, slab_df=4.0, scale_global=0.01)

# predict probabilities
probs = hs.predict(result, X_u_new, X_new, slab_scale=2.0, slab_df=4.0, scale_global=0.01)

# variable selection
selected, kl_path, kl_null = hs.projpred_forward_search(result, X_u, X, V=10)
hs.plot_projpred(selected, kl_path, kl_null, "my_analysis")

# cross-validation
cv = hs.crossvalidate(X_u, X, y, K=5, slab_scale=2.0, slab_df=4.0, scale_global=0.01)
```

## Demo

A full worked example on simulated sparse data (N=200, J=20, 3 true signals):

```
python demo_hslogistic.py
```

## API

| Function | Description |
|---|---|
| `fit` | Fit horseshoe logistic regression (NUTS or MCLMC) |
| `predict` | Posterior predictive probabilities |
| `crossvalidate` | K-fold CV with parallel fold execution |
| `projpred_forward_search` | Projection predictive variable selection |
| `summary_report` | Parameter summary table (CSV) |
| `learning_curve` | Info for discrimination vs training size |
| `cstatistic` | Concordance statistic (AUC) |
| `log_score` | Logarithmic scoring rule |
| `Wdensities` | Weight of evidence density estimation |
| `wevid` | Kernel density weight of evidence |
| `get_info_discrim` | Expected information for discrimination |
| `recalibrate_probs` | Platt scaling recalibration |
| `plot_learning_curve` | Learning curve with fitted saturation model |
| `plot_pair_diagnostic` | log(tau)--log(eta) scatter with divergences |
| `plot_wevid` | Weight of evidence density plot |
| `plot_projpred` | KL divergence path from projpred search |

## References

- Piironen, J. and Vehtari, A. (2017). Sparsity information and regularization
  in the horseshoe and other shrinkage priors. *Electronic Journal of
  Statistics*, 11(2):5018-5051.
- Piironen, J. and Vehtari, A. (2020). Projective inference in
  high-dimensional problems: prediction and feature selection. *Electronic
  Journal of Statistics*, 14(1):2155-2197.
