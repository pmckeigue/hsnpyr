import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from scipy import stats
from scipy.special import logit, expit
from scipy.optimize import minimize_scalar, minimize, curve_fit
from sklearn.linear_model import LogisticRegression

import jax
import jax.numpy as jnp
import numpyro as npyr
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer.util import initialize_model
import blackjax

# Ensure tqdm can detect a terminal width in non-TTY environments
# (cloud notebooks, piped output) so progress bars update in-place.
os.environ.setdefault("COLUMNS", "120")

__all__ = [
    "fit", "predict", "crossvalidate",
    "cstatistic", "log_score",
    "Wdensities", "wevid", "get_info_discrim",
    "recalibrate_probs",
    "learning_curve", "summary_report",
    "projpred_forward_search",
    "plot_learning_curve", "plot_pair_diagnostic",
    "plot_wevid", "plot_forest", "plot_projpred",
    "run_analysis",
]


def hslogistic(X_u=None, X=None, y=None, slab_scale=None, slab_df=None, scale_global=None):
    """NumPyro model for logistic regression with regularized horseshoe prior.

    Unpenalized coefficients (beta_u) receive a wide Normal(0, 10) prior.
    Penalized coefficients (beta) receive a regularized horseshoe prior
    controlled by ``scale_global`` (tau), ``slab_scale`` (eta), and
    ``slab_df``.

    Parameters
    ----------
    X_u : ndarray (N, U)
        Design matrix for unpenalized covariates (typically includes an
        intercept column).
    X : ndarray (N, J)
        Design matrix for penalized covariates.
    y : ndarray (N,)
        Binary outcome (0/1).
    slab_scale : float
        Scale of the regularizing slab on large coefficients.
    slab_df : float
        Degrees of freedom for the slab inverse-gamma prior.
    scale_global : float
        Global shrinkage scale, controls overall sparsity.
    """
    nu_local = 1.
    nu_global = 1.
    U = X_u.shape[1]
    J = X.shape[1]
    N = X.shape[0]
    with npyr.plate("U unpenalized covariates", U):
        beta_u = npyr.sample("beta_u", dist.Normal(0, 10.0))

    aux1_global =  npyr.sample("aux1_global", dist.HalfNormal(1.0))
    aux2_global = npyr.sample("aux2_global", dist.InverseGamma(0.5 * nu_global, 0.5 * nu_global))
    τ = npyr.deterministic("tau", aux1_global * jnp.sqrt(aux2_global) * scale_global)
    caux = npyr.sample("caux", dist.InverseGamma(0.5 * slab_df, 0.5 * slab_df))
    eta = npyr.deterministic("eta", slab_scale * jnp.sqrt(caux))
    log_τ = npyr.deterministic("log_tau", jnp.log(τ))
    log_eta = npyr.deterministic("log_eta", jnp.log(eta))
    with npyr.plate("J penalized covariates", J):
        z = npyr.sample("z", dist.Normal(0, 1.0))
        aux1_local = npyr.sample("aux1_local", dist.HalfNormal(1.0))
        aux2_local = npyr.sample("aux2_local", dist.InverseGamma(0.5 * nu_local, 0.5 * nu_local))
        lambda_raw = jnp.multiply(aux1_local, jnp.sqrt(aux2_local))
        lambda_tilde = jnp.sqrt(jnp.divide(eta**2 * jnp.square(lambda_raw),
                                           eta**2 + τ**2 * jnp.square(lambda_raw)))
        beta = npyr.deterministic("beta", jnp.multiply(z, lambda_tilde * τ))
    logodds =  npyr.deterministic("logodds", jnp.dot(X_u, beta_u) + jnp.dot(X, beta))
    with npyr.plate("N observations", N):
        npyr.sample("y", dist.Bernoulli(logits=logodds), obs=y)


class _SamplesResult:
    """Wrapper so that MCLMC results have the same interface as MCMC."""
    def __init__(self, samples):
        self._samples = samples
    def get_samples(self):
        return self._samples


def fit(X_u, X, y, slab_scale=1.0, slab_df=4.0, scale_global=1.0,
        num_warmup=1000, num_samples=1000, num_chains=4,
        target_accept_prob=0.95, max_tree_depth=12, rng_seed=0,
        sampler="nuts", print_summary=True):
    """Fit the horseshoe logistic regression model via MCMC.

    Parameters
    ----------
    X_u : array (N, U)
        Unpenalized design matrix.
    X : array (N, J)
        Penalized design matrix.
    y : array (N,)
        Binary outcome.
    slab_scale : float
        Regularizing slab scale.
    slab_df : float
        Slab inverse-gamma degrees of freedom.
    scale_global : float
        Global shrinkage scale.
    num_warmup : int
        Number of warmup (adaptation) iterations per chain.
    num_samples : int
        Number of posterior samples per chain.
    num_chains : int
        Number of MCMC chains.
    target_accept_prob : float
        NUTS target acceptance probability.
    max_tree_depth : int
        NUTS maximum tree depth.
    rng_seed : int
        Random seed.
    sampler : {"nuts", "mclmc"}
        Sampling algorithm.
    print_summary : bool
        If True, print the full NumPyro summary table to stdout.

    Returns
    -------
    MCMC or _SamplesResult
        Fitted sampler result with ``.get_samples()`` method.
    """
    # Validate inputs: non-finite values cause BernoulliLogits errors
    for name, arr in [("X_u", X_u), ("X", X), ("y", y)]:
        if not jnp.all(jnp.isfinite(arr)):
            n_bad = int((~jnp.isfinite(arr)).sum())
            raise ValueError(
                f"{name} contains {n_bad} non-finite values (NaN/Inf). "
                f"Clean the data before fitting.")
    model_kwargs = dict(X_u=X_u, X=X, y=y,
                        slab_scale=slab_scale, slab_df=slab_df,
                        scale_global=scale_global)
    if sampler == "nuts":
        kernel = NUTS(
            hslogistic,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
        )
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
        )
        mcmc.run(jax.random.PRNGKey(rng_seed), **model_kwargs)
        if print_summary:
            mcmc.print_summary()
        return mcmc
    elif sampler == "mclmc":
        return _fit_mclmc(model_kwargs, num_warmup, num_samples, rng_seed)
    else:
        raise ValueError(f"Unknown sampler: {sampler!r}. Use 'nuts' or 'mclmc'.")


def _fit_mclmc(model_kwargs, num_warmup, num_samples, rng_seed):
    """Fit using the MCLMC sampler via BlackJAX.

    Parameters
    ----------
    model_kwargs : dict
        Arguments forwarded to the NumPyro model.
    num_warmup : int
        Tuning steps for step-size and trajectory-length selection.
    num_samples : int
        Number of posterior samples to draw.
    rng_seed : int
        Random seed.

    Returns
    -------
    _SamplesResult
        Wrapper around the constrained posterior samples.
    """
    rng_key = jax.random.PRNGKey(rng_seed)
    init_key, tune_key, run_key = jax.random.split(rng_key, 3)

    init_params, potential_fn_gen, postprocess_fn, _ = initialize_model(
        init_key, hslogistic,
        model_kwargs=model_kwargs,
        dynamic_args=False,
    )
    logdensity_fn = lambda position: -potential_fn_gen(position)
    initial_position = init_params.z

    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        rng_key=init_key,
    )

    kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    (state_after_tuning, sampler_params, _) = blackjax.mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_warmup,
        state=initial_state,
        rng_key=tune_key,
        diagonal_preconditioning=False,
    )

    sampling_alg = blackjax.mclmc(
        logdensity_fn,
        L=sampler_params.L,
        step_size=sampler_params.step_size,
    )

    _, samples_unconstrained = blackjax.util.run_inference_algorithm(
        rng_key=run_key,
        initial_state=state_after_tuning,
        inference_algorithm=sampling_alg,
        num_steps=num_samples,
        transform=lambda state, info: state.position,
        progress_bar=True,
    )

    # transform unconstrained samples to constrained space
    constrained = jax.vmap(postprocess_fn)(samples_unconstrained)

    print(f"MCLMC: L={float(sampler_params.L):.3f}, "
          f"step_size={float(sampler_params.step_size):.4f}, "
          f"{num_samples} samples")

    return _SamplesResult(constrained)


def predict(result, X_u_new, X_new, slab_scale=1.0, slab_df=4.0,
            scale_global=1.0, rng_seed=1):
    """Compute posterior predictive probabilities for new observations.

    Parameters
    ----------
    result : MCMC or _SamplesResult
        Fitted model returned by :func:`fit`.
    X_u_new : array (N_new, U)
        Unpenalized design matrix for new data.
    X_new : array (N_new, J)
        Penalized design matrix for new data.
    slab_scale, slab_df, scale_global : float
        Must match the values used in :func:`fit`.
    rng_seed : int
        Random seed for the predictive.

    Returns
    -------
    ndarray (S, N_new)
        Predicted probabilities, one row per posterior sample.
    """
    posterior_samples = result.get_samples()
    predictive = Predictive(
        hslogistic,
        posterior_samples=posterior_samples,
        return_sites=["logodds"],
    )
    preds = predictive(
        jax.random.PRNGKey(rng_seed),
        X_u=X_u_new, X=X_new, y=None,
        slab_scale=slab_scale, slab_df=slab_df, scale_global=scale_global,
    )
    return jax.nn.sigmoid(preds["logodds"])


def cstatistic(y, probs):
    """Concordance statistic (area under the ROC curve).

    Computed as the proportion of concordant pairs among all
    case-control pairs: P(prob_case > prob_control).
    Ties contribute 0.5.
    """
    y = jnp.asarray(y, dtype=bool)
    probs = jnp.asarray(probs)
    cases = probs[y]
    controls = probs[~y]
    # (n_cases, n_controls) pairwise comparisons
    diff = cases[:, None] - controls[None, :]
    concordant = jnp.sum(diff > 0) + 0.5 * jnp.sum(diff == 0)
    return float(concordant / (len(cases) * len(controls)))


def reweight_densities(theta, n_ctrls, n_cases, fhat_ctrls, fhat_cases, xseq, wts):
    """Reweight KDE densities so that control and case integrals are equal.

    Parameters
    ----------
    theta : float
        Reweighting parameter (optimised externally).
    n_ctrls, n_cases : int
        Number of controls and cases.
    fhat_ctrls, fhat_cases : ndarray
        Raw KDE densities evaluated on *xseq*.
    xseq : ndarray
        Grid of weight-of-evidence values.
    wts : ndarray (len(xseq), 2)
        Mixture weights for controls and cases at each grid point.

    Returns
    -------
    DataFrame
        Columns ``f_ctrls`` and ``f_cases``: reweighted densities.
    """
    mean_ctrls = np.sum(fhat_ctrls * xseq) / np.sum(fhat_ctrls)
    mean_cases = np.sum(fhat_cases * xseq) / np.sum(fhat_cases)
    weights_ctrls = n_ctrls * np.exp(-theta * np.square(xseq - mean_ctrls))
    weights_cases = n_cases * np.exp(+theta * np.square(xseq - mean_cases))
    weights_ctrls = weights_ctrls / np.sum(weights_ctrls)
    weights_cases = weights_cases / np.sum(weights_cases)

    fhat_geomean = (wts[:, 0] * fhat_ctrls * weights_ctrls * np.exp(+0.5 * xseq)
                    + wts[:, 1] * fhat_cases * weights_cases * np.exp(-0.5 * xseq))

    f_ctrls = fhat_geomean * np.exp(-0.5 * xseq)
    f_cases = fhat_geomean * np.exp(+0.5 * xseq)
    return pd.DataFrame({'f_ctrls': f_ctrls, 'f_cases': f_cases})


def error_integrals(theta, n_ctrls, n_cases, f0, f1, xseq, wts):
    """Objective for density reweighting: absolute log-ratio of integrals.

    Returns zero when the reweighted control and case densities integrate
    to the same value, i.e. the densities are balanced.
    """
    wdens = reweight_densities(theta, n_ctrls, n_cases, f0, f1, xseq, wts)
    obj = abs(np.log(np.sum(wdens["f_ctrls"]) / np.sum(wdens["f_cases"])))
    return obj


def wevid(W_df, n_ctrls, n_cases):
    """Compute weight-of-evidence density curves from a W DataFrame.

    Estimates separate KDEs for controls and cases, reweights them so
    their integrals match, and returns the adjusted density curves.

    Parameters
    ----------
    W_df : DataFrame
        Must contain columns ``y`` (0/1) and ``W`` (log-likelihood ratio
        minus prior log-odds).
    n_ctrls, n_cases : int
        Number of controls and cases.

    Returns
    -------
    dict
        Keys ``xseq``, ``x_stepsize``, ``f_ctrls``, ``f_cases``.
    """
    W0 = W_df.query("y==0")["W"]
    W1 = W_df.query("y==1")["W"]
    try:
        kernel0 = stats.gaussian_kde(W0)
        kernel1 = stats.gaussian_kde(W1)
    except np.linalg.LinAlgError:
        bw0 = 2.0 * W0.std() * len(W0) ** (-1.0 / 5)
        bw1 = 2.0 * W1.std() * len(W1) ** (-1.0 / 5)
        kernel0 = stats.gaussian_kde(W0, bw_method=bw0)
        kernel1 = stats.gaussian_kde(W1, bw_method=bw1)
    x_stepsize = 0.1
    xseq = np.arange(-10, 10, x_stepsize)
    f0 = kernel0(xseq)
    f1 = kernel1(xseq)
    wts = np.column_stack((np.exp(-0.5 * xseq) * n_ctrls,
                           np.exp(+0.5 * xseq) * n_cases))
    wts = np.divide(wts, wts.sum(axis=1).reshape(len(xseq), 1))

    f = lambda theta: error_integrals(theta, n_ctrls=n_ctrls, n_cases=n_cases,
                                      f0=f0, f1=f1, xseq=xseq, wts=wts)
    res = minimize_scalar(f, bounds=(-0.5, 0.5), method='bounded')
    theta = res.x
    wdens_adjusted = reweight_densities(theta, n_ctrls, n_cases, f0, f1, xseq, wts)

    z = 0.5 * (np.sum(wdens_adjusted["f_ctrls"]) + np.sum(wdens_adjusted["f_cases"])) * x_stepsize
    f_cases = wdens_adjusted["f_cases"] / z
    f_ctrls = wdens_adjusted["f_ctrls"] / z
    return {'xseq': xseq, 'x_stepsize': x_stepsize, 'f_ctrls': f_ctrls, 'f_cases': f_cases}


def recalibrate_probs(y, probs):
    """Recalibrate predicted probabilities using Platt scaling.

    Fits a univariate logistic regression of *y* on logit(*probs*) and
    returns the recalibrated probabilities.

    Parameters
    ----------
    y : array-like (N,)
        Binary outcome.
    probs : array-like (N,)
        Predicted probabilities to recalibrate.

    Returns
    -------
    ndarray (N,)
        Recalibrated probabilities.
    """
    eps = 1e-8
    probs = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    logit_p = logit(probs).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, solver='lbfgs')
    model.fit(logit_p, np.asarray(y))
    return model.predict_proba(logit_p)[:, 1]


def Wdensities(y, predicted_y, recalibrate=True):
    """Compute weight-of-evidence densities from predicted probabilities.

    Converts predicted probabilities to log-likelihood-ratio weights,
    optionally recalibrates them, and returns density curves via
    :func:`wevid`.

    Parameters
    ----------
    y : array-like (N,)
        Binary outcome.
    predicted_y : array-like (N,)
        Predicted probabilities.
    recalibrate : bool
        If True, apply Platt scaling before computing W.

    Returns
    -------
    dict
        Keys ``xseq``, ``x_stepsize``, ``f_ctrls``, ``f_cases``.
    """
    y = np.asarray(y)
    predicted_y = np.asarray(predicted_y, dtype=np.float64)
    if recalibrate:
        predicted_y = recalibrate_probs(y, predicted_y)
    unique, counts = np.unique(y, return_counts=True)
    y_counts = dict(zip(unique, counts))
    n_ctrls = y_counts[0]
    n_cases = y_counts[1]
    logodds_prior = logit(n_cases / (n_ctrls + n_cases))

    eps = 1e-8
    predicted_y = np.clip(predicted_y, eps, 1.0 - eps)
    W = logit(predicted_y) - logodds_prior
    W_df = pd.DataFrame({'y': y, 'W': W})
    return wevid(W_df, n_ctrls, n_cases)


def get_info_discrim(w):
    """Expected information for discrimination (bits) from W-densities.

    Parameters
    ----------
    w : dict
        Output of :func:`wevid` or :func:`Wdensities`.

    Returns
    -------
    float
        Information for discrimination, rounded to two decimal places.
    """
    info_discrim = ((w["xseq"] * w["f_cases"]).sum()
                    - (w["xseq"] * w["f_ctrls"]).sum()) * w["x_stepsize"] * 0.5 / np.log(2)
    return round(info_discrim, 2)


def log_score(y, probs):
    """Logarithmic (log-likelihood) scoring rule.

    Parameters
    ----------
    y : array-like (N,)
        Binary outcome.
    probs : array-like (N,)
        Predicted probabilities.

    Returns
    -------
    float
        Sum of log-likelihoods: sum[y*log(p) + (1-y)*log(1-p)].
    """
    eps = 1e-15
    probs = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(y)
    return float(np.sum(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))


def _get_available_memory_bytes():
    """Read available memory from /proc/meminfo (Linux)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB to bytes
    except (FileNotFoundError, ValueError, OSError):
        pass
    return None


def _estimate_fold_memory_bytes(N_train, U, J, num_chains, num_samples):
    """Rough estimate of memory needed per CV fold in bytes."""
    n_params = U + J * 4 + 4  # beta_u, z/aux1/aux2_local per J, plus globals
    samples_bytes = num_chains * num_samples * n_params * 4
    data_bytes = N_train * (J + U + 1) * 4
    # 3x safety factor for JAX intermediates during sampling
    model_bytes = int((samples_bytes + data_bytes) * 3)
    # baseline for JAX/XLA runtime + JIT compilation cache per process
    jax_baseline = 512 * 1024 * 1024  # 512 MB
    return model_bytes + jax_baseline


def _cv_fold_worker(fold_args):
    """Run fit + predict for a single CV fold (module-level for pickling)."""
    (k, K, train_idx, test_idx, X_u, X, y,
     slab_scale, slab_df, scale_global,
     num_warmup, num_samples, num_chains,
     target_accept_prob, max_tree_depth, rng_seed, sampler) = fold_args

    print(f"\n--- Fold {k+1}/{K}: train={len(train_idx)}, test={len(test_idx)} ---")
    result_k = fit(
        jnp.array(X_u[train_idx]), jnp.array(X[train_idx]),
        jnp.array(y[train_idx]),
        slab_scale=slab_scale, slab_df=slab_df, scale_global=scale_global,
        num_warmup=num_warmup, num_samples=num_samples,
        num_chains=num_chains, target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth, rng_seed=rng_seed + k,
        sampler=sampler,
    )
    probs_k = predict(
        result_k, jnp.array(X_u[test_idx]), jnp.array(X[test_idx]),
        slab_scale=slab_scale, slab_df=slab_df, scale_global=scale_global,
        rng_seed=rng_seed + k,
    )
    return k, y[test_idx], np.array(probs_k.mean(axis=0))


def crossvalidate(X_u, X, y, K=5, slab_scale=1.0, slab_df=4.0,
                  scale_global=1.0, num_warmup=1000, num_samples=1000,
                  num_chains=4, target_accept_prob=0.95, max_tree_depth=12,
                  rng_seed=0, sampler="nuts", max_workers=None):
    """K-fold cross-validation with automatic parallelisation.

    Fits the model on K-1 folds and predicts the held-out fold,
    then aggregates predictions to compute C-statistic, expected
    information for discrimination, and log score.

    Parameters
    ----------
    X_u, X, y : array-like
        Design matrices and outcome (see :func:`fit`).
    K : int
        Number of folds.
    slab_scale, slab_df, scale_global : float
        Horseshoe prior parameters.
    num_warmup, num_samples, num_chains : int
        MCMC sampler settings.
    target_accept_prob : float
        NUTS target acceptance probability.
    max_tree_depth : int
        NUTS maximum tree depth.
    rng_seed : int
        Random seed (incremented per fold).
    sampler : {"nuts", "mclmc"}
        Sampling algorithm.
    max_workers : int or None
        Maximum parallel fold workers.  None uses all available CPUs
        subject to a memory-based limit.

    Returns
    -------
    dict
        Keys: ``y``, ``probs``, ``c_stat``, ``info_discrim``,
        ``log_score``, ``wevid``.
    """
    X_u = np.asarray(X_u)
    X = np.asarray(X)
    y = np.asarray(y)
    N = X.shape[0]
    U = X_u.shape[1]
    J = X.shape[1]
    indices = np.arange(N)
    rng = np.random.RandomState(rng_seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, K)

    # determine number of parallel workers
    n_cpus = os.cpu_count() or 1
    n_workers = min(K, n_cpus)

    mem_avail = _get_available_memory_bytes()
    if mem_avail is not None:
        N_train = int(N * (K - 1) / K)
        mem_per_fold = _estimate_fold_memory_bytes(N_train, U, J,
                                                   num_chains, num_samples)
        mem_limit = max(1, int(mem_avail * 0.8) // mem_per_fold)
        n_workers = min(n_workers, mem_limit)
        print(f"Memory: {mem_avail / 1e9:.1f} GB available, "
              f"~{mem_per_fold / 1e6:.0f} MB per fold, "
              f"limit {mem_limit} parallel folds")

    if max_workers is not None:
        n_workers = min(n_workers, max_workers)
    n_workers = max(1, n_workers)

    # build fold arguments
    fold_args = []
    for k in range(K):
        test_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
        fold_args.append((
            k, K, train_idx, test_idx, X_u, X, y,
            slab_scale, slab_df, scale_global,
            num_warmup, num_samples, num_chains,
            target_accept_prob, max_tree_depth, rng_seed, sampler,
        ))

    if n_workers > 1:
        print(f"Running {K}-fold CV with {n_workers} parallel workers")
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            results = list(pool.map(_cv_fold_worker, fold_args))
    else:
        print(f"Running {K}-fold CV sequentially")
        results = [_cv_fold_worker(a) for a in fold_args]

    # sort by fold index and collect
    results.sort(key=lambda x: x[0])
    all_y = np.concatenate([r[1] for r in results])
    all_probs = np.concatenate([r[2] for r in results])

    c_stat = cstatistic(all_y, all_probs)
    w = Wdensities(all_y, all_probs, recalibrate=True)
    info_discrim = get_info_discrim(w)
    lscore = log_score(all_y, all_probs)

    print(f"\n{K}-fold cross-validation (N={N}):")
    print(f"  C-statistic                            = {c_stat:.3f}")
    print(f"  Expected information for discrimination = {info_discrim} bits")
    print(f"  Logarithmic score                      = {lscore:.3f}")

    return {'y': all_y, 'probs': all_probs, 'c_stat': c_stat,
            'info_discrim': info_discrim, 'log_score': lscore, 'wevid': w}


def learning_curve(X_u, X, y, K_values=(2, 3, 4, 5), slab_scale=1.0,
                   slab_df=4.0, scale_global=1.0, num_warmup=1000,
                   num_samples=1000, num_chains=4, target_accept_prob=0.95,
                   max_tree_depth=12, rng_seed=0, sampler="nuts",
                   max_workers=None):
    """Learning curve: information for discrimination vs training size.

    Runs :func:`crossvalidate` for each value of K and records how
    discriminative performance varies with training-set size.

    Parameters
    ----------
    X_u, X, y : array-like
        Design matrices and outcome.
    K_values : tuple of int
        Fold counts to evaluate.  Each K yields a training size of
        approximately N*(K-1)/K.
    slab_scale, slab_df, scale_global : float
        Horseshoe prior parameters.
    num_warmup, num_samples, num_chains : int
        MCMC sampler settings.
    target_accept_prob, max_tree_depth : float, int
        NUTS settings.
    rng_seed : int
        Random seed.
    sampler : {"nuts", "mclmc"}
        Sampling algorithm.
    max_workers : int or None
        Forwarded to :func:`crossvalidate`.

    Returns
    -------
    train_sizes : ndarray
        Training-set sizes.
    info_values : ndarray
        Corresponding information-for-discrimination values (bits).
    """
    N = X.shape[0]
    train_sizes = []
    info_values = []
    for K in K_values:
        n_train = int(N * (K - 1) / K)
        print(f"\n{'='*60}")
        print(f"K={K}  (train size ~ {n_train})")
        print(f"{'='*60}")
        cv = crossvalidate(
            X_u, X, y, K=K,
            slab_scale=slab_scale, slab_df=slab_df,
            scale_global=scale_global, num_warmup=num_warmup,
            num_samples=num_samples, num_chains=num_chains,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth, rng_seed=rng_seed,
            sampler=sampler, max_workers=max_workers,
        )
        train_sizes.append(n_train)
        info_values.append(cv["info_discrim"])
    return np.array(train_sizes), np.array(info_values)


def plot_learning_curve(train_sizes, info_values, filestem):
    """Plot learning curve and fit a saturation model.

    Fits Lambda(n) = a*n / (b + n) and saves the figure to
    ``{filestem}_learning_curve.pdf``.

    Parameters
    ----------
    train_sizes : array-like
        Training-set sizes.
    info_values : array-like
        Information-for-discrimination values (bits).
    filestem : str
        Output file prefix.

    Returns
    -------
    str
        Path to the saved PDF.
    """
    def _saturation(n, a, b):
        return a * n / (b + n)

    popt, _ = curve_fit(_saturation, train_sizes, info_values,
                        p0=[max(info_values) * 2, train_sizes[0]],
                        bounds=([0, 0], [np.inf, np.inf]))
    a_fit, b_fit = popt

    plt.figure()
    plt.plot(train_sizes, info_values, "ko", markersize=7, label="CV estimates")
    n_curve = np.linspace(0, train_sizes[-1] * 1.2, 200)
    plt.plot(n_curve, _saturation(n_curve, a_fit, b_fit), "b-",
             label=rf"$\Lambda(n) = {a_fit:.2f}\,n\,/\,({b_fit:.0f} + n)$")
    plt.xlabel("Training set size")
    plt.ylabel("Expected information for discrimination (bits)")
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    outpath = filestem + "_learning_curve.pdf"
    plt.savefig(outpath)
    plt.show()
    plt.close()

    print(f"\nFitted curve: Lambda(n) = {a_fit:.2f} * n / ({b_fit:.0f} + n)")
    print(f"  Asymptote (a) = {a_fit:.2f} bits")
    print(f"  Half-max (b)  = {b_fit:.0f} observations")
    return outpath


def summary_report(mcmc, filepath, unpenalized_names=None,
                    penalized_names=None):
    """Posterior summary table saved to CSV.

    Parameters
    ----------
    mcmc : MCMC
        Fitted NUTS result (must support ``group_by_chain``).
    filepath : str
        Path for the output CSV.
    unpenalized_names : list of str, optional
        Display names for beta_u parameters (length U).  When provided,
        ``beta_u[0]`` is labelled ``unpenalized_names[0]`` (typically
        ``"Intercept"``), etc.  Default ``None`` keeps ``beta_u[i]``.
    penalized_names : list of str, optional
        Display names for penalized covariates (length J).  When
        provided, the top-5 penalized betas (by squared posterior mean)
        are appended to the table using these names.
    """
    chain_samples = mcmc.get_samples(group_by_chain=True)

    def _row(name, x_chain):
        x_flat = x_chain.reshape(-1)
        return {
            "parameter": name,
            "mean": float(jnp.mean(x_flat)),
            "q0.03": float(jnp.percentile(x_flat, 3)),
            "q0.97": float(jnp.percentile(x_flat, 97)),
            "n_eff": float(effective_sample_size(np.array(x_chain))),
            "r_hat": float(split_gelman_rubin(np.array(x_chain))),
        }

    rows = [_row("tau", chain_samples["tau"]),
            _row("eta", chain_samples["eta"])]

    beta_u = chain_samples.get("beta_u")
    if beta_u is not None:
        U = beta_u.shape[-1]
        for i in range(U):
            label = unpenalized_names[i] if unpenalized_names is not None else f"beta_u[{i}]"
            rows.append(_row(label, beta_u[..., i]))

    # top-5 penalized covariates by squared posterior mean
    beta_chain = chain_samples.get("beta")      # (chains, samples, J)
    if penalized_names is not None and beta_chain is not None:
        beta_mean = np.array(beta_chain.reshape(-1, beta_chain.shape[-1]).mean(axis=0))
        n_top = min(5, len(penalized_names))
        top_idx = np.argsort(beta_mean ** 2)[-n_top:][::-1]
        for idx in top_idx:
            rows.append(_row(penalized_names[idx], beta_chain[..., idx]))

    # effective number of nonzero coefficients (m_eff)
    tau_ch = chain_samples["tau"][..., None]     # (chains, samples, 1)
    eta_ch = chain_samples["eta"][..., None]
    lambda_raw = (chain_samples["aux1_local"]
                  * jnp.sqrt(chain_samples["aux2_local"]))
    lambda_tilde_sq = ((eta_ch ** 2 * lambda_raw ** 2)
                       / (eta_ch ** 2 + tau_ch ** 2 * lambda_raw ** 2))
    kappa = 1.0 / (1.0 + tau_ch ** 2 * lambda_tilde_sq)
    m_eff_chain = (1.0 - kappa).sum(axis=-1)    # (chains, samples)
    rows.append(_row("m_eff", m_eff_chain))

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False, float_format="%.4f")
    try:
        from IPython.display import display
        display(df)
    except ImportError:
        print(df.to_string(index=False))
    print(f"Summary saved to {filepath}")
    return df


def plot_pair_diagnostic(mcmc, filestem):
    """Scatter plot of log(tau) vs log(eta) with divergences highlighted.

    Saves the figure to ``{filestem}_logtau_logeta.pdf``.

    Parameters
    ----------
    mcmc : MCMC
        Fitted NUTS result.
    filestem : str
        Output file prefix.

    Returns
    -------
    str
        Path to the saved PDF.
    """
    idata = az.from_numpyro(mcmc)
    ax = az.plot_pair(
        idata,
        var_names=["log_tau", "log_eta"],
        divergences=True,
        divergences_kwargs={"color": "red", "marker": "o", "markersize": 5},
        scatter_kwargs={"alpha": 0.5, "s": 16},
    )
    outpath = filestem + "_logtau_logeta.pdf"
    fig = ax.get_figure() if hasattr(ax, "get_figure") else ax.ravel()[0].get_figure()
    fig.savefig(outpath)
    plt.show()
    plt.close(fig)
    return outpath


def plot_wevid(w, filestem):
    """Plot weight-of-evidence density curves for controls and cases.

    Saves the figure to ``{filestem}_wevid_dist.pdf``.

    Parameters
    ----------
    w : dict
        Output of :func:`wevid` or :func:`Wdensities`.
    filestem : str
        Output file prefix.

    Returns
    -------
    str
        Path to the saved PDF.
    """
    plt.figure()
    plt.plot(w["xseq"], w["f_ctrls"], label="controls")
    plt.plot(w["xseq"], w["f_cases"], label="cases")
    plt.xlim(-10, 10)
    plt.xlabel("Weight of evidence favouring case over control (nat log units)")
    plt.ylabel("Density")
    plt.legend()
    outpath = filestem + "_wevid_dist.pdf"
    plt.savefig(outpath)
    plt.show()
    plt.close()
    return outpath


def plot_forest(beta_samples, penalized_names, filestem):
    """Forest plot of penalized betas with 90% credible intervals.

    Parameters
    ----------
    beta_samples : array (S, J)
        Posterior samples of penalized coefficients.
    penalized_names : list of str
        Names of penalized covariates (length J).
    filestem : str
        Prefix for the output PDF file.

    Returns
    -------
    str
        Path to the saved plot.
    """
    beta_mean = np.array(beta_samples.mean(axis=0))
    beta_lo = np.array(jnp.percentile(beta_samples, 5, axis=0))
    beta_hi = np.array(jnp.percentile(beta_samples, 95, axis=0))

    # Sort by absolute posterior mean (largest at top)
    order = np.argsort(np.abs(beta_mean))
    beta_mean = beta_mean[order]
    beta_lo = beta_lo[order]
    beta_hi = beta_hi[order]
    names = [penalized_names[i] for i in order]

    J = len(names)
    fig, ax = plt.subplots(figsize=(6, max(3, 0.3 * J)))
    y_pos = np.arange(J)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.errorbar(beta_mean, y_pos,
                xerr=[beta_mean - beta_lo, beta_hi - beta_mean],
                fmt="o", color="steelblue", ecolor="steelblue",
                elinewidth=1.5, capsize=3, markersize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Coefficient (standardized)")
    ax.set_title("Penalized betas (posterior mean, 90% CI)")
    fig.tight_layout()
    outpath = filestem + "_forest.pdf"
    fig.savefig(outpath)
    plt.show()
    plt.close(fig)
    print(f"Plot saved to {outpath}")
    return outpath


def _project_onto_submodel(mu, Z, w0=None):
    """Project reference model probabilities onto a submodel.

    Fits logistic regression with soft targets by minimizing
    KL(p_ref || p_sub) = sum_i [mu_i log(mu_i/p_i) + (1-mu_i) log((1-mu_i)/(1-p_i))].

    Parameters
    ----------
    mu : ndarray (N,)
        Reference model posterior mean probabilities.
    Z : ndarray (N, d)
        Design matrix for the submodel.
    w0 : ndarray (d,), optional
        Initial coefficients for warm-starting. Defaults to zeros.

    Returns
    -------
    w : ndarray (d,)
        Fitted submodel coefficients.
    kl : float
        KL divergence from reference to submodel.
    """
    N, d = Z.shape
    eps = 1e-12
    mu_safe = np.clip(mu, eps, 1.0 - eps)

    # negative entropy of reference (constant w.r.t. w)
    neg_entropy = np.sum(mu_safe * np.log(mu_safe) + (1.0 - mu_safe) * np.log(1.0 - mu_safe))

    def objective(w):
        logits = Z @ w
        p = expit(logits)
        # cross-entropy: -sum [mu log p + (1-mu) log(1-p)]
        # use numerically stable form via log-sum-exp
        cross_ent = np.sum(-mu_safe * logits + np.logaddexp(0.0, logits))
        kl = cross_ent + neg_entropy
        grad = Z.T @ (p - mu_safe)
        return kl, grad

    if w0 is None:
        w0 = np.zeros(d)
    res = minimize(objective, w0, method="L-BFGS-B", jac=True)
    return res.x, res.fun


def projpred_forward_search(result, X_u, X, V=5, prescreen_k=50):
    """Projection predictive forward search (Piironen & Vehtari).

    Uses pre-screening and warm-starting for efficiency on large problems.

    Parameters
    ----------
    result : MCMC or _SamplesResult
        Fitted reference model.
    X_u : ndarray (N, U)
        Unpenalized design matrix.
    X : ndarray (N, J)
        Penalized design matrix.
    V : int
        Maximum number of penalized covariates to select.
    prescreen_k : int
        Number of top candidates to evaluate at each step.  When
        J_remaining <= prescreen_k, all candidates are evaluated.

    Returns
    -------
    selected : list of int
        Indices of selected penalized covariates in order.
    kl_path : list of float
        KL divergence at each step (length V).
    kl_null : float
        KL divergence for null submodel (X_u only).
    """
    X_u = np.asarray(X_u)
    X = np.asarray(X)
    J = X.shape[1]

    if V > J:
        import warnings
        warnings.warn(f"V={V} > J={J}; capping at J={J}")
        V = J

    # reference probabilities: average expit(logodds) over posterior samples
    logodds = np.asarray(result.get_samples()["logodds"])
    mu = expit(logodds).mean(axis=0)

    # null-model KL (unpenalized covariates only)
    w_base, kl_null = _project_onto_submodel(mu, X_u)

    selected = []
    remaining = list(range(J))
    kl_path = []

    for v in range(V):
        Z_base = np.hstack([X_u] + [X[:, [j]] for j in selected]) if selected else X_u

        # current fitted probabilities for pre-screening
        p_current = expit(Z_base @ w_base)
        residual = p_current - mu  # (N,)

        # pre-screen: rank candidates by |X_j^T residual| (gradient magnitude)
        remaining_arr = np.array(remaining)
        scores = np.abs(X[:, remaining_arr].T @ residual)
        if len(remaining_arr) > prescreen_k:
            top_idx = np.argpartition(scores, -prescreen_k)[-prescreen_k:]
            candidates = remaining_arr[top_idx].tolist()
            print(f"  step {v+1}: pre-screened {len(remaining_arr)} -> {len(candidates)} candidates")
        else:
            candidates = remaining

        # warm-start: pad previous solution with 0 for the new coefficient
        w0_template = np.append(w_base, 0.0)

        best_kl = np.inf
        best_j = None
        best_w = None
        for j in candidates:
            Z_cand = np.hstack([Z_base, X[:, [j]]])
            w, kl = _project_onto_submodel(mu, Z_cand, w0=w0_template)
            if kl < best_kl:
                best_kl = kl
                best_j = j
                best_w = w
        selected.append(best_j)
        remaining.remove(best_j)
        kl_path.append(best_kl)
        w_base = best_w
        print(f"  step {v+1}: selected X[{best_j}], KL={best_kl:.6f}")

    return selected, kl_path, kl_null


def plot_projpred(selected, kl_path, kl_null, filestem, var_names=None):
    """Plot KL divergence path from projection predictive forward search.

    Saves the figure to ``{filestem}_projpred.pdf``.

    Parameters
    ----------
    selected : list of int
        Indices of selected covariates (from :func:`projpred_forward_search`).
    kl_path : list of float
        KL divergence at each selection step.
    kl_null : float
        KL divergence for the null (unpenalized-only) submodel.
    filestem : str
        Output file prefix.
    var_names : list of str, optional
        Display names for penalized covariates.  When provided,
        annotations use these names; otherwise falls back to ``X[j]``.

    Returns
    -------
    str
        Path to the saved PDF.
    """
    steps = np.arange(len(kl_path) + 1)
    kl_values = [kl_null] + list(kl_path)

    fig, ax = plt.subplots()
    ax.plot(steps, kl_values, "ko-", markersize=7)
    # annotate each selected variable, alternating above/below to avoid overlap
    for i, j in enumerate(selected):
        label = var_names[j] if var_names is not None else f"X[{j}]"
        if i % 2 == 0:
            xytext = (6, 8)
            va = "bottom"
        else:
            xytext = (6, -8)
            va = "top"
        ax.annotate(label, (i + 1, kl_path[i]),
                    textcoords="offset points", xytext=xytext,
                    fontsize=8, va=va)
    ax.set_xlabel("Number of selected covariates")
    ax.set_ylabel("KL divergence from reference model")
    ax.set_title("Projection predictive forward search")
    ax.set_xlim(-0.3, len(kl_path) + 0.3)
    fig.tight_layout()
    outpath = filestem + "_projpred.pdf"
    fig.savefig(outpath)
    plt.show()
    plt.close(fig)
    print(f"Plot saved to {outpath}")
    return outpath


def run_analysis(df, y_col, unpenalized_cols, penalized_cols, filestem,
                 slab_scale=2.0, slab_df=4.0, p0=None, scale_global=None,
                 standardize=True, sampler="nuts", crossvalidate_=False,
                 num_warmup=1000, num_samples=1000, num_chains=4,
                 target_accept_prob=0.95, max_tree_depth=12,
                 rng_seed=0, projpred_V=None, max_workers=None):
    """High-level entry point: fit horseshoe logistic regression from a DataFrame.

    Extracts arrays, estimates ``scale_global`` if needed, fits the
    model, and runs in-sample diagnostics.  Optionally runs
    cross-validation and projection predictive variable selection.

    Parameters
    ----------
    df : DataFrame
        Data containing outcome and covariates.
    y_col : str
        Name of the binary outcome column.
    unpenalized_cols : list of str
        Columns for unpenalized covariates.  An intercept column of ones
        is always prepended automatically.
    penalized_cols : list of str
        Columns for penalized (horseshoe) covariates.
    filestem : str
        Prefix for all output files (CSV summaries and PDF plots).
    slab_scale : float
        Regularizing slab scale.
    slab_df : float
        Slab inverse-gamma degrees of freedom.
    p0 : int or None
        Prior guess for the number of penalized covariates with nonzero
        effects.  Used to compute ``scale_global`` when it is not given
        explicitly.  If None, defaults to ``max(1, J // 4)`` where J is
        the number of penalized covariates.
    scale_global : float or None
        Global shrinkage scale.  If None, estimated as
        ``p0 / (J - p0) / sqrt(N * p * (1 - p))`` where *p* is the
        proportion of cases.
    standardize : bool
        If True (default), centre and scale the penalized covariates to
        zero mean and unit variance before fitting.  Posterior summaries
        report coefficients on the standardized scale.
    sampler : {"nuts", "mclmc"}
        Sampling algorithm.
    crossvalidate_ : bool
        If True, run only cross-validation: a learning curve (K=2..5)
        and 5-fold CV.  The full model fit, posterior summaries, and
        projpred are skipped.
    num_warmup : int
        Warmup iterations per chain.
    num_samples : int
        Posterior samples per chain.
    num_chains : int
        Number of MCMC chains.
    target_accept_prob : float
        NUTS target acceptance probability.
    max_tree_depth : int
        NUTS maximum tree depth.
    rng_seed : int
        Random seed.
    projpred_V : int or None
        If not None, run projection predictive forward search selecting
        up to this many variables.
    max_workers : int or None
        Maximum parallel workers for cross-validation.

    Returns
    -------
    dict
        When ``crossvalidate_=False`` (default): ``result``, ``N``,
        ``n_controls``, ``n_cases``, ``unpenalized_cols``,
        ``penalized_cols``, ``beta_hat``, ``m_eff``, ``insample``,
        ``projpred`` (or None).
        When ``crossvalidate_=True``: ``N``, ``n_controls``,
        ``n_cases``, ``unpenalized_cols``, ``penalized_cols``, ``cv``.
    """
    # --- 1. Extract arrays ---
    used_cols = [y_col] + list(unpenalized_cols) + list(penalized_cols)
    N_before = len(df)
    df = df[used_cols].dropna()
    N_after = len(df)
    if N_after < N_before:
        print(f"Dropped {N_before - N_after} rows with missing values "
              f"({N_before} -> {N_after})")

    y = df[y_col].values.astype(np.float32)
    X = df[penalized_cols].values.astype(np.float32)

    if unpenalized_cols:
        X_u_raw = df[unpenalized_cols].values.astype(np.float32)
    else:
        X_u_raw = np.empty((len(y), 0), dtype=np.float32)

    # Drop rows with non-finite values (Inf/-Inf/NaN from float32 overflow)
    finite_mask = (np.isfinite(y)
                   & np.all(np.isfinite(X), axis=1)
                   & np.all(np.isfinite(X_u_raw), axis=1))
    n_nonfinite = int((~finite_mask).sum())
    if n_nonfinite > 0:
        print(f"Dropped {n_nonfinite} rows with non-finite values "
              f"(Inf/NaN after float32 conversion)")
        y = y[finite_mask]
        X = X[finite_mask]
        X_u_raw = X_u_raw[finite_mask]

    # Validate y contains only 0 and 1
    unique_y = np.unique(y)
    if not np.all((unique_y == 0) | (unique_y == 1)):
        raise ValueError(
            f"y_col '{y_col}' must be binary (0/1), "
            f"got unique values: {unique_y}")

    N, J = X.shape

    # Standardize penalized covariates
    if standardize:
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1.0  # avoid division by zero for constant cols
        X = (X - X_mean) / X_std
        print("Penalized covariates standardized to zero mean, unit variance")

    intercept_col = np.ones((N, 1), dtype=np.float32)
    if unpenalized_cols:
        X_u = np.hstack([intercept_col, X_u_raw])
    else:
        X_u = intercept_col
    U = X_u.shape[1]

    print(f"N={N}, U={U} (incl intercept), J={J}")

    # --- 2. Default scale_global ---
    if scale_global is None:
        if p0 is None:
            p0 = max(1, J // 4)
        p_cases = float(y.mean())
        scale_global = p0 / (J - p0) / np.sqrt(N * p_cases * (1 - p_cases))
        print(f"scale_global estimated: p0={p0}, scale_global={scale_global:.4f}")

    n_cases = int(y.sum())
    n_controls = int(len(y) - n_cases)

    if crossvalidate_:
        # --- Cross-validation only: learning curve + K-fold CV ---
        train_sizes, info_values = learning_curve(
            X_u, X, y, K_values=(2, 3, 4, 5),
            slab_scale=slab_scale, slab_df=slab_df, scale_global=scale_global,
            num_warmup=num_warmup, num_samples=num_samples,
            num_chains=num_chains, target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth, rng_seed=rng_seed,
            sampler=sampler, max_workers=max_workers,
        )
        plot_learning_curve(train_sizes, info_values, filestem)

        cv_result = crossvalidate(
            X_u, X, y, K=5,
            slab_scale=slab_scale, slab_df=slab_df, scale_global=scale_global,
            num_warmup=num_warmup, num_samples=num_samples,
            num_chains=num_chains, target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth, rng_seed=rng_seed,
            sampler=sampler, max_workers=max_workers,
        )
        plot_wevid(cv_result["wevid"], filestem + "_cv")

        return {
            "N": N, "n_controls": n_controls, "n_cases": n_cases,
            "unpenalized_cols": ["Intercept"] + list(unpenalized_cols),
            "penalized_cols": list(penalized_cols),
            "cv": cv_result,
        }

    # --- 3. Fit full model ---
    result = fit(
        jnp.array(X_u), jnp.array(X), jnp.array(y),
        slab_scale=slab_scale, slab_df=slab_df, scale_global=scale_global,
        num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
        target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth,
        rng_seed=rng_seed, sampler=sampler, print_summary=False,
    )

    # --- 4. In-sample diagnostics ---
    is_nuts = sampler == "nuts"

    unpenalized_names = ["Intercept"] + list(unpenalized_cols)
    if is_nuts:
        summary_report(result, filestem + "_summary.csv",
                       unpenalized_names=unpenalized_names,
                       penalized_names=list(penalized_cols))

    # posterior mean betas and forest plot
    samples = result.get_samples()
    beta_hat = np.array(samples["beta"].mean(axis=0))
    plot_forest(samples["beta"], list(penalized_cols), filestem)

    # effective nonzero coefficients (m_eff)
    tau = samples["tau"][:, None]
    eta = samples["eta"][:, None]
    lambda_raw = samples["aux1_local"] * jnp.sqrt(samples["aux2_local"])
    lambda_tilde_sq = (eta**2 * lambda_raw**2) / (eta**2 + tau**2 * lambda_raw**2)
    kappa = 1.0 / (1.0 + tau**2 * lambda_tilde_sq)
    m_eff = (1.0 - kappa).sum(axis=1)
    print(f"\nPosterior m_eff:  mean={float(m_eff.mean()):.2f}  "
          f"median={float(jnp.median(m_eff)):.2f}  "
          f"90% CI=[{float(jnp.percentile(m_eff, 5)):.2f}, "
          f"{float(jnp.percentile(m_eff, 95)):.2f}]")

    # in-sample predictions
    probs = predict(
        result, jnp.array(X_u), jnp.array(X),
        slab_scale=slab_scale, slab_df=slab_df, scale_global=scale_global,
        rng_seed=rng_seed + 1,
    )
    probs_mean = np.array(probs.mean(axis=0))
    c_stat = cstatistic(y, probs_mean)
    w = Wdensities(y, probs_mean, recalibrate=True)
    info_discrim = get_info_discrim(w)
    lscore = log_score(y, probs_mean)

    print(f"\nIn-sample (N={N}):")
    print(f"  C-statistic                            = {c_stat:.3f}")
    print(f"  Expected information for discrimination = {info_discrim} bits")
    print(f"  Logarithmic score                      = {lscore:.3f}")

    insample = {"c_stat": c_stat, "info_discrim": info_discrim,
                "log_score": lscore, "wevid": w}

    if is_nuts:
        plot_pair_diagnostic(result, filestem)
    plot_wevid(w, filestem)

    # --- 5. Projpred ---
    projpred_result = None
    if projpred_V is not None:
        print("\n" + "=" * 60)
        print("Projection predictive forward search")
        print("=" * 60)
        selected, kl_path, kl_null = projpred_forward_search(
            result, X_u, X, V=projpred_V,
        )
        print(f"\nSelected covariates (in order):")
        for i, j in enumerate(selected):
            print(f"  {i+1}. {penalized_cols[j]} (index {j})")
        plot_projpred(selected, kl_path, kl_null, filestem,
                      var_names=list(penalized_cols))
        projpred_result = {"selected": selected, "kl_path": kl_path,
                           "kl_null": kl_null,
                           "selected_names": [penalized_cols[j] for j in selected]}

    # --- 6. Return ---
    return {
        "result": result,
        "N": N, "n_controls": n_controls, "n_cases": n_cases,
        "unpenalized_cols": ["Intercept"] + list(unpenalized_cols),
        "penalized_cols": list(penalized_cols),
        "beta_hat": beta_hat, "m_eff": np.array(m_eff),
        "insample": insample, "projpred": projpred_result,
    }


