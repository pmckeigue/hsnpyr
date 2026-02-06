
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az
from scipy import stats
from scipy.special import logit
from scipy.optimize import minimize_scalar, curve_fit
from sklearn.linear_model import LogisticRegression

import jax
import jax.numpy as jnp
import numpyro as npyr
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.infer.util import initialize_model
import blackjax

def hslogistic(X_u=None, X=None, y=None, slab_scale=None, slab_df=None, scale_global=None):
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
        sampler="nuts"):
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
        mcmc.print_summary()
        return mcmc
    elif sampler == "mclmc":
        return _fit_mclmc(model_kwargs, num_warmup, num_samples, rng_seed)
    else:
        raise ValueError(f"Unknown sampler: {sampler!r}. Use 'nuts' or 'mclmc'.")


def _fit_mclmc(model_kwargs, num_warmup, num_samples, rng_seed):
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
    wdens = reweight_densities(theta, n_ctrls, n_cases, f0, f1, xseq, wts)
    obj = abs(np.log(np.sum(wdens["f_ctrls"]) / np.sum(wdens["f_cases"])))
    return obj


def wevid(W_df, n_ctrls, n_cases):
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
    eps = 1e-8
    probs = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    logit_p = logit(probs).reshape(-1, 1)
    model = LogisticRegression(C=np.inf, solver='lbfgs')
    model.fit(logit_p, np.asarray(y))
    return model.predict_proba(logit_p)[:, 1]


def Wdensities(y, predicted_y, recalibrate=True):
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
    info_discrim = ((w["xseq"] * w["f_cases"]).sum()
                    - (w["xseq"] * w["f_ctrls"]).sum()) * w["x_stepsize"] * 0.5 / np.log(2)
    return round(info_discrim, 2)


def log_score(y, probs):
    eps = 1e-15
    probs = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0 - eps)
    y = np.asarray(y)
    return float(np.sum(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs)))


def crossvalidate(X_u, X, y, K=5, slab_scale=1.0, slab_df=4.0,
                  scale_global=1.0, num_warmup=1000, num_samples=1000,
                  num_chains=4, target_accept_prob=0.95, max_tree_depth=12,
                  rng_seed=0, sampler="nuts"):
    X_u = np.asarray(X_u)
    X = np.asarray(X)
    y = np.asarray(y)
    N = X.shape[0]
    indices = np.arange(N)
    rng = np.random.RandomState(rng_seed)
    rng.shuffle(indices)
    folds = np.array_split(indices, K)

    all_y = []
    all_probs = []
    for k, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(K) if j != k])
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
        all_y.append(y[test_idx])
        all_probs.append(np.array(probs_k.mean(axis=0)))

    all_y = np.concatenate(all_y)
    all_probs = np.concatenate(all_probs)

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
                   max_tree_depth=12, rng_seed=0, sampler="nuts"):
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
            sampler=sampler,
        )
        train_sizes.append(n_train)
        info_values.append(cv["info_discrim"])
    return np.array(train_sizes), np.array(info_values)


def plot_learning_curve(train_sizes, info_values, filestem):
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
    plt.close()

    print(f"\nFitted curve: Lambda(n) = {a_fit:.2f} * n / ({b_fit:.0f} + n)")
    print(f"  Asymptote (a) = {a_fit:.2f} bits")
    print(f"  Half-max (b)  = {b_fit:.0f} observations")
    return outpath


def plot_pair_diagnostic(mcmc, filestem):
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
    plt.close(fig)
    return outpath


def plot_wevid(w, filestem):
    plt.figure()
    plt.plot(w["xseq"], w["f_ctrls"], label="controls")
    plt.plot(w["xseq"], w["f_cases"], label="cases")
    plt.xlim(-10, 10)
    plt.xlabel("Weight of evidence favouring case over control (nat log units)")
    plt.ylabel("Density")
    plt.legend()
    outpath = filestem + "_wevid_dist.pdf"
    plt.savefig(outpath)
    plt.close()
    return outpath


if __name__ == "__main__":

    np.random.seed(42)
    N = 200
    J = 20  # penalized covariates (most will be noise)
    U = 1   # unpenalized (intercept)

    # true coefficients: only 3 of 20 penalized covariates are active
    beta_true = np.zeros(J)
    beta_true[0] = 2.0
    beta_true[1] = -1.5
    beta_true[2] = 1.0
    intercept = -0.5

    X = np.random.randn(N, J).astype(np.float32)
    X_u = np.ones((N, U), dtype=np.float32)  # intercept column
    logits = intercept + X @ beta_true
    y = np.random.binomial(1, 1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    # Piironen & Vehtari recommended scale_global ~ p0 / (J - p0) / sqrt(N)
    p0 = 3  # expected number of active covariates
    scale_global = p0 / (J - p0) / np.sqrt(N)

    print(f"N={N}, J={J}, p0={p0}, scale_global={scale_global:.4f}")
    print(f"True nonzero betas: {beta_true[beta_true != 0]}")
    print(f"Observed y mean: {y.mean():.3f}")
    print()

    mcmc = fit(
        jnp.array(X_u), jnp.array(X), jnp.array(y),
        slab_scale=2.0, slab_df=4.0, scale_global=scale_global,
        num_warmup=500, num_samples=500, num_chains=2, rng_seed=0,
    )

    # posterior mean of penalized betas
    beta_samples = mcmc.get_samples()["beta"]
    beta_hat = beta_samples.mean(axis=0)
    print("\nTrue vs estimated penalized betas:")
    for j in range(J):
        flag = " *" if beta_true[j] != 0 else ""
        print(f"  beta[{j:2d}]: true={beta_true[j]:+6.2f}  est={beta_hat[j]:+6.3f}{flag}")

    # effective number of nonzero coefficients (Piironen & Vehtari 2017)
    samples = mcmc.get_samples()
    tau = samples["tau"][:, None]                          # (S, 1)
    eta = samples["eta"][:, None]                          # (S, 1)
    lambda_raw = samples["aux1_local"] * jnp.sqrt(samples["aux2_local"])  # (S, J)
    lambda_tilde_sq = (eta**2 * lambda_raw**2) / (eta**2 + tau**2 * lambda_raw**2)
    kappa = 1.0 / (1.0 + tau**2 * lambda_tilde_sq)        # shrinkage factors (S, J)
    m_eff = (1.0 - kappa).sum(axis=1)                      # (S,)
    p0_true = int((beta_true != 0).sum())
    print(f"\nTrue number of nonzero coefficients: {p0_true}")
    print(f"Posterior m_eff:  mean={float(m_eff.mean()):.2f}  "
          f"median={float(jnp.median(m_eff)):.2f}  "
          f"90% CI=[{float(jnp.percentile(m_eff, 5)):.2f}, "
          f"{float(jnp.percentile(m_eff, 95)):.2f}]")

    # Learning curve: info for discrimination vs training size
    train_sizes, info_values = learning_curve(
        X_u, X, y, K_values=(2, 3, 4, 5),
        slab_scale=2.0, slab_df=4.0, scale_global=scale_global,
        num_warmup=500, num_samples=500, num_chains=2, rng_seed=0,
    )
    outpath3 = plot_learning_curve(train_sizes, info_values, "hslogistic")
    print(f"Plot saved to {outpath3}")

    outpath2 = plot_pair_diagnostic(mcmc, "hslogistic")
    print(f"Plot saved to {outpath2}")
