
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import logit
from scipy.optimize import minimize_scalar

import jax
import jax.numpy as jnp
import numpyro as npyr
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

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


def fit(X_u, X, y, slab_scale=1.0, slab_df=4.0, scale_global=1.0,
        num_warmup=1000, num_samples=1000, num_chains=4,
        target_accept_prob=0.95, max_tree_depth=12, rng_seed=0):
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
    mcmc.run(
        jax.random.PRNGKey(rng_seed),
        X_u=X_u, X=X, y=y,
        slab_scale=slab_scale, slab_df=slab_df, scale_global=scale_global,
    )
    mcmc.print_summary()
    return mcmc


def predict(mcmc, X_u_new, X_new, slab_scale=1.0, slab_df=4.0,
            scale_global=1.0, rng_seed=1):
    posterior_samples = mcmc.get_samples()
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


def get_wevid(y, predicted_y):
    unique, counts = np.unique(y, return_counts=True)
    y_counts = dict(zip(unique, counts))
    n_ctrls = y_counts[0]
    n_cases = y_counts[1]
    logodds_prior = logit(n_cases / (n_ctrls + n_cases))

    W = logit(predicted_y) - logodds_prior
    W_df = pd.DataFrame({'y': y, 'W': W})
    return wevid(W_df, n_ctrls, n_cases)


def get_info_discrim(w):
    info_discrim = ((w["xseq"] * w["f_cases"]).sum()
                    - (w["xseq"] * w["f_ctrls"]).sum()) * w["x_stepsize"] * 0.5 / np.log(2)
    return round(info_discrim, 2)


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

    # generate test dataset from the same DGP
    N_test = 500
    X_test = np.random.randn(N_test, J).astype(np.float32)
    X_u_test = np.ones((N_test, U), dtype=np.float32)
    logits_test = intercept + X_test @ beta_true
    y_test = np.random.binomial(1, 1.0 / (1.0 + np.exp(-logits_test))).astype(np.float32)

    # posterior predictive probabilities on test set
    probs_test = predict(
        mcmc, jnp.array(X_u_test), jnp.array(X_test),
        slab_scale=2.0, slab_df=4.0, scale_global=scale_global,
    )
    mean_probs_test = probs_test.mean(axis=0)

    c_stat = cstatistic(y_test, mean_probs_test)
    w = get_wevid(np.array(y_test), np.array(mean_probs_test))
    info_discrim = get_info_discrim(w)
    print(f"\nTest set (N={N_test}):")
    print(f"  C-statistic                        = {c_stat:.3f}")
    print(f"  Expected information for discrimination = {info_discrim} bits")

    outpath = plot_wevid(w, "hslogistic")
    print(f"\nPlot saved to {outpath}")
