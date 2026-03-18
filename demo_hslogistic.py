#!/usr/bin/env python3
"""Demo: horseshoe logistic regression on simulated sparse data."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import jax.numpy as jnp

import hsnpyr as hs


def main():
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

    # scale_global = f0/(1-f0) / sqrt(I), where I = N*p*(1-p) for logistic regression
    fraction_nonzero = 0.2  # prior guess: 20% of covariates have nonzero effects
    p_cases = float(y.mean())
    scale_global = fraction_nonzero / (1 - fraction_nonzero) / np.sqrt(N * p_cases * (1 - p_cases))

    print(f"N={N}, J={J}, fraction_nonzero={fraction_nonzero}, scale_global={scale_global:.4f}")
    print(f"True nonzero betas: {beta_true[beta_true != 0]}")
    print(f"Observed y mean: {y.mean():.3f}")
    print()

    mcmc = hs.fit(
        jnp.array(X_u), jnp.array(X), jnp.array(y),
        slab_scale=2.0, slab_df=4.0, scale_global=scale_global,
        num_warmup=500, num_samples=500, num_chains=2, rng_seed=0,
    )

    # summary report for full-dataset fit
    hs.summary_report(mcmc, "hslogistic_summary.csv")

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
    f_eff = m_eff / J
    f_true = float((beta_true != 0).sum()) / J
    print(f"\nTrue fraction of nonzero coefficients: {f_true:.3f}")
    print(f"Posterior f_eff:  mean={float(f_eff.mean()):.3f}  "
          f"median={float(jnp.median(f_eff)):.3f}  "
          f"90% CI=[{float(jnp.percentile(f_eff, 5)):.3f}, "
          f"{float(jnp.percentile(f_eff, 95)):.3f}]")

    # Learning curve: info for discrimination vs training size
    train_sizes, info_values, cv_results = hs.learning_curve(
        X_u, X, y, K_values=(2, 3, 4, 5),
        slab_scale=2.0, slab_df=4.0, scale_global=scale_global,
        num_warmup=500, num_samples=500, num_chains=2, rng_seed=0,
    )
    outpath3 = hs.plot_learning_curve(train_sizes, info_values, "hslogistic")
    print(f"Plot saved to {outpath3}")

    outpath2 = hs.plot_pair_diagnostic(mcmc, "hslogistic")
    print(f"Plot saved to {outpath2}")

    cv_result = cv_results[-1]  # reuse the K=5 result from the learning curve
    outpath1 = hs.plot_wevid(cv_result["wevid"], "hslogistic_cv")
    print(f"Plot saved to {outpath1}")

    # Projection predictive forward search
    print("\n" + "=" * 60)
    print("Projection predictive forward search")
    print("=" * 60)
    selected, kl_path, kl_null = hs.projpred_forward_search(mcmc, X_u, X, V=5)
    print(f"\nSelected covariates (in order): {selected}")
    true_active = set(np.where(beta_true != 0)[0])
    print(f"True active covariates: {sorted(true_active)}")
    print(f"KL null (intercept only): {kl_null:.6f}")
    print(f"KL path: {[f'{k:.6f}' for k in kl_path]}")
    hs.plot_projpred(selected, kl_path, kl_null, "hslogistic")

    # ---- MCLMC sampler ----
    print("\n" + "=" * 60)
    print("MCLMC sampler demo")
    print("=" * 60)

    mclmc_result = hs.fit(
        jnp.array(X_u), jnp.array(X), jnp.array(y),
        slab_scale=2.0, slab_df=4.0, scale_global=scale_global,
        num_warmup=500, num_samples=10_000, num_chains=2,
        rng_seed=0, sampler="mclmc",
    )

    hs.summary_report(mclmc_result, "hslogistic_mclmc_summary.csv")

    beta_mclmc = mclmc_result.get_samples()["beta"].mean(axis=0)
    print("\nMCLMC vs NUTS penalized betas:")
    for j in range(J):
        flag = " *" if beta_true[j] != 0 else ""
        print(f"  beta[{j:2d}]: true={beta_true[j]:+6.2f}  "
              f"NUTS={beta_hat[j]:+6.3f}  MCLMC={beta_mclmc[j]:+6.3f}{flag}")

    hs.plot_mclmc_tuning_traces(mclmc_result, "hslogistic_mclmc")

    # ---- run_analysis: high-level DataFrame interface ----
    print("\n" + "=" * 60)
    print("run_analysis demo (DataFrame interface)")
    print("=" * 60)

    penalized_names = [f"x{j}" for j in range(J)]
    data = {col: X[:, j] for j, col in enumerate(penalized_names)}
    data["outcome"] = y
    df = pd.DataFrame(data)

    out = hs.run_analysis(
        df, y_col="outcome",
        unpenalized_cols=[],
        penalized_cols=penalized_names,
        filestem="hslogistic_ra",
        slab_scale=2.0, slab_df=4.0, fraction_nonzero=0.2,
        num_warmup=500, num_samples=500, num_chains=2,
        rng_seed=0, projpred_V=5,
    )
    print(f"\nrun_analysis returned keys: {sorted(out.keys())}")


if __name__ == "__main__":
    main()
