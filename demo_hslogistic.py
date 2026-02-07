#!/usr/bin/env python3
"""Demo: horseshoe logistic regression on simulated sparse data."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import jax.numpy as jnp

import hslogistic as hs


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

    # Piironen & Vehtari recommended scale_global ~ p0 / (J - p0) / sqrt(N)
    p0 = 3  # expected number of active covariates
    scale_global = p0 / (J - p0) / np.sqrt(N)

    print(f"N={N}, J={J}, p0={p0}, scale_global={scale_global:.4f}")
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
    p0_true = int((beta_true != 0).sum())
    print(f"\nTrue number of nonzero coefficients: {p0_true}")
    print(f"Posterior m_eff:  mean={float(m_eff.mean()):.2f}  "
          f"median={float(jnp.median(m_eff)):.2f}  "
          f"90% CI=[{float(jnp.percentile(m_eff, 5)):.2f}, "
          f"{float(jnp.percentile(m_eff, 95)):.2f}]")

    # Learning curve: info for discrimination vs training size
    train_sizes, info_values = hs.learning_curve(
        X_u, X, y, K_values=(2, 3, 4, 5),
        slab_scale=2.0, slab_df=4.0, scale_global=scale_global,
        num_warmup=500, num_samples=500, num_chains=2, rng_seed=0,
    )
    outpath3 = hs.plot_learning_curve(train_sizes, info_values, "hslogistic")
    print(f"Plot saved to {outpath3}")

    outpath2 = hs.plot_pair_diagnostic(mcmc, "hslogistic")
    print(f"Plot saved to {outpath2}")

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
        slab_scale=2.0, slab_df=4.0,
        num_warmup=500, num_samples=500, num_chains=2,
        rng_seed=0, projpred_V=5,
    )
    print(f"\nrun_analysis returned keys: {sorted(out.keys())}")


if __name__ == "__main__":
    main()
