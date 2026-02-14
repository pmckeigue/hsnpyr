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
    "plot_wevid", "plot_forest", "plot_pairs", "plot_trace", "plot_autocorr",
    "plot_mclmc_tuning_traces", "plot_projpred",
    "sample_matched_controls",
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
    """Wrapper so that MCLMC results have the same interface as MCMC.

    Samples are stored with shape ``(num_chains, num_samples, ...)``
    per site so that ``get_samples(group_by_chain=True)`` works for
    computing r_hat and n_eff.
    """
    def __init__(self, samples, num_chains=1, tuning_results=None,
                 tuning_selected_idx=None):
        self._samples = samples      # (num_chains, num_samples, ...) per site
        self._num_chains = num_chains
        self.tuning_results = tuning_results
        self.tuning_selected_idx = tuning_selected_idx

    def get_samples(self, group_by_chain=False):
        if group_by_chain:
            return self._samples
        # Flatten chain dimension: (num_chains * num_samples, ...)
        return jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]),
            self._samples,
        )


def fit(X_u, X, y, slab_scale=1.0, slab_df=4.0, scale_global=1.0,
        num_warmup=1000, num_samples=None, num_chains=4,
        target_accept_prob=0.95, max_tree_depth=12, rng_seed=0,
        sampler="nuts", thin=None, print_summary=True):
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
    num_samples : int or None
        Number of posterior samples per chain.  Defaults to 50000 for
        MCLMC and 1000 for NUTS when None.
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
    thin : int or None
        Thinning factor applied to MCLMC samples (every *thin*-th sample
        is retained).  Defaults to 5 for MCLMC and 1 for NUTS when None.
        Ignored for NUTS.
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
    # Sampler-specific defaults
    if num_samples is None:
        num_samples = 50_000 if sampler == "mclmc" else 1000
    if thin is None:
        thin = 5 if sampler == "mclmc" else 1
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
        return _fit_mclmc(model_kwargs, num_warmup, num_samples, num_chains,
                          rng_seed, thin=thin)
    else:
        raise ValueError(f"Unknown sampler: {sampler!r}. Use 'nuts' or 'mclmc'.")


def _mclmc_find_L_and_step_size_with_trace(
    mclmc_kernel, num_steps, state, rng_key,
    frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.1,
    desired_energy_var=5e-4, trust_in_estimate=1.5,
    num_effective_samples=150, diagonal_preconditioning=True,
):
    """Like ``blackjax.mclmc_find_L_and_step_size`` but returns per-step traces.

    Returns
    -------
    state : MCLMCState
    params : MCLMCAdaptationState
    total_steps : int
    trace : dict
        ``step_size``, ``L``, ``energy_var`` arrays (one entry per tuning step).
    """
    from blackjax.adaptation.mclmc_adaptation import (
        MCLMCAdaptationState, handle_nans,
    )
    from blackjax.util import (
        generate_unit_vector, incremental_value_update, pytree_size,
    )
    from blackjax.diagnostics import effective_sample_size as bjx_ess
    from jax.flatten_util import ravel_pytree

    dim = pytree_size(state.position)
    params = MCLMCAdaptationState(
        jnp.sqrt(dim), jnp.sqrt(dim) * 0.25,
        inverse_mass_matrix=jnp.ones((dim,)),
    )

    part1_key, part2_key = jax.random.split(rng_key, 2)
    num_steps1 = round(num_steps * frac_tune1)
    num_steps2 = round(num_steps * frac_tune2)
    num_steps2_extra = num_steps2 if diagonal_preconditioning else 0
    num_steps3 = round(num_steps * frac_tune3)

    decay_rate = (num_effective_samples - 1.0) / (num_effective_samples + 1.0)

    # -- predictor: one MCMC step + step-size update --
    def predictor(previous_state, params, adaptive_state, rng_key):
        time, x_average, step_size_max = adaptive_state
        rng_key, nan_key = jax.random.split(rng_key)
        next_state, info = mclmc_kernel(params.inverse_mass_matrix)(
            rng_key=rng_key, state=previous_state,
            L=params.L, step_size=params.step_size,
        )
        success, state, step_size_max, energy_change = handle_nans(
            previous_state, next_state, params.step_size,
            step_size_max, info.energy_change, nan_key,
        )
        xi = (jnp.square(energy_change) / (dim * desired_energy_var)) + 1e-8
        weight = jnp.exp(
            -0.5 * jnp.square(jnp.log(xi) / (6.0 * trust_in_estimate))
        )
        x_average = decay_rate * x_average + weight * (
            xi / jnp.power(params.step_size, 6.0)
        )
        time = decay_rate * time + weight
        step_size = jnp.power(x_average / time, -1.0 / 6.0)
        step_size = jnp.where(step_size < step_size_max,
                              step_size, step_size_max)
        params_new = params._replace(step_size=step_size)
        adaptive_state = (time, x_average, step_size_max)
        return state, params_new, adaptive_state, success, energy_change

    # -- scan step for stages 1 & 2: returns trace row --
    def step12(iteration_state, weight_and_key):
        mask, rng_key = weight_and_key
        state, params, adaptive_state, streaming_avg = iteration_state
        state, params, adaptive_state, success, energy_change = predictor(
            state, params, adaptive_state, rng_key,
        )
        x = ravel_pytree(state.position)[0]
        streaming_avg = incremental_value_update(
            expectation=jnp.array([x, jnp.square(x)]),
            incremental_val=streaming_avg,
            weight=mask * success * params.step_size,
        )
        trace_row = (params.step_size, params.L, jnp.square(energy_change))
        return (state, params, adaptive_state, streaming_avg), trace_row

    def run_steps12(xs, state, params):
        carry, traces = jax.lax.scan(
            step12,
            init=(
                state, params,
                (0.0, 0.0, jnp.inf),
                (0.0, jnp.array([jnp.zeros(dim), jnp.zeros(dim)])),
            ),
            xs=xs,
        )
        return carry, traces  # traces = (step_size[], L[], evar[])

    # --- Stage 1 + 2 ---
    n12 = num_steps1 + num_steps2
    keys12 = jax.random.split(part1_key, n12 + 1)
    keys12, final_key = keys12[:-1], keys12[-1]
    mask = jnp.concatenate((jnp.zeros(num_steps1), jnp.ones(num_steps2)))

    (state, params, _, (_, average)), traces12 = run_steps12(
        xs=(mask, keys12), state=state, params=params,
    )
    all_ss = [traces12[0]]
    all_L = [traces12[1]]
    all_ev = [traces12[2]]

    # Compute L and diagonal preconditioner
    inverse_mass_matrix = params.inverse_mass_matrix
    L = params.L
    if num_steps2 > 1:
        x_average, x_squared_average = average[0], average[1]
        variances = x_squared_average - jnp.square(x_average)
        L = jnp.sqrt(jnp.sum(variances))
        if diagonal_preconditioning:
            inverse_mass_matrix = variances
            params = params._replace(inverse_mass_matrix=inverse_mass_matrix)
            L = jnp.sqrt(dim)
            # Readjust step size with new preconditioner
            steps_extra = num_steps2_extra
            if steps_extra > 0:
                keys_extra = jax.random.split(final_key, steps_extra)
                (state, params, _, _), traces_extra = run_steps12(
                    xs=(jnp.ones(steps_extra), keys_extra),
                    state=state, params=params,
                )
                all_ss.append(traces_extra[0])
                all_L.append(traces_extra[1])
                all_ev.append(traces_extra[2])

    params = MCLMCAdaptationState(L, params.step_size, inverse_mass_matrix)

    # --- Stage 3: L from ESS ---
    if num_steps3 >= 2:
        adaptation_L_keys = jax.random.split(part2_key, num_steps3)
        built_kernel = mclmc_kernel(params.inverse_mass_matrix)

        def step3(state, key):
            next_state, info = built_kernel(
                rng_key=key, state=state,
                L=params.L, step_size=params.step_size,
            )
            trace_row = (params.step_size, params.L,
                         jnp.square(info.energy_change))
            return next_state, (next_state.position, trace_row)

        state, (samples3, traces3) = jax.lax.scan(
            f=step3, init=state, xs=adaptation_L_keys,
        )
        all_ss.append(traces3[0])
        all_L.append(traces3[1])
        all_ev.append(traces3[2])

        flat_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(samples3)
        del samples3
        ess = bjx_ess(flat_samples[None, ...])
        del flat_samples
        params = params._replace(
            L=0.4 * params.step_size * jnp.mean(num_steps3 / ess),
        )

    trace = {
        "step_size": jnp.concatenate(all_ss),
        "L": jnp.concatenate(all_L),
        "energy_var": jnp.concatenate(all_ev),
        "stage_boundaries": (num_steps1,
                             num_steps1 + num_steps2 + num_steps2_extra),
    }
    total_steps = num_steps1 + num_steps2 + num_steps2_extra + (
        num_steps3 if num_steps3 >= 2 else 0)
    return state, params, total_steps, trace


def _fit_mclmc(model_kwargs, num_warmup, num_samples, num_chains, rng_seed,
               thin=1):
    """Fit using the MCLMC sampler via BlackJAX.

    Runs 5 tuning attempts and selects the median step size for
    robustness, then runs ``num_chains`` independent sampling chains.  Uses a Python-level sampling loop (instead of
    ``jax.lax.scan``) to avoid excessive XLA compilation time with
    high-dimensional models.

    Parameters
    ----------
    model_kwargs : dict
        Arguments forwarded to the NumPyro model.
    num_warmup : int
        Tuning steps for step-size and trajectory-length selection.
    num_samples : int
        Number of posterior samples to draw per chain.
    num_chains : int
        Number of independent chains.
    rng_seed : int
        Random seed.
    thin : int
        Thinning factor: keep every *thin*-th sample.

    Returns
    -------
    _SamplesResult
        Wrapper with samples shaped ``(num_chains, num_retained, ...)``.
    """
    import time
    import sys
    from tqdm.auto import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    NUM_TUNE_RUNS = 5

    rng_key = jax.random.PRNGKey(rng_seed)
    init_key, tune_key, run_key = jax.random.split(rng_key, 3)

    # --- 1. Initialize model ---
    t0 = time.time()
    print("MCLMC: initializing model...", flush=True)
    init_params, potential_fn_gen, postprocess_fn, _ = initialize_model(
        init_key, hslogistic,
        model_kwargs=model_kwargs,
        dynamic_args=False,
    )
    logdensity_fn = lambda position: -potential_fn_gen(position)
    initial_position = init_params.z
    del init_params  # free initialization data; only position is needed
    n_params = sum(v.size for v in jax.tree.leaves(initial_position))
    print(f"MCLMC: {n_params} unconstrained parameters "
          f"({time.time() - t0:.1f}s)", flush=True)

    # --- 2. Initialize MCLMC state ---
    t0 = time.time()
    print("MCLMC: computing initial state...", flush=True)
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position,
        logdensity_fn=logdensity_fn,
        rng_key=init_key,
    )
    del initial_position, init_key  # no longer needed
    print(f"MCLMC: initial logdensity = {float(initial_state.logdensity):.2f} "
          f"({time.time() - t0:.1f}s)", flush=True)

    # --- 3. Tune L and step_size (parallel runs, median selection) ---
    tune_steps = max(num_warmup, 3 * n_params)
    t0 = time.time()
    print(f"MCLMC: tuning ({tune_steps} steps x {NUM_TUNE_RUNS} runs, "
          f"diagonal preconditioning)...", flush=True)
    kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
        logdensity_fn=logdensity_fn,
        integrator=blackjax.mcmc.integrators.isokinetic_mclachlan,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    tune_keys = jax.random.split(tune_key, NUM_TUNE_RUNS)

    def _run_one_tuning(idx):
        """Run a single tuning attempt; return a result dict."""
        try:
            (state, params, _, trace) = _mclmc_find_L_and_step_size_with_trace(
                mclmc_kernel=kernel,
                num_steps=tune_steps,
                state=initial_state,
                rng_key=tune_keys[idx],
                diagonal_preconditioning=True,
            )
            L_val = float(params.L)
            step_val = float(params.step_size)
            ld_val = float(state.logdensity)
            valid = (step_val > 0 and np.isfinite(step_val)
                     and np.isfinite(ld_val))
            return {
                "idx": idx, "L": L_val, "step_size": step_val,
                "logdensity": ld_val, "valid": valid,
                "state": state, "sampler_params": params,
                "trace": trace,
            }
        except Exception as e:
            return {
                "idx": idx, "L": float("nan"), "step_size": float("nan"),
                "logdensity": float("nan"), "valid": False,
                "state": None, "sampler_params": None, "error": str(e),
            }

    # Estimate memory per tuning run (rough upper bound)
    # Stage 3 dominates: samples3 pytree ~ num_steps3 x dim, flat_samples ~ same
    # Plus scan trace arrays, streaming averages, state, kernel intermediates
    _num_steps3_est = round(tune_steps * 0.1)
    _est_tune_bytes = (10 * n_params + 2 * _num_steps3_est * n_params) * 4

    # Run sequentially with memory reporting and cleanup
    import gc
    tuning_results = []
    for i in range(NUM_TUNE_RUNS):
        gc.collect()
        jax.clear_caches()
        mem_avail = _get_available_memory_bytes()
        if mem_avail is not None:
            print(f"  tuning run {i+1}/{NUM_TUNE_RUNS}: "
                  f"{mem_avail / 1e9:.1f} GB available, "
                  f"~{_est_tune_bytes / 1e6:.0f} MB estimated per run",
                  flush=True)
        else:
            print(f"  tuning run {i+1}/{NUM_TUNE_RUNS}: "
                  f"memory info unavailable", flush=True)
        tuning_results.append(_run_one_tuning(i))

    # Print summary of each tuning run
    for r in tuning_results:
        status = "OK" if r["valid"] else "DEGENERATE"
        err = r.get("error", "")
        extra = f" ({err})" if err else ""
        print(f"  run {r['idx']+1}: L={r['L']:.3f}, "
              f"step_size={r['step_size']:.4f}, "
              f"logdensity={r['logdensity']:.2f} [{status}]{extra}",
              flush=True)

    # Select best tuning run
    valid_runs = [r for r in tuning_results if r["valid"]]
    if not valid_runs:
        raise RuntimeError(
            f"MCLMC tuning failed: all {NUM_TUNE_RUNS} runs produced "
            f"degenerate results. Try increasing num_warmup or using "
            f"sampler='nuts'.")

    if len(valid_runs) >= 3:
        # Sort by step_size and pick the median
        valid_runs.sort(key=lambda r: r["step_size"])
        selected = valid_runs[len(valid_runs) // 2]
    else:
        # Few valid runs — pick the one with best (highest) logdensity
        selected = max(valid_runs, key=lambda r: r["logdensity"])

    selected_idx = selected["idx"]
    state_after_tuning = selected["state"]
    sampler_params = selected["sampler_params"]

    print(f"MCLMC: selected run {selected_idx+1} — "
          f"L={selected['L']:.3f}, step_size={selected['step_size']:.4f}, "
          f"logdensity={selected['logdensity']:.2f} "
          f"({time.time() - t0:.1f}s)", flush=True)

    # Free memory: strip heavy objects from non-selected tuning runs
    for r in tuning_results:
        if r["idx"] != selected_idx:
            r.pop("state", None)
            r.pop("sampler_params", None)

    # --- 4. Sample num_chains chains (Python-level loop) ---
    L_tuned = sampler_params.L
    step_tuned = sampler_params.step_size
    inv_mass = sampler_params.inverse_mass_matrix
    MAX_RETRIES = 5

    def _build_kernel(step_size, L):
        alg = blackjax.mclmc(
            logdensity_fn, L=L, step_size=step_size,
            inverse_mass_matrix=inv_mass,
        )
        return jax.jit(alg.step)

    print("MCLMC: JIT-compiling step function...", flush=True)
    t0 = time.time()
    step_size_cur = step_tuned
    L_cur = L_tuned
    step_fn = _build_kernel(step_size_cur, L_cur)
    # Force JIT compilation with a single test step
    test_key = jax.random.fold_in(run_key, 999999)
    test_state, _ = step_fn(test_key, state_after_tuning)
    test_state.logdensity.block_until_ready()
    print(f"MCLMC: JIT compilation done ({time.time() - t0:.1f}s)", flush=True)

    # --- Helper: sample one chain (no progress bar — used by threads) ---
    def _sample_chain(chain_idx, step_fn_, run_key_, state0, n_samples,
                      keep_every=1):
        """Return (sample_list, None) on success, (None, fail_step) on NaN."""
        chain_key = jax.random.fold_in(run_key_, chain_idx)
        state_cur = state0
        sample_list = []
        n_nan = 0
        for i in range(n_samples):
            step_key = jax.random.fold_in(chain_key, i)
            state_cur, _ = step_fn_(step_key, state_cur)
            ld = float(state_cur.logdensity)
            if not np.isfinite(ld):
                n_nan += 1
                if n_nan >= 10:
                    return None, i
            else:
                n_nan = 0
            if (i + 1) % keep_every == 0:
                sample_list.append(state_cur.position)
        return sample_list, None

    # Decide whether to run chains in parallel (threads).
    # JAX releases the GIL during XLA execution, so threading works.
    n_cpus = os.cpu_count() or 1
    parallel = num_chains > 1 and n_cpus >= 2

    # Run all chains with automatic step-size retry on NaN divergence.
    # If any chain hits NaN, halve step_size and restart all chains.
    for retry in range(MAX_RETRIES + 1):
        t0 = time.time()
        nan_detected = False
        nan_chain = nan_step = None

        if parallel:
            n_workers = min(num_chains, n_cpus)
            print(f"MCLMC: sampling {num_chains} chains "
                  f"({n_workers} threads)...", flush=True)
            sys.stdout.flush()
            futures = {}
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for c in range(num_chains):
                    f = pool.submit(_sample_chain, c, step_fn, run_key,
                                    state_after_tuning, num_samples, thin)
                    futures[f] = c
                all_chain_samples = [None] * num_chains
                n_done = 0
                for f in as_completed(futures):
                    c = futures[f]
                    samples, fail_step = f.result()
                    if fail_step is not None:
                        nan_detected = True
                        nan_chain = c + 1
                        nan_step = fail_step
                        print(f"MCLMC: chain {c + 1} — NaN at step "
                              f"{fail_step}", flush=True)
                        sys.stdout.flush()
                        # Cancel remaining futures
                        for other_f in futures:
                            other_f.cancel()
                        break
                    all_chain_samples[c] = samples
                    n_done += 1
                    print(f"MCLMC: chain {c + 1}/{num_chains} done",
                          flush=True)
                    sys.stdout.flush()
        else:
            all_chain_samples = []
            for chain_idx in range(num_chains):
                desc = f"MCLMC chain {chain_idx + 1}/{num_chains}"
                if retry > 0:
                    desc += f" (step={float(step_size_cur):.2f})"
                pbar = tqdm(range(num_samples), desc=desc, ncols=100)
                chain_key = jax.random.fold_in(run_key, chain_idx)
                state_cur = state_after_tuning
                sample_list = []
                n_nan = 0
                for i in pbar:
                    step_key = jax.random.fold_in(chain_key, i)
                    state_cur, _ = step_fn(step_key, state_cur)
                    ld = float(state_cur.logdensity)
                    if not np.isfinite(ld):
                        n_nan += 1
                        if n_nan >= 10:
                            pbar.close()
                            nan_detected = True
                            nan_chain = chain_idx + 1
                            nan_step = i
                            break
                    else:
                        n_nan = 0
                    if (i + 1) % thin == 0:
                        sample_list.append(state_cur.position)
                    if (i + 1) % max(1, num_samples // 10) == 0:
                        pbar.set_postfix_str(f"logp={ld:.1f}")
                else:
                    pbar.close()
                if nan_detected:
                    break
                all_chain_samples.append(sample_list)

        if not nan_detected:
            break

        if retry >= MAX_RETRIES:
            raise RuntimeError(
                f"MCLMC sampling failed: NaN logdensity at step {nan_step} "
                f"(chain {nan_chain}) after {MAX_RETRIES} step-size "
                f"reductions (step_size={float(step_size_cur):.4f}). "
                f"Try using sampler='nuts'.")
        old_ss = float(step_size_cur)
        step_size_cur = step_size_cur * 0.5
        L_cur = L_cur * 0.5
        print(f"MCLMC: NaN at step {nan_step} (chain {nan_chain}) "
              f"with step_size={old_ss:.4f} — halving to "
              f"{float(step_size_cur):.4f} and restarting "
              f"(retry {retry + 1}/{MAX_RETRIES})", flush=True)
        step_fn = _build_kernel(step_size_cur, L_cur)

    elapsed = time.time() - t0
    print(f"MCLMC: {num_chains} x {num_samples} samples in {elapsed:.1f}s "
          f"[step_size={float(step_size_cur):.4f}]", flush=True)

    # --- 5. Transform to constrained space ---
    n_retained = len(all_chain_samples[0])
    if thin > 1:
        print(f"MCLMC: thinned by {thin} during sampling — {n_retained} "
              f"samples retained per chain", flush=True)

    print("MCLMC: transforming to constrained space...", flush=True)
    t0 = time.time()
    chain_constrained = []
    for ci in range(len(all_chain_samples)):
        unconstrained = jax.tree.map(
            lambda *arrs: jnp.stack(arrs), *all_chain_samples[ci])
        all_chain_samples[ci] = None  # free raw samples immediately
        constrained = jax.vmap(postprocess_fn)(unconstrained)
        del unconstrained
        chain_constrained.append(constrained)
    del all_chain_samples
    # Stack to (num_chains, num_retained, ...) per site
    all_constrained = jax.tree.map(
        lambda *arrs: jnp.stack(arrs), *chain_constrained)
    del chain_constrained
    print(f"MCLMC: done ({time.time() - t0:.1f}s)", flush=True)

    return _SamplesResult(all_constrained, num_chains=num_chains,
                          tuning_results=tuning_results,
                          tuning_selected_idx=selected_idx)


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
     target_accept_prob, max_tree_depth, rng_seed, sampler, thin) = fold_args

    print(f"\n--- Fold {k+1}/{K}: train={len(train_idx)}, test={len(test_idx)} ---")
    result_k = fit(
        jnp.array(X_u[train_idx]), jnp.array(X[train_idx]),
        jnp.array(y[train_idx]),
        slab_scale=slab_scale, slab_df=slab_df, scale_global=scale_global,
        num_warmup=num_warmup, num_samples=num_samples,
        num_chains=num_chains, target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth, rng_seed=rng_seed + k,
        sampler=sampler, thin=thin,
    )
    probs_k = predict(
        result_k, jnp.array(X_u[test_idx]), jnp.array(X[test_idx]),
        slab_scale=slab_scale, slab_df=slab_df, scale_global=scale_global,
        rng_seed=rng_seed + k,
    )
    return k, y[test_idx], np.array(probs_k.mean(axis=0))


def crossvalidate(X_u, X, y, K=5, slab_scale=1.0, slab_df=4.0,
                  scale_global=1.0, num_warmup=1000, num_samples=None,
                  num_chains=4, target_accept_prob=0.95, max_tree_depth=12,
                  rng_seed=0, sampler="nuts", thin=None, max_workers=None):
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
        MCMC sampler settings.  ``num_samples`` and ``thin`` default
        to sampler-specific values (see :func:`fit`).
    target_accept_prob : float
        NUTS target acceptance probability.
    max_tree_depth : int
        NUTS maximum tree depth.
    rng_seed : int
        Random seed (incremented per fold).
    sampler : {"nuts", "mclmc"}
        Sampling algorithm.
    thin : int or None
        Thinning factor (see :func:`fit`).
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
            target_accept_prob, max_tree_depth, rng_seed, sampler, thin,
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
                   num_samples=None, num_chains=4, target_accept_prob=0.95,
                   max_tree_depth=12, rng_seed=0, sampler="nuts", thin=None,
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
        MCMC sampler settings.  ``num_samples`` and ``thin`` default
        to sampler-specific values (see :func:`fit`).
    target_accept_prob, max_tree_depth : float, int
        NUTS settings.
    rng_seed : int
        Random seed.
    sampler : {"nuts", "mclmc"}
        Sampling algorithm.
    thin : int or None
        Thinning factor (see :func:`fit`).
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
            sampler=sampler, thin=thin, max_workers=max_workers,
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

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(train_sizes, info_values, "ko", markersize=6, label="CV estimates")
    n_curve = np.linspace(0, train_sizes[-1] * 1.2, 200)
    ax.plot(n_curve, _saturation(n_curve, a_fit, b_fit), "b-",
            label=rf"$\Lambda(n) = {a_fit:.2f}\,n\,/\,({b_fit:.0f} + n)$")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Expected information for discrimination (bits)")
    ax.legend(fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    outpath = filestem + "_learning_curve.pdf"
    fig.savefig(outpath)
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
    mcmc : MCMC or _SamplesResult
        Fitted result (NUTS or MCLMC) supporting ``get_samples(group_by_chain=True)``.
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

    def _row(name, x_chain, kappa_val=None):
        x_flat = x_chain.reshape(-1)
        d = {
            "parameter": name,
            "kappa": kappa_val if kappa_val is not None else "",
            "mean": float(jnp.mean(x_flat)),
            "q0.03": float(jnp.percentile(x_flat, 3)),
            "q0.97": float(jnp.percentile(x_flat, 97)),
            "n_eff": float(effective_sample_size(np.array(x_chain))),
            "r_hat": float(split_gelman_rubin(np.array(x_chain))),
        }
        return d

    # Compute shrinkage factors per covariate (kappa_j)
    tau_ch = chain_samples["tau"][..., None]     # (chains, samples, 1)
    eta_ch = chain_samples["eta"][..., None]
    lambda_raw = (chain_samples["aux1_local"]
                  * jnp.sqrt(chain_samples["aux2_local"]))
    lambda_tilde_sq = ((eta_ch ** 2 * lambda_raw ** 2)
                       / (eta_ch ** 2 + tau_ch ** 2 * lambda_raw ** 2))
    kappa_all = 1.0 / (1.0 + tau_ch ** 2 * lambda_tilde_sq)  # (chains, S, J)
    kappa_mean = np.array(kappa_all.reshape(-1, kappa_all.shape[-1]).mean(axis=0))

    # m_eff
    m_eff_chain = (1.0 - kappa_all).sum(axis=-1)  # (chains, samples)

    # tau, eta, m_eff
    rows = [_row("tau", chain_samples["tau"]),
            _row("eta", chain_samples["eta"]),
            _row("m_eff", m_eff_chain)]

    # unpenalized betas
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
            rows.append(_row(penalized_names[idx], beta_chain[..., idx],
                             kappa_val=round(float(kappa_mean[idx]), 4)))

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
    fig.set_size_inches(5, 4)
    fig.tight_layout()
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
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(w["xseq"], w["f_ctrls"], label="controls")
    ax.plot(w["xseq"], w["f_cases"], label="cases")
    ax.set_xlim(-10, 10)
    ax.set_xlabel("Weight of evidence (log units)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    outpath = filestem + "_wevid_dist.pdf"
    fig.savefig(outpath)
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

    # Sort by absolute posterior mean, keep top 20
    order = np.argsort(np.abs(beta_mean))
    n_show = min(20, len(order))
    order = order[-n_show:]  # largest absolute values
    beta_mean = beta_mean[order]
    beta_lo = beta_lo[order]
    beta_hi = beta_hi[order]
    names = [penalized_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(5, max(2, 0.25 * n_show)))
    y_pos = np.arange(n_show)
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.errorbar(beta_mean, y_pos,
                xerr=[beta_mean - beta_lo, beta_hi - beta_mean],
                fmt="o", color="steelblue", ecolor="steelblue",
                elinewidth=1.5, capsize=2, markersize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_ylim(-0.5, n_show - 0.5)
    ax.set_xlabel("Log odds ratio")
    ax.set_title("Top penalized betas (posterior mean, 90% CI)", fontsize=9)
    fig.tight_layout()
    outpath = filestem + "_forest.pdf"
    fig.savefig(outpath)
    plt.show()
    plt.close(fig)
    print(f"Plot saved to {outpath}")
    return outpath


def _m_eff_from_chain_samples(chain_samples):
    """Compute m_eff (effective nonzero coefficients) per chain-sample.

    Parameters
    ----------
    chain_samples : dict
        Posterior samples with chain dimension, as returned by
        ``result.get_samples(group_by_chain=True)``.

    Returns
    -------
    ndarray (num_chains, num_samples)
    """
    tau_ch = chain_samples["tau"][..., None]
    eta_ch = chain_samples["eta"][..., None]
    lam_raw = (chain_samples["aux1_local"]
               * jnp.sqrt(chain_samples["aux2_local"]))
    lam_sq = ((eta_ch ** 2 * lam_raw ** 2)
              / (eta_ch ** 2 + tau_ch ** 2 * lam_raw ** 2))
    kappa = 1.0 / (1.0 + tau_ch ** 2 * lam_sq)
    return (1.0 - kappa).sum(axis=-1)


def plot_pairs(result, filestem):
    """Pairs plot of tau, eta, and m_eff posterior samples.

    Saves the figure to ``{filestem}_pairs.pdf``.

    Parameters
    ----------
    result : MCMC or _SamplesResult
        Fitted model.
    filestem : str
        Output file prefix.

    Returns
    -------
    str
        Path to the saved PDF.
    """
    chain_samples = result.get_samples(group_by_chain=True)
    m_eff_ch = _m_eff_from_chain_samples(chain_samples)

    idata = az.from_dict(posterior={
        "tau": np.array(chain_samples["tau"]),
        "eta": np.array(chain_samples["eta"]),
        "m_eff": np.array(m_eff_ch),
    })
    axes = az.plot_pair(
        idata, var_names=["tau", "eta", "m_eff"],
        scatter_kwargs={"alpha": 0.3, "s": 10},
    )
    fig = (axes.ravel()[0].get_figure() if hasattr(axes, "ravel")
           else axes.get_figure())
    fig.set_size_inches(6, 5)
    fig.tight_layout()
    outpath = filestem + "_pairs.pdf"
    fig.savefig(outpath)
    plt.show()
    plt.close(fig)
    print(f"Plot saved to {outpath}")
    return outpath


def plot_trace(result, filestem, penalized_names=None):
    """Trace plot of tau, eta, m_eff and top 3 penalized coefficients.

    Each chain is plotted in a distinct colour.  Saves the figure to
    ``{filestem}_trace.pdf``.

    Parameters
    ----------
    result : MCMC or _SamplesResult
        Fitted model.
    filestem : str
        Output file prefix.
    penalized_names : list of str, optional
        Display names for the penalized covariates (length J).

    Returns
    -------
    str
        Path to the saved PDF.
    """
    chain_samples = result.get_samples(group_by_chain=True)
    m_eff_ch = _m_eff_from_chain_samples(chain_samples)

    data = {
        "tau": np.array(chain_samples["tau"]),
        "eta": np.array(chain_samples["eta"]),
        "m_eff": np.array(m_eff_ch),
    }

    # Add top 3 penalized coefficients (by absolute posterior mean)
    beta_ch = chain_samples.get("beta")
    if beta_ch is not None:
        J = beta_ch.shape[-1]
        names = (penalized_names if penalized_names is not None
                 else [f"beta[{j}]" for j in range(J)])
        abs_means = np.abs(np.array(beta_ch).reshape(-1, J).mean(axis=0))
        n_show = min(3, J)
        top_idx = np.argsort(abs_means)[-n_show:][::-1]
        for idx in top_idx:
            data[names[idx]] = np.array(beta_ch[..., idx])

    idata = az.from_dict(posterior=data)
    n_vars = len(data)
    axes = az.plot_trace(idata, figsize=(10, 1.5 * n_vars))
    fig = axes.ravel()[0].get_figure()
    fig.tight_layout()
    outpath = filestem + "_trace.pdf"
    fig.savefig(outpath)
    plt.show()
    plt.close(fig)
    print(f"Plot saved to {outpath}")
    return outpath


def plot_autocorr(result, filestem, penalized_names=None):
    """Autocorrelation plot for tau, eta, m_eff and top 3 penalized betas.

    Uses chain 0 only.  Saves the figure to ``{filestem}_autocorr.pdf``.

    Parameters
    ----------
    result : MCMC or _SamplesResult
        Fitted model.
    filestem : str
        Output file prefix.
    penalized_names : list of str, optional
        Display names for the penalized covariates (length J).

    Returns
    -------
    str
        Path to the saved PDF.
    """
    chain_samples = result.get_samples(group_by_chain=True)
    m_eff_ch = _m_eff_from_chain_samples(chain_samples)

    # Use only chain 0: keep shape (1, num_samples, ...) for arviz
    data = {
        "tau": np.array(chain_samples["tau"][:1]),
        "eta": np.array(chain_samples["eta"][:1]),
        "m_eff": np.array(m_eff_ch[:1]),
    }

    beta_ch = chain_samples.get("beta")
    if beta_ch is not None:
        J = beta_ch.shape[-1]
        names = (penalized_names if penalized_names is not None
                 else [f"beta[{j}]" for j in range(J)])
        abs_means = np.abs(np.array(beta_ch).reshape(-1, J).mean(axis=0))
        n_show = min(3, J)
        top_idx = np.argsort(abs_means)[-n_show:][::-1]
        for idx in top_idx:
            data[names[idx]] = np.array(beta_ch[:1, :, idx])

    idata = az.from_dict(posterior=data)
    n_vars = len(data)
    axes = az.plot_autocorr(idata, figsize=(10, 1.5 * n_vars))
    fig = axes.ravel()[0].get_figure()
    fig.tight_layout()
    outpath = filestem + "_autocorr.pdf"
    fig.savefig(outpath)
    plt.show()
    plt.close(fig)
    print(f"Plot saved to {outpath}")
    return outpath


def plot_mclmc_tuning(result, filestem):
    """Diagnostic plot for MCLMC parallel tuning runs.

    Three vertical subplots show logdensity, L, and step_size for each
    tuning run.  The selected run is highlighted in red, other valid
    runs in blue, and degenerate runs in grey.

    Parameters
    ----------
    result : _SamplesResult
        Fitted MCLMC result with ``tuning_results`` attribute.
    filestem : str
        Output file prefix.

    Returns
    -------
    str or None
        Path to the saved PNG, or None if no tuning data is available.
    """
    tuning = getattr(result, "tuning_results", None)
    if tuning is None:
        return None
    selected_idx = getattr(result, "tuning_selected_idx", None)

    fig, axes = plt.subplots(3, 1, figsize=(5, 6), sharex=True)
    metrics = [("logdensity", "Log density"), ("L", "L"),
               ("step_size", "Step size")]

    for ax, (key, label) in zip(axes, metrics):
        for r in tuning:
            i = r["idx"]
            val = r[key]
            if i == selected_idx:
                color, marker, zorder = "red", "D", 10
            elif r["valid"]:
                color, marker, zorder = "steelblue", "o", 5
            else:
                color, marker, zorder = "grey", "x", 1
            ax.plot(i + 1, val, marker=marker, color=color, markersize=8,
                    zorder=zorder)
        ax.set_ylabel(label)
        ax.set_xlim(0.5, len(tuning) + 0.5)

    axes[-1].set_xlabel("Tuning run")
    axes[-1].set_xticks(range(1, len(tuning) + 1))
    axes[0].set_title("MCLMC tuning diagnostics")
    fig.tight_layout()
    outpath = filestem + "_mclmc_tuning.png"
    fig.savefig(outpath, dpi=150)
    plt.show()
    plt.close(fig)
    print(f"Plot saved to {outpath}")
    return outpath


def plot_mclmc_tuning_traces(result, filestem):
    """Overlay trace plots for all MCLMC tuning runs.

    A single figure with two subplots shows step_size and
    energy_change^2 (sqrt scale) across tuning iterations.
    All runs are overlaid; the selected run is drawn in red,
    others in blue.  Vertical dashed lines mark stage boundaries.

    Parameters
    ----------
    result : _SamplesResult
        Fitted MCLMC result with ``tuning_results`` attribute.
    filestem : str
        Output file prefix.

    Returns
    -------
    str or None
        Path to the saved PNG, or None if no tuning data.
    """
    tuning = getattr(result, "tuning_results", None)
    if tuning is None:
        return None
    selected_idx = getattr(result, "tuning_selected_idx", None)

    # Collect runs that have traces
    runs_with_traces = [r for r in tuning if r.get("trace") is not None]
    if not runs_with_traces:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(7, 4.5), sharex=True)

    # Draw stage boundaries from first available trace
    b1, b2 = runs_with_traces[0]["trace"]["stage_boundaries"]
    n_iters = len(runs_with_traces[0]["trace"]["step_size"])
    for ax in axes:
        for bnd in (b1, b2):
            if 0 < bnd < n_iters:
                ax.axvline(bnd, color="grey", linestyle="--",
                           linewidth=0.8)

    # Plot non-selected runs first, then selected on top
    for r in sorted(runs_with_traces,
                    key=lambda r: r["idx"] == selected_idx):
        trace = r["trace"]
        i = r["idx"]
        is_selected = (i == selected_idx)
        ss = np.array(trace["step_size"])
        ev = np.array(trace["energy_var"])
        iters = np.arange(1, len(ss) + 1)
        color = "red" if is_selected else "steelblue"
        alpha = 1.0 if is_selected else 0.4
        lw = 0.9 if is_selected else 0.5
        zorder = 10 if is_selected else 1
        label = f"run {i+1}" + (" (selected)" if is_selected else "")

        axes[0].plot(iters, ss, linewidth=lw, color=color, alpha=alpha,
                     zorder=zorder, label=label)
        axes[1].plot(iters, np.sqrt(ev), linewidth=lw, color=color,
                     alpha=alpha, zorder=zorder)

    axes[0].set_ylabel("step_size")
    axes[1].set_ylabel(r"$\sqrt{\Delta E^2}$")
    axes[-1].set_xlabel("Tuning iteration")
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[0].set_title("MCLMC tuning traces")
    fig.tight_layout()
    outpath = f"{filestem}_mclmc_tuning_traces.png"
    fig.savefig(outpath, dpi=150)
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

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(steps, kl_values, "ko-", markersize=5)
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


def sample_matched_controls(df, case_col, sex_col, age_col, R=1,
                            rng_seed=0):
    """Draw a stratum-matched case-control sample from a DataFrame.

    All cases are retained.  Within each stratum defined by sex and
    10-year age group, *R* controls are sampled per case (without
    replacement).  If a stratum has fewer than *R* × n_cases controls,
    all available controls in that stratum are included and a warning
    is printed.

    Parameters
    ----------
    df : DataFrame
        Source data.
    case_col : str
        Column defining case status (True/1 = case, False/0 = control).
    sex_col : str
        Column defining sex (categorical or integer).
    age_col : str
        Numeric column defining age.  Binned into 10-year groups
        (0–9, 10–19, …).
    R : int
        Number of controls to sample per case within each stratum.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    DataFrame
        Subset of rows from *df* (original index preserved) containing
        all cases and the sampled controls.
    """
    rng = np.random.RandomState(rng_seed)

    cases = df[df[case_col].astype(bool)].copy()
    controls = df[~df[case_col].astype(bool)].copy()

    # 10-year age bins: 0-9, 10-19, …
    age_bin_label = "___age_bin"
    cases[age_bin_label] = (cases[age_col] // 10).astype(int)
    controls[age_bin_label] = (controls[age_col] // 10).astype(int)

    sampled_idx = []
    n_short = 0
    for (sex, abin), stratum_cases in cases.groupby([sex_col, age_bin_label]):
        n_cases = len(stratum_cases)
        n_needed = n_cases * R
        pool = controls[(controls[sex_col] == sex)
                        & (controls[age_bin_label] == abin)]
        if len(pool) >= n_needed:
            chosen = pool.sample(n=n_needed, random_state=rng)
        else:
            chosen = pool
            if len(pool) < n_needed:
                n_short += 1
                print(f"  Stratum (sex={sex}, age={abin * 10}-{abin * 10 + 9}): "
                      f"only {len(pool)} controls for {n_cases} cases "
                      f"(needed {n_needed})")
        sampled_idx.extend(chosen.index.tolist())

    sampled_idx.extend(cases.index.tolist())
    result = df.loc[sampled_idx]

    n_cases_total = len(cases)
    n_ctrl_total = len(sampled_idx) - n_cases_total
    print(f"Matched sample: {n_cases_total} cases, "
          f"{n_ctrl_total} controls "
          f"(ratio {n_ctrl_total / max(1, n_cases_total):.1f}:1)")
    if n_short > 0:
        print(f"  {n_short} strata had fewer than {R} controls per case")

    return result


def run_analysis(df, y_col, unpenalized_cols, penalized_cols, filestem,
                 slab_scale=2.0, slab_df=4.0, p0=None, scale_global=None,
                 standardize=True, sampler="nuts", crossvalidate_=False,
                 num_warmup=1000, num_samples=None, num_chains=4,
                 target_accept_prob=0.95, max_tree_depth=12,
                 rng_seed=0, thin=None, projpred_V=None, max_workers=None):
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
    num_samples : int or None
        Posterior samples per chain.  Defaults to sampler-specific
        values (see :func:`fit`).
    num_chains : int
        Number of MCMC chains.
    target_accept_prob : float
        NUTS target acceptance probability.
    max_tree_depth : int
        NUTS maximum tree depth.
    rng_seed : int
        Random seed.
    thin : int or None
        Thinning factor (see :func:`fit`).
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

    # Standardize penalized covariates; centre unpenalized covariates
    if standardize:
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1.0  # avoid division by zero for constant cols
        X = (X - X_mean) / X_std
        if X_u_raw.shape[1] > 0:
            X_u_raw = X_u_raw - X_u_raw.mean(axis=0)
        print("Penalized covariates standardized to zero mean, unit variance")
        if unpenalized_cols:
            print("Unpenalized covariates centred to zero mean")

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
            sampler=sampler, thin=thin, max_workers=max_workers,
        )
        plot_learning_curve(train_sizes, info_values, filestem)

        cv_result = crossvalidate(
            X_u, X, y, K=5,
            slab_scale=slab_scale, slab_df=slab_df, scale_global=scale_global,
            num_warmup=num_warmup, num_samples=num_samples,
            num_chains=num_chains, target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth, rng_seed=rng_seed,
            sampler=sampler, thin=thin, max_workers=max_workers,
        )
        plot_wevid(cv_result["wevid"], filestem + "_cv")

        return {
            "N": N, "n_controls": n_controls, "n_cases": n_cases,
            "unpenalized_cols": ["Intercept"] + list(unpenalized_cols),
            "penalized_cols": list(penalized_cols),
            "cv": cv_result,
        }

    # --- 3. Fit full model ---
    try:
        result = fit(
            jnp.array(X_u), jnp.array(X), jnp.array(y),
            slab_scale=slab_scale, slab_df=slab_df, scale_global=scale_global,
            num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
            target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth,
            rng_seed=rng_seed, sampler=sampler, thin=thin,
            print_summary=False,
        )
    except RuntimeError as e:
        print(f"\nrun_analysis: sampling failed — {e}")
        print("Exiting without producing plots or summaries.")
        return {
            "error": str(e),
            "N": N, "n_controls": n_controls, "n_cases": n_cases,
            "unpenalized_cols": ["Intercept"] + list(unpenalized_cols),
            "penalized_cols": list(penalized_cols),
        }

    # --- 4. Tuning diagnostics (before posterior summaries) ---
    is_nuts = sampler == "nuts"
    if not is_nuts and getattr(result, "tuning_results", None) is not None:
        plot_mclmc_tuning_traces(result, filestem)

    # --- 5. In-sample diagnostics ---
    unpenalized_names = ["Intercept"] + list(unpenalized_cols)
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
    else:
        plot_pairs(result, filestem)
        plot_trace(result, filestem, penalized_names=list(penalized_cols))
        plot_autocorr(result, filestem, penalized_names=list(penalized_cols))
    plot_wevid(w, filestem)

    # --- 6. Projpred ---
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

    # --- 7. Return ---
    return {
        "result": result,
        "N": N, "n_controls": n_controls, "n_cases": n_cases,
        "unpenalized_cols": ["Intercept"] + list(unpenalized_cols),
        "penalized_cols": list(penalized_cols),
        "beta_hat": beta_hat, "m_eff": np.array(m_eff),
        "insample": insample, "projpred": projpred_result,
    }


