"""Profile peak memory of permutation and experiment workers across sizes.

Two profiles (selected by --profile):

  permutation (default): WGN grid (num_vox, b, num_img), subprocess runs
  ``process_permutation``, fits Lasso, saves memory_permutation.json for
  ``AWSBatchRunner.estimate_memory_mb``.

  experiment: WGN grid (num_vox, b, num_img, n_perm), subprocess runs
  full ``run_ana`` (minimal config), fits Lasso, saves
  memory_experiment.json for ``estimate_experiment_memory_mb``.

Shared machinery: subprocess + peak RSS, fit_poly_lasso(), load_model(path),
and apply-poly prediction used by both predict() and predict_experiment_memory().

Usage::

    python -m glow.benchmark.memory [--output /tmp/.../memory_profile.csv]
    python -m glow.benchmark.memory --profile experiment [--output /tmp/.../experiment_memory_profile.csv]
"""

import argparse
import json
import math
import multiprocessing as mp
import resource
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm

MODEL_PATH = Path(__file__).resolve().parent.parent / 'aws' / 'memory_permutation.json'
EXPERIMENT_MODEL_PATH = Path(__file__).resolve().parent.parent / 'aws' / 'memory_experiment.json'


# ---------------------------------------------------------------------------
# Permutation profile: worker and measurement
# ---------------------------------------------------------------------------

def _worker_perm(shape, b, num_img, result_queue):
    """Run one permutation in a child process and report peak RSS."""
    from glow.experiment.exper import Experiment
    from glow.aws.worker import process_permutation

    exp = Experiment.from_gauss(shape=shape, b=b, num_img=num_img,
                                a=2, seed=0)
    process_permutation(exp, {}, perm_idx=1)

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result_queue.put(peak_kb / 1024.0)


def measure_peak_rss(num_vox, b, num_img):
    """Fork a subprocess and return its peak RSS in MB (permutation profile)."""
    side = math.ceil(num_vox ** (1 / 3))
    shape = (side, side, side)

    q = mp.Queue()
    p = mp.Process(target=_worker_perm, args=(shape, b, num_img, q))
    p.start()
    p.join()

    if p.exitcode != 0:
        return None
    return q.get()


# ---------------------------------------------------------------------------
# Experiment profile: worker and measurement
# ---------------------------------------------------------------------------

def _worker_experiment(num_vox, b, num_img, n_perm, result_queue):
    """Run full RunAna in a child process and report peak RSS."""
    from glow.experiment.exper import Experiment
    from glow.experiment.exper import ExperimentScaled
    import glow.effect
    from glow.benchmark.runner import RunAna

    side = math.ceil(num_vox ** (1 / 3))
    shape = (side, side, side)
    exp = Experiment.from_gauss(shape=shape, b=b, num_img=num_img, a=2, seed=0)
    exp_scaled = ExperimentScaled.from_exp(exp)
    effect_n_vox = max(1, int(0.2 * num_vox))
    extenter = glow.effect.ExtenterSphere(n_vox=effect_n_vox)
    exp_eff, effect = exp_scaled.impose_effect(
        extenter=extenter, effect_llr=0.05, seed=0
    )

    runner = RunAna({
        'GLOW': (
            __import__('glow.analysis', fromlist=['AnalysisGLOW']).AnalysisGLOW,
            dict(
                n_perm_fwer=n_perm,
                n_perm_fwer_size_adjust=min(25, n_perm // 4),
                alpha_fwer=0.05,
                min_size=1,
            ),
        ),
    })

    class MinimalConfig:
        def get_exp_eff(self, **kwargs):
            return exp_eff, effect

    config = MinimalConfig()
    config.runner = runner
    config.folder = Path(tempfile.mkdtemp(prefix='glow_mem_'))
    config.detail_save = False
    config.error_save = False

    runner.run(config, seed=0, effect_llr=0.05)

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result_queue.put(peak_kb / 1024.0)


def measure_peak_rss_experiment(num_vox, b, num_img, n_perm):
    """Fork a subprocess and return its peak RSS in MB (experiment profile)."""
    q = mp.Queue()
    p = mp.Process(
        target=_worker_experiment,
        args=(num_vox, b, num_img, n_perm, q),
    )
    p.start()
    p.join()

    if p.exitcode != 0:
        return None
    return q.get()


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def build_grid():
    """Grid for permutation profile."""
    vox_targets = np.geomspace(500, 10_000, 8).round().astype(int)
    b_values = [1, 2, 4]
    img_values = [10, 25, 50, 100]
    return list(product(vox_targets, b_values, img_values))


def build_grid_experiment():
    """Grid for experiment profile (smaller for ~3 min total runtime)."""
    vox_targets = np.geomspace(500, 5_000, 4).round().astype(int)
    b_values = [1, 2]
    img_values = [10, 25]
    n_perm_values = [50, 100]
    return list(product(vox_targets, b_values, img_values, n_perm_values))


# ---------------------------------------------------------------------------
# Shared: fit and save model
# ---------------------------------------------------------------------------

def fit_poly_lasso(df, feature_cols, target_col='peak_rss_mb', model_path=None,
                   log_target=False):
    """Fit PolynomialFeatures(degree=2) + LassoCV and save JSON to model_path.

    With ``log_target=True`` the polynomial is fit against ``log(target)`` so
    multiplicative errors are treated uniformly — appropriate for targets
    that span multiple orders of magnitude (e.g. runtime). The saved model
    carries a ``log_target`` flag that ``_apply_poly_model`` honors.
    """
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    if model_path is None:
        model_path = MODEL_PATH

    X_raw = df[feature_cols].values
    y_raw = df[target_col].values
    if log_target:
        if np.any(y_raw <= 0):
            raise ValueError('log_target requires all target values > 0')
        y_fit = np.log(y_raw)
    else:
        y_fit = y_raw

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_raw)
    feature_names = poly.get_feature_names_out(feature_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    lasso = LassoCV(cv=5, max_iter=50_000, fit_intercept=True)
    lasso.fit(X_scaled, y_fit)

    coefs_original = lasso.coef_ / scaler.scale_
    intercept_original = lasso.intercept_ - (coefs_original * scaler.mean_).sum()

    mask = np.abs(coefs_original) > 1e-10
    surviving_names = feature_names[mask].tolist()
    surviving_coefs = coefs_original[mask].tolist()

    y_pred_fit = intercept_original + X_poly @ coefs_original
    ss_res = ((y_fit - y_pred_fit) ** 2).sum()
    ss_tot = ((y_fit - y_fit.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot

    scale_tag = 'log(target)' if log_target else 'target'
    print(f'\n  LassoCV ({scale_tag})  alpha={lasso.alpha_:.6f}  R²={r2:.4f}')
    print(f'  intercept = {intercept_original:.2f}')
    print(f'\n  {"term":<30} {"coefficient":>14}')
    print(f'  {"-"*30} {"-"*14}')
    for name, coef in zip(surviving_names, surviving_coefs):
        print(f'  {name:<30} {coef:>14.6f}')

    zeroed = feature_names[~mask].tolist()
    if zeroed:
        print(f'\n  zeroed out: {", ".join(zeroed)}')

    # Safety-factor calibration: fit Gaussian to actual/predicted ratios,
    # use 99.99% quantile (z=3.7190). Exposed so downstream callers
    # (estimate_timeout_minutes) can size timeouts without hardcoding 2.5x.
    Z_99_99 = 3.7190
    y_pred_linear = np.exp(y_pred_fit) if log_target else y_pred_fit
    valid = y_pred_linear > 0
    if valid.sum() >= 2:
        ratios = y_raw[valid] / y_pred_linear[valid]
        mu_r = float(ratios.mean())
        sd_r = float(ratios.std(ddof=1))
        factor = mu_r + Z_99_99 * sd_r
        max_ratio = float(ratios.max())
        print(f'\n  safety factor (99.99% Gaussian): '
              f'μ={mu_r:.3f}, σ={sd_r:.3f}  →  factor={factor:.3f}')
        print(f'  max observed actual/predicted = {max_ratio:.3f}')
        if max_ratio > factor:
            print(f'  ⚠  max ratio exceeds computed factor — '
                  f'residuals may be non-Gaussian (heavy tails)')
    else:
        factor, mu_r, sd_r, max_ratio = None, None, None, None

    model = {
        'feature_names': surviving_names,
        'intercept': round(intercept_original, 4),
        'coefficients': [round(c, 10) for c in surviving_coefs],
        'r2': round(r2, 4),
        'alpha': round(float(lasso.alpha_), 6),
    }
    if log_target:
        model['log_target'] = True
    if factor is not None:
        model['safety_factor'] = round(factor, 4)
        model['residual_mean'] = round(mu_r, 4)
        model['residual_std'] = round(sd_r, 4)
        model['residual_max_ratio'] = round(max_ratio, 4)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'w') as f:
        json.dump(model, f, indent=2)
    print(f'\n  model saved to {model_path}')

    return model


# ---------------------------------------------------------------------------
# Shared: load and predict (apply-poly helper)
# ---------------------------------------------------------------------------

def _apply_poly_model(model, feature_cols, **kwargs):
    """Apply saved polynomial model; kwargs are feature values (e.g. num_vox=1, b=2, ...).

    If the model was fit with ``log_target=True``, the raw polynomial output
    is exponentiated before being returned.
    """
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=2, include_bias=False)
    n = len(feature_cols)
    poly.fit(np.zeros((1, n)))
    all_names = poly.get_feature_names_out(feature_cols).tolist()

    x = [kwargs[col] for col in feature_cols]
    X_poly = poly.transform([x])[0]

    est = model['intercept']
    for name, coef in zip(model['feature_names'], model['coefficients']):
        idx = all_names.index(name)
        est += coef * X_poly[idx]
    if model.get('log_target'):
        est = float(np.exp(est))
    return est


def load_model(path=None):
    """Load saved memory model coefficients, or None if not found."""
    if path is None:
        path = MODEL_PATH
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def predict(model, num_vox, b, num_img):
    """Predict peak RSS in MB from a saved permutation model dict."""
    return _apply_poly_model(
        model, ['num_vox', 'b', 'num_img'],
        num_vox=num_vox, b=b, num_img=num_img,
    )


def load_experiment_model():
    """Load saved experiment memory model, or None if not found."""
    return load_model(EXPERIMENT_MODEL_PATH)


def predict_experiment_memory(model, num_vox, b, num_img, n_perm):
    """Predict peak RSS in MB from a saved experiment model dict."""
    return _apply_poly_model(
        model, ['num_vox', 'b', 'num_img', 'n_perm'],
        num_vox=num_vox, b=b, num_img=num_img, n_perm=n_perm,
    )


# ---------------------------------------------------------------------------
# Benchmarks: run grid and fit
# ---------------------------------------------------------------------------

def run_benchmark(output_path):
    """Permutation profile: run grid, save CSV, return DataFrame."""
    grid = build_grid()
    rows = []

    for num_vox, b, num_img in tqdm(grid, desc='profiling'):
        peak_mb = measure_peak_rss(num_vox, b, num_img)
        if peak_mb is None:
            continue
        rows.append({
            'num_vox': int(num_vox),
            'b': b,
            'num_img': num_img,
            'peak_rss_mb': round(peak_mb, 1),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f'\n  saved {len(df)} measurements to {output_path}')
    return df


def run_benchmark_experiment(output_path):
    """Experiment profile: run grid, save CSV, return DataFrame."""
    grid = build_grid_experiment()
    rows = []

    for num_vox, b, num_img, n_perm in tqdm(grid, desc='profiling experiment'):
        peak_mb = measure_peak_rss_experiment(num_vox, b, num_img, n_perm)
        if peak_mb is None:
            continue
        rows.append({
            'num_vox': int(num_vox),
            'b': b,
            'num_img': num_img,
            'n_perm': n_perm,
            'peak_rss_mb': round(peak_mb, 1),
        })

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f'\n  saved {len(df)} measurements to {output_path}')
    return df


def fit_model(df):
    """Fit permutation model (backward-compatible wrapper)."""
    return fit_poly_lasso(
        df,
        feature_cols=['num_vox', 'b', 'num_img'],
        target_col='peak_rss_mb',
        model_path=MODEL_PATH,
    )


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--profile',
        choices=['permutation', 'experiment'],
        default='permutation',
        help='Profile to run (default: permutation)',
    )
    p.add_argument(
        '--output',
        default=None,
        help='CSV output path (default: a tempfile)',
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.output is None:
        tmpdir = tempfile.mkdtemp(prefix='glow_memory_')
        name = 'experiment_memory_profile.csv' if args.profile == 'experiment' else 'memory_profile.csv'
        args.output = Path(tmpdir) / name
    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.profile == 'experiment':
        print('Memory profiling benchmark (experiment profile)')
        print('=' * 50)
        df = run_benchmark_experiment(args.output)
        if df.empty:
            print('no measurements collected')
            return
        fit_poly_lasso(
            df,
            feature_cols=['num_vox', 'b', 'num_img', 'n_perm'],
            target_col='peak_rss_mb',
            model_path=EXPERIMENT_MODEL_PATH,
        )
        print('\n  sample predictions (from fitted model):')
        for nv, b, ni, np_ in [(1000, 2, 50, 100), (5000, 2, 50, 250)]:
            model = load_experiment_model()
            if model:
                pred = predict_experiment_memory(model, nv, b, ni, np_)
                print(f'    ({nv} vox, b={b}, {ni} img, n_perm={np_}) -> {pred:,.0f} MB')
        return

    print('Memory profiling benchmark (permutation profile)')
    print('=' * 50)
    df = run_benchmark(args.output)
    if df.empty:
        print('no measurements collected')
        return

    model = fit_model(df)

    print(f'\n  sample predictions (from fitted model):')
    for nv, b, ni in [(1000, 2, 50), (5000, 2, 100), (10000, 4, 100)]:
        pred = predict(model, nv, b, ni)
        print(f'    ({nv:>6} vox, b={b}, {ni:>3} img) -> {pred:,.0f} MB')


if __name__ == '__main__':
    main()
