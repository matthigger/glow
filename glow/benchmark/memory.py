"""Profile peak memory of a permutation worker across experiment sizes.

Generates WGN experiments over a grid of (num_vox, b, num_img), forks
a subprocess for each to run ``process_permutation``, and measures peak
RSS.  Then fits a Lasso regression (degree-2 polynomial features) to
predict peak memory from the raw variables, and saves the model
coefficients for use by ``AWSBatchRunner.estimate_memory_mb``.

Usage::

    python -m glow.benchmark.memory [--output memory_profile.csv]
"""

import argparse
import json
import math
import multiprocessing as mp
import resource
from itertools import product
from pathlib import Path

import numpy as np
from tqdm import tqdm

MODEL_PATH = Path(__file__).resolve().parent.parent / 'aws' / 'memory_model.json'


def _worker(shape, b, num_img, result_queue):
    """Run one permutation in a child process and report peak RSS."""
    from glow.experiment.exper import Experiment
    from glow.aws.worker import process_permutation

    exp = Experiment.from_gauss(shape=shape, b=b, num_img=num_img,
                                a=2, seed=0)
    process_permutation(exp, {}, perm_idx=1)

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result_queue.put(peak_kb / 1024.0)


def measure_peak_rss(num_vox, b, num_img):
    """Fork a subprocess and return its peak RSS in MB."""
    side = math.ceil(num_vox ** (1 / 3))
    shape = (side, side, side)

    q = mp.Queue()
    p = mp.Process(target=_worker, args=(shape, b, num_img, q))
    p.start()
    p.join()

    if p.exitcode != 0:
        return None
    return q.get()


def build_grid():
    vox_targets = np.geomspace(500, 10_000, 8).round().astype(int)
    b_values = [1, 2, 4]
    img_values = [10, 25, 50, 100]
    return list(product(vox_targets, b_values, img_values))


def run_benchmark(output_path):
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


def fit_model(df):
    """Fit PolynomialFeatures(degree=2) + LassoCV on the measurements."""
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    from sklearn.pipeline import make_pipeline

    feature_cols = ['num_vox', 'b', 'num_img']
    X_raw = df[feature_cols].values
    y = df['peak_rss_mb'].values

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_raw)
    feature_names = poly.get_feature_names_out(feature_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    lasso = LassoCV(cv=5, max_iter=50_000, fit_intercept=True)
    lasso.fit(X_scaled, y)

    coefs_original = lasso.coef_ / scaler.scale_
    intercept_original = lasso.intercept_ - (coefs_original * scaler.mean_).sum()

    mask = np.abs(coefs_original) > 1e-10
    surviving_names = feature_names[mask].tolist()
    surviving_coefs = coefs_original[mask].tolist()

    y_pred = intercept_original + X_poly @ coefs_original
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot

    print(f'\n  LassoCV  alpha={lasso.alpha_:.6f}  R²={r2:.4f}')
    print(f'  intercept = {intercept_original:.2f}')
    print(f'\n  {"term":<30} {"coefficient":>14}')
    print(f'  {"-"*30} {"-"*14}')
    for name, coef in zip(surviving_names, surviving_coefs):
        print(f'  {name:<30} {coef:>14.6f}')

    zeroed = feature_names[~mask].tolist()
    if zeroed:
        print(f'\n  zeroed out: {", ".join(zeroed)}')

    model = {
        'feature_names': surviving_names,
        'intercept': round(intercept_original, 4),
        'coefficients': [round(c, 10) for c in surviving_coefs],
        'r2': round(r2, 4),
        'alpha': round(float(lasso.alpha_), 6),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'w') as f:
        json.dump(model, f, indent=2)
    print(f'\n  model saved to {MODEL_PATH}')

    return model


def load_model():
    """Load saved memory model coefficients, or None if not found."""
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH) as f:
        return json.load(f)


def predict(model, num_vox, b, num_img):
    """Predict peak RSS in MB from a saved model dict."""
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=2, include_bias=False)
    feature_cols = ['num_vox', 'b', 'num_img']
    poly.fit(np.zeros((1, 3)))
    all_names = poly.get_feature_names_out(feature_cols).tolist()

    X_poly = poly.transform([[num_vox, b, num_img]])[0]

    est = model['intercept']
    for name, coef in zip(model['feature_names'], model['coefficients']):
        idx = all_names.index(name)
        est += coef * X_poly[idx]
    return est


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', default='memory_profile.csv',
                   help='CSV output path (default: memory_profile.csv)')
    return p.parse_args()


def main():
    args = parse_args()
    print('Memory profiling benchmark')
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
