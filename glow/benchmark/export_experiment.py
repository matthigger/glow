"""Interactive CLI to regenerate and export a paper experiment.

Usage::

    python -m glow.benchmark.export_experiment
    python -m glow.benchmark.export_experiment --output /tmp/my_export
"""
import argparse
from pathlib import Path

import numpy as np


def _choose(prompt, options, default=0):
    """Present numbered options, return the chosen value."""
    for i, (key, label) in enumerate(options):
        marker = ' *' if i == default else ''
        print(f'  {i + 1}. {label}{marker}')
    raw = input(f'{prompt} [{default + 1}]: ').strip()
    idx = (int(raw) - 1) if raw else default
    return options[idx][0]


def _choose_float(prompt, choices, default_idx=None):
    """Present numeric choices, return the chosen value."""
    if default_idx is None:
        default_idx = len(choices) // 2
    for i, v in enumerate(choices):
        marker = ' *' if i == default_idx else ''
        print(f'  {i + 1}. {v:.4g}{marker}')
    raw = input(f'{prompt} [{default_idx + 1}]: ').strip()
    idx = (int(raw) - 1) if raw else default_idx
    return choices[idx]


def _input_int(prompt, default):
    raw = input(f'{prompt} [{default}]: ').strip()
    return int(raw) if raw else default


def main():
    from glow.benchmark.paper_config import CONFIG_BY_LABEL, COMMON
    from glow.experiment.export import to_folder

    parser = argparse.ArgumentParser(
        description='Regenerate and export a paper experiment to a folder')
    parser.add_argument('--output', '-o', default=None,
                        help='output directory (default: auto-named in cwd)')
    args = parser.parse_args()

    # --- pick config ---
    labels = list(CONFIG_BY_LABEL.keys())
    print('\nAvailable configs:')
    config_key = _choose('Config', [(k, k) for k in labels])
    config = CONFIG_BY_LABEL[config_key]

    # --- pick iteration parameters ---
    # determine what parameters this config iterates over
    if config.iter_params is not None:
        iter_spec = config.iter_params.copy()
    else:
        iter_spec = {'effect_llr': list(config.effect_llr_all)}

    fixed = config.fixed_params.copy() if config.fixed_params else {}

    kwargs = {}

    # seed
    seed = _input_int('Seed', 0)
    kwargs['seed'] = seed

    # for each iterable param (other than seed), let user pick a value
    for param, values in iter_spec.items():
        if param == 'seed':
            continue
        values = list(values)
        if all(isinstance(v, (int, float, np.integer, np.floating))
               for v in values):
            chosen = _choose_float(f'{param}', values)
        else:
            chosen = _choose(f'{param}',
                             [(v, str(v)) for v in values])
        kwargs[param] = chosen

    # merge fixed params
    kwargs.update(fixed)

    # --- generate experiment ---
    print(f'\nGenerating experiment for {config_key}...')
    print(f'  params: {kwargs}')
    exp, effect = config.get_exp_eff(**kwargs)

    # --- export ---
    if args.output:
        out = Path(args.output)
    else:
        parts = [config_key, f'seed{seed}']
        for k, v in kwargs.items():
            if k == 'seed':
                continue
            if isinstance(v, float):
                parts.append(f'{k}_{v:.4g}')
            else:
                parts.append(f'{k}_{v}')
        out = Path.cwd() / '_'.join(parts)

    to_folder(exp, out, effect=effect)


if __name__ == '__main__':
    main()
