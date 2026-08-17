"""CLI entry point for the paper benchmarks.

Running python -m glow._extra.benchmark resolves cache names from
config.CONFIG and drives each selected one through driver.drive. A sweep only
fills the shared provenance records; turning them into the per-figure CSVs is
a separate step (python -m glow._extra.benchmark.make_csv, which plot also
runs on its way to the figures).

A sweep runs only the cells the records do not already hold in full, so a
rerun fills the gaps rather than recomputing finished work (--no-skip forces
the whole grid).

--method narrows the sweep to one analysis recipe (repeatable), the path to
take after changing one method's recipe: a changed knob is a new hash, so that
method's leaves go missing everywhere while its siblings' stay valid, and
completeness is judged against the narrowed grid -- so only the named recipes
run and no sibling fit is recomputed.

GLOW's leaves fit on the GPU where one is visible and on many cores where it
is not (config.GLOW_FIT_PARAMS). That parallelism multiplies against -j rather
than sharing it, so a parallel sweep on a machine with a card is refused
(driver.check_fit_params); --no-gpu is the way to take -j instead, and the two
score identically.

Usage:
    python -m glow._extra.benchmark                    # everything
    python -m glow._extra.benchmark 'sweep_*'          # glob
    python -m glow._extra.benchmark sweep_llr          # GPU where visible
    python -m glow._extra.benchmark -j 4 --no-gpu sweep_llr  # parallel, CPU
    python -m glow._extra.benchmark --no-skip sweep_llr # recompute every cell
    python -m glow._extra.benchmark --method VBA --method CET   # one recipe
    python -m glow._extra.benchmark --list             # list cache names
"""
import argparse
from fnmatch import fnmatch


def resolve_names(patterns) -> list:
    """Expand literal cache names / fnmatch patterns into CONFIG names.

    Empty input means "everything". An unknown literal name raises, and a
    pattern that matches nothing also raises, so a typo surfaces rather than
    silently selecting no caches.

    Args:
        patterns (list): literal CONFIG names and/or fnmatch patterns; empty
            selects every catalogue entry.

    Returns:
        out (list): the resolved cache names, de-duplicated and in first-seen
            order.

    Raises:
        ValueError: a literal name is unknown, or a pattern matches nothing.
    """
    # imported lazily so --help / --list construct nothing they don't need
    from .config import CONFIG

    if not patterns:
        return list(CONFIG)

    out, seen = [], set()
    for pattern in patterns:
        if pattern in CONFIG:
            matches = [pattern]
        else:
            matches = [k for k in CONFIG if fnmatch(k, pattern)]
            if not matches:
                raise ValueError(f'no cache names match: {pattern}')
        for k in matches:
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def run(names=None, n_jobs: int = 1, verbose: bool = True,
        skip_recorded: bool = True, methods=None,
        no_gpu: bool = False) -> list:
    """Drive the selected CONFIG caches into the shared records.

    For each resolved cache name, runs drive(*CONFIG[name], n_jobs=n_jobs). The
    sweep writes nothing but records; exporting them as the per-figure CSVs is
    make_csv's job, so a run and its aggregation can happen (and fail)
    independently.

    A sweep skips the cells already complete in the records by default, so a
    rerun (or a grid widened by a config edit) computes only what is missing
    rather than everything the local joblib cache happens not to hold; see
    drive.

    methods narrows each cache's leaf grid to the named analysis recipes
    (grid.filter_ana_list), which is how one method is rerun on its own after
    its recipe changed: completeness is judged against the narrowed grid, so a
    cell already holding those leaves is skipped and the recipes left out are
    never called -- the sibling methods' fits are not recomputed. A selected
    cache with no matching recipe (a segment / prune / vba_stat leaf grid,
    which has no per-method axis) is skipped.

    The HCP reference dataset is ensured once up front (idempotent / cached)
    for the HCP-backed caches.

    Args:
        names (list | None): cache names or fnmatch patterns; None selects all.
        n_jobs (int): joblib worker count (1 = serial; -1 = all cores).
        verbose (bool): print per-cache headers and the drive() progress bar.
        skip_recorded (bool): skip the cells already complete in the records
            (default); False recomputes every cell of the grid.
        methods (list[str] | None): analysis-recipe labels
            (config.ana_kwargs_dict keys, e.g. ['VBA', 'CET']) to run; None
            (default) runs each cache's whole leaf grid.
        no_gpu (bool): run every leaf on the CPU (grid.strip_gpu), which is
            what makes a parallel sweep legal on a machine with a card; see
            driver.check_fit_params. The scores are identical either way.

    Returns:
        driven (list[str]): the cache names actually swept, in selection order
            -- the resolved names less those methods left with no recipe.
    """
    resolved = resolve_names(names or [])

    from .config import CONFIG, ana_kwargs_dict
    from .grid import filter_ana_list, strip_gpu
    from .data import RECORDER
    from .driver import drive
    from .hcp import ensure_hcp_data

    # most caches build on the HCP reference dataset; ensure it once (cached)
    ensure_hcp_data()

    driven = []
    for name in resolved:
        kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc = \
            CONFIG[name]
        if methods:
            kwargs_fnc_list = filter_ana_list(kwargs_fnc_list, methods,
                                              ana_kwargs_dict)
            if not kwargs_fnc_list:
                if verbose:
                    print(f'\n=== {name}: no {methods} recipe in its leaf '
                          f'grid, skipped ===')
                continue
        if no_gpu:
            kwargs_fnc_list = strip_gpu(kwargs_fnc_list)
        if verbose:
            print(f'\n=== {name} ({RECORDER.folder}) ===')
        drive(kwargs_data_list, kwargs_effect_list, kwargs_fnc_list, fnc,
              n_jobs=n_jobs, verbose=verbose, skip_recorded=skip_recorded)
        driven.append(name)

    if verbose:
        print(f'\nswept {len(driven)} cache(s) into the records; export them '
              f'with python -m glow._extra.benchmark.make_csv')
    return driven


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the benchmark CLI arguments.

    Args:
        argv (list | None): argument list to parse; None reads sys.argv.

    Returns:
        the parsed argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description='Run the paper benchmarks.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('names', nargs='*',
                        help='cache names or fnmatch patterns (default: all)')
    parser.add_argument('-j', '--n-jobs', type=int, default=1,
                        help='joblib worker count (1=serial; -1=all cores)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='suppress per-cache headers and progress bars')
    parser.add_argument('--list', action='store_true', dest='list_names',
                        help='print the catalogue cache names and exit')
    parser.add_argument('--no-skip', action='store_true',
                        help='recompute every cell, including those already '
                             'complete in the records')
    parser.add_argument('--method', action='append', dest='methods',
                        default=None, metavar='LABEL',
                        help='run only this analysis recipe (repeatable; a '
                             'config.ana_kwargs_dict key, e.g. VBA)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='fit every leaf on the CPU, which is what lets '
                             '-j run above 1 on a machine with a card')
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Parse the CLI arguments and run the benchmark.

    Args:
        argv (list | None): argument list to parse; None reads sys.argv.
    """
    args = parse_args(argv)

    if args.list_names:
        from .config import CONFIG
        for name in CONFIG:
            print(name)
        return

    run(names=args.names, n_jobs=args.n_jobs, verbose=not args.quiet,
        skip_recorded=not args.no_skip, methods=args.methods,
        no_gpu=args.no_gpu)


if __name__ == '__main__':
    main()
