"""On-disk path helpers and confusion-count metric derivation."""

import pathlib

from platformdirs import user_data_dir


def get_path_data() -> pathlib.Path:
    """Return glow's per-user data directory, creating it if missing."""
    path_data = pathlib.Path(user_data_dir('glow', 'glow_author'))
    path_data.mkdir(parents=True, exist_ok=True)
    return path_data


def get_path_result() -> pathlib.Path:
    """Return the per-user results directory, creating it if missing."""
    path_result = get_path_data() / 'results'
    path_result.mkdir(parents=True, exist_ok=True)
    return path_result


def get_path_cache() -> pathlib.Path:
    """Return the per-user joblib.Memory cache dir, made if missing."""
    path_cache = get_path_data() / 'cache'
    path_cache.mkdir(parents=True, exist_ok=True)
    return path_cache


def get_path_records() -> pathlib.Path:
    """Return the per-user recorder dir (per-hash json), made if missing.

    Sibling to the joblib.Memory cache (get_path_cache): a record is keyed by
    the same args hash joblib files its result under, so the two line up. See
    glow._extra.benchmark.recorder.
    """
    path_records = get_path_data() / 'records'
    path_records.mkdir(parents=True, exist_ok=True)
    return path_records


def add_metric_cols(df):
    """Add derived dice/sens/ppv/spec columns from tp/fp/tn/fn counts.

    The per-config CSVs store only the four confusion counts; this derives the
    overlap metrics in memory for any consumer (plots, comparison REPL).
    A no-op when the count columns are absent (e.g. an empty frame).

    Args:
        df (pd.DataFrame): a results frame, possibly carrying tp/fp/tn/fn.

    Returns:
        df with dice/sens/ppv/spec columns added (a copy via assign when
        the counts are present, else the input unchanged).
    """
    import glow.mask

    if not {'tp', 'fp', 'tn', 'fn'}.issubset(df.columns):
        return df
    stats = glow.mask.stats_from_counts(
        tp=df['tp'], fp=df['fp'], tn=df['tn'], fn=df['fn'])
    return df.assign(**stats)
