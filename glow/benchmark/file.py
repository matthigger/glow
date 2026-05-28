"""On-disk result helpers: result directory, csv aggregation, uuids."""

import json
import pathlib
from uuid import uuid4

import pandas as pd
from platformdirs import user_data_dir

OUT = 'out'
ERROR = 'error'


def short_uuid() -> str:
    """Return an 8-character hex UUID string."""
    return str(uuid4())[:8]


def get_path_result() -> pathlib.Path:
    """Return the per-user results directory, creating it if missing."""
    path_result = (pathlib.Path(user_data_dir('glow', 'glow_author')) /
                   'results')
    path_result.mkdir(parents=True, exist_ok=True)
    return path_result


def load_update_all(label: str, verbose: bool = True, result_dir=None):
    """Load all experiment results for a label, folding in any new json.

    Reads the label's results.csv (if present), appends any per-trial
    *result.json files left in the out subfolder, de-duplicates, rewrites
    the csv, and deletes the now-redundant json files.

    Args:
        label (str): result subfolder name under result_dir.
        verbose (bool): print a summary of old / new counts.
        result_dir (pathlib.Path | None): base results directory;
            defaults to get_path_result() when None.

    Returns:
        A tuple (df, folder, n_new) of the combined results DataFrame,
        the label's folder Path, and the number of json files folded in.
    """
    base = result_dir if result_dir is not None else get_path_result()
    folder = base / label
    if not folder.exists():
        return pd.DataFrame(), folder, 0

    f_csv = folder / 'results.csv'
    if f_csv.exists():
        df = pd.read_csv(f_csv, index_col=None)
    else:
        df = pd.DataFrame()

    n_old = df.shape[0]

    folder_out = folder / 'out'
    file_list = list(folder_out.glob('*result.json')) if folder_out.exists() else []
    dict_list = list()
    for file in file_list:
        with open(file, 'r') as f:
            dict_list.append(json.load(f))

    df = pd.concat((df, pd.DataFrame(dict_list)))

    if df.empty:
        return df, folder, 0

    # round so equal effects compare equal despite float representation
    # (otherwise drop_duplicates below keeps near-identical rows)
    df['effect_llr'] = df['effect_llr'].round(14)

    # lists are unhashable, so tuple-ify list columns for drop_duplicates,
    # then restore them afterwards
    list_cols = [c for c in df.columns if df[c].apply(type).eq(list).any()]
    for c in list_cols:
        df[c] = df[c].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    df.drop_duplicates(inplace=True)
    for c in list_cols:
        df[c] = df[c].apply(lambda x: list(x) if isinstance(x, tuple) else x)

    df.to_csv(f_csv, index=False)

    # json results are now folded into the csv
    for file in file_list:
        file.unlink()

    n_new = len(file_list)
    if verbose:
        f_csv = f_csv.resolve()
        print(f'{n_old} old and {n_new} new experiments stored in {f_csv}')

    return df, folder, n_new
