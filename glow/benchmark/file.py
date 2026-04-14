import json
import pathlib
from uuid import uuid4

import pandas as pd
from platformdirs import user_data_dir

OUT = 'out'
ERROR = 'error'


def short_uuid():
    """Return an 8-character hex UUID string."""
    return str(uuid4())[:8]


def get_path_result():
    path_result = (pathlib.Path(user_data_dir('glow', 'glow_author')) /
                   'results')
    path_result.mkdir(parents=True, exist_ok=True)
    return path_result


def load_update_all(label, verbose=True, result_dir=None):
    """load all experiment results from csv, updating from json if needed."""
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

    # aggregate results into csv if necessary
    folder_out = folder / 'out'
    file_list = list(folder_out.glob('*result.json')) if folder_out.exists() else []
    dict_list = list()
    for file in file_list:
        with open(file, 'r') as f:
            dict_list.append(json.load(f))

    # add in existing result
    df = pd.concat((df, pd.DataFrame(dict_list)))

    if df.empty:
        return df, folder, 0

    # round effect_llr to 14 decimal places (avoids floating point comparison failure)
    df['effect_llr'] = df['effect_llr'].round(14)

    # drop duplicates — convert list columns to tuples for hashing
    list_cols = [c for c in df.columns if df[c].apply(type).eq(list).any()]
    for c in list_cols:
        df[c] = df[c].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    df.drop_duplicates(inplace=True)
    for c in list_cols:
        df[c] = df[c].apply(lambda x: list(x) if isinstance(x, tuple) else x)

    # overwrite csv with latest / greatest
    df.to_csv(f_csv, index=False)

    # delete json files (they're in csv)
    for file in file_list:
        file.unlink()

    n_new = len(file_list)
    if verbose:
        f_csv = f_csv.resolve()
        print(f'{n_old} old and {n_new} new experiments stored in {f_csv}')

    return df, folder, n_new


if __name__ == '__main__':
    df = load_update_all('vba_hcp')
