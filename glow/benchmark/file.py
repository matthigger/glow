import gzip
import json
import pathlib

import cloudpickle as pickle
import pandas as pd
from platformdirs import user_data_dir

OUT = 'out'
ERROR = 'error'


def get_path_result():
    path_result = (pathlib.Path(user_data_dir('glow', 'glow_author')) /
                   'results')
    path_result.mkdir(parents=True, exist_ok=True)
    return path_result


def load_update_all(label, time=None, verbose=True):
    """ loads all experiments stats from csv, updates csv as needed """
    # use latest folder (if none given)
    folder = get_path_result() / label
    assert folder.exists(), f'label not found: {folder}'

    if time is None or not time:
        if verbose:
            print(f'selecting latest folder')
        folder = sorted(folder.glob('*'))[-1]
    else:
        folder = folder / time
    assert folder.exists(), f'time not found: {folder}'

    # load aggregated results
    assert folder.exists()
    f_csv = folder / 'results.csv'
    if f_csv.exists():
        df = pd.read_csv(f_csv, index_col=None)
    else:
        df = pd.DataFrame()

    n_old = df.shape[0]

    # aggregate results into csv necessary
    folder_out = folder / 'out'
    file_list = list(folder_out.glob('*result.json'))
    dict_list = list()
    for file in file_list:
        with open(file, 'r') as f:
            dict_list.append(json.load(f))

    # add in existing result
    df = pd.concat((df, pd.DataFrame(dict_list)))

    if df.empty:
        return df, folder, 0

    # round hotel_tr to 14 decimal places (avoids floating point comparison failure)
    df['hotel_tr'] = df['hotel_tr'].round(14)

    # drop duplicates & check for conflicting results
    df.drop_duplicates(inplace=True)

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


def get_uuid(df, **match_dict):
    """ gets series of all uuid values matching the input dict """
    s_bool = pd.Series(True, index=df.index)
    for col, val in match_dict.items():
        s_bool &= df[col] == val
    return df[s_bool]['uuid']


def load(df, folder='', uuid=None, **kwargs):
    """ loads (Analysis, Effect) from file (detail_save must be True) """
    folder = pathlib.Path(folder) / 'out'
    assert folder.exists()
    if uuid is None:
        s_uuid = get_uuid(df=df, **kwargs)
        assert s_uuid.size == 1, f'search found {s_uuid.size} unique experiments'
        uuid = s_uuid.iloc[0]

    file_list = list(folder.glob(f'*{uuid}_detail*'))
    assert len(file_list) == 1, f'unique file not found for uuid: {uuid}'
    file = file_list[0]

    with gzip.open(file, 'rb') as f:
        x = pickle.load(f)

    return x


if __name__ == '__main__':
    df = load_update_all('vba_hcp')
