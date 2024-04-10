import gzip
import json
import pathlib

import cloudpickle as pickle
import pandas as pd


def load_update_all(folder='', verbose=True):
    # load aggregated results
    folder = pathlib.Path(folder)
    assert folder.exists()
    f_csv = folder / 'results.csv'
    if f_csv.exists():
        df = pd.read_csv(f_csv, index_col=None)
    else:
        df = pd.DataFrame()

    n_old = df.shape[0]

    # aggregate results into csv necessary
    file_list = list(folder.glob('out*.json'))
    dict_list = list()
    for file in file_list:
        with open(file, 'r') as f:
            dict_list.append(json.load(f))

    # add in existing result
    df = pd.concat((df, pd.DataFrame(dict_list)))

    # round p_value to 14 decimal places (avoids floating point comparison failure)
    df['p_val'] = df['p_val'].round(14)

    # drop duplicates & check for conflicting results
    df.drop_duplicates(inplace=True)
    assert df.value_counts(subset=['p_val', 'seed', 'Analysis']).max() == 1

    # overwrite csv with latest / greatest
    df.to_csv(f_csv, index=False)

    # delete json files (they're in csv)
    for file in file_list:
        file.unlink()

    if verbose:
        n_new = len(file_list)
        f_csv = f_csv.resolve()
        print(f'{n_old} old and {n_new} new experiments stored in {f_csv}')

    return df


def get_uuid(df, **match_dict):
    """ gets series of all uuid values matching the input dict """
    s_bool = pd.Series(True, index=df.index)
    for col, val in match_dict.items():
        s_bool &= df[col] == val
    return df[s_bool]['uuid']


def load(df, folder='', uuid=None, **kwargs):
    """ loads detail file """
    folder = pathlib.Path(folder)
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
