import gzip
import json
import pathlib
import shutil
from datetime import datetime

import cloudpickle as pickle
import pandas as pd


def prep_folder_out(folder, files_to_copy=tuple()):
    timestamp = datetime.now().strftime('%y%b%d-%H%M')
    folder_out = pathlib.Path(folder)
    assert folder_out.exists()
    folder_out = pathlib.Path(folder_out) / f'./exp_{timestamp}'
    if folder_out.exists():
        choice = input(f'folder exists: {folder_out}\ndelete? [y/n]:')
        if choice != 'y':
            raise Exception('quitting')
        shutil.rmtree(folder_out)

    # make folder_out and its "out" subfolder
    (folder_out / 'out').mkdir(exist_ok=True, parents=True)

    # store copy of given file (stores parameters with results)
    for file in files_to_copy:
        shutil.copy(file, folder_out / pathlib.Path(file).name)

    return folder_out


def load_update_all(folder='', verbose=True):
    """ loads all experiments stats from csv, updates csv as needed """
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
    folder_out = folder / 'out'
    file_list = list(folder_out.glob('*result.json'))
    dict_list = list()
    for file in file_list:
        with open(file, 'r') as f:
            dict_list.append(json.load(f))

    # add in existing result
    df = pd.concat((df, pd.DataFrame(dict_list)))

    # round f_ratio to 14 decimal places (avoids floating point comparison failure)
    df['f_ratio'] = df['f_ratio'].round(14)

    # drop duplicates & check for conflicting results
    df.drop_duplicates(inplace=True)
    subset = ['f_ratio', 'seed', 'Analysis', 'stat']
    assert (df.value_counts(subset=subset).max() == 1)

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
