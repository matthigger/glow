from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()


def extract(df):
    # extract
    f_list = sorted(df['hotel_tr'].unique())
    seed_list = sorted(df['seed'].unique())

    shape = len(seed_list), len(f_list)
    score_dict = defaultdict(lambda: np.full(shape=shape, fill_value=np.nan))

    for _, row in df.iterrows():
        seed_idx = seed_list.index(row['seed'])
        f_idx = f_list.index(row['hotel_tr'])

        for feat in ('f1', 'sens', 'spec'):
            score_dict[row['Analysis'], feat][seed_idx, f_idx] = row[feat]

    return f_list, seed_list, score_dict