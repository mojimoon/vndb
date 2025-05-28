import os
import sys
import numpy as np
import pandas as pd

def data_preprocessing():
    df = pd.read_csv('tmp/ulist_vns_min.csv')
    df = df[df['notes'].notna()]
    df = df.rename(columns={'lastmod': 'date', 'sp': 'score'})
    df = df[['uid', 'vid', 'date', 'notes', 'state', 'score']]
    # remove rows with only one column (due to incorrect parsing)
    df = df[df.apply(lambda x: x.count(), axis=1) > 1]
    df.to_csv('tmp/ulist_vns_minimal.csv', index=False)

data_preprocessing()