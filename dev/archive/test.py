from rankit.Table import Table
from rankit.Ranker import MasseyRanker, ColleyRanker, KeenerRanker, MarkovRanker, ODRanker, DifferenceRanker, EloRanker
import pandas as pd
import numpy as np
from scipy.stats import kendalltau

def rankit_wrapper(data, ranker='massey'):
    table = None
    if ranker == 'massey':
        worker = MasseyRanker()
    elif ranker == 'colley':
        worker = ColleyRanker()
    elif ranker == 'keener':
        worker = KeenerRanker()
    elif ranker.startswith('markov'):
        if ranker == 'markov_rv':
            hscore = data.iloc[:, 2] / (data.iloc[:, 2] + data.iloc[:, 3] + 1e-10)
            vscore = data.iloc[:, 3] / (data.iloc[:, 2] + data.iloc[:, 3] + 1e-10)
        elif ranker == 'markov_rdv':
            hscore = (data.iloc[:, 2] - data.iloc[:, 3]) / (data.iloc[:, 2] + data.iloc[:, 3] + 1e-10)
            vscore = -hscore
            hscore = np.maximum(0, hscore)
            vscore = np.maximum(0, vscore)
        elif ranker == 'markov_sdv':
            hscore = data.iloc[:, 2] - data.iloc[:, 3]
            vscore = -hscore
            hscore = np.maximum(0, hscore)
            vscore = np.maximum(0, vscore)
        data.iloc[:, 2] = hscore
        data.iloc[:, 3] = vscore
        table = Table(data)
        worker = MarkovRanker()
    elif ranker == 'od':
        worker = ODRanker()
    elif ranker == 'difference':
        worker = DifferenceRanker()
    if table is None:
        table = Table(data)
    return worker.rank(table)

# goal: test if each of the rankers has their results affected by the multiplier (holding the ratio constant)

dt = pd.read_csv('sample_data.csv')
dt.columns = ['host', 'visit', 'hscore', 'vscore']

for ranker in ['massey', 'colley', 'keener', 'markov_rv', 'markov_rdv', 'markov_sdv', 'od', 'difference']:
    data = dt.copy()
    res0 = rankit_wrapper(data, ranker=ranker)
    data.iloc[:, 2] = data.iloc[:, 2] * 10
    data.iloc[:, 3] = data.iloc[:, 3] * 10
    res1 = rankit_wrapper(data, ranker=ranker)
    data.iloc[:, 2] = data.iloc[:, 2] * 0.01
    data.iloc[:, 3] = data.iloc[:, 3] * 0.01
    res2 = rankit_wrapper(data, ranker=ranker)
    res0, res1, res2 = res0.iloc[:, 1], res1.iloc[:, 1], res2.iloc[:, 1]

    print(f'{ranker}\t{kendalltau(res0, res1).correlation:.4f}\t{kendalltau(res0, res2).correlation:.4f}\t{kendalltau(res1, res2).correlation:.4f}')