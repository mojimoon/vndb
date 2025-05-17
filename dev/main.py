import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

pwd = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(pwd)
dump = os.path.join(root, "db")
tmp = os.path.join(pwd, "tmp")

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not os.path.exists(dump):
    raise FileNotFoundError(f"Database directory {dump} does not exist")

def load(table):
    path = os.path.join(dump, "db", table)
    header_path = os.path.join(dump, "db", f"{table}.header")
    if not os.path.exists(path) or not os.path.exists(header_path):
        raise FileNotFoundError(f"Data file {path} or header file {header_path} does not exist")
    with open(header_path, "r") as f:
        header = f.read().strip().split("\t")
    df = pd.read_csv(path, sep="\t", header=None, names=header)
    return df

def connect():
    from supabase import create_client
    if SUPABASE_URL is None or SUPABASE_KEY is None:
        raise ValueError("Supabase URL or key is not set in environment variables")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def is_empty(sp, schema):
    data = sp.table(schema).select("*").execute()
    if data.data is None:
        return True
    return len(data.data) == 0

def insert(sp, schema, df, batch_size=10000):
    import time
    df = df.where(pd.notnull(df), None)
    data = df.to_dict(orient="records")
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        sp.table(schema).insert(batch).execute()
        time.sleep(1)

def upsert(sp, schema, df, batch_size=10000):
    import time
    df = df.where(pd.notnull(df), None)
    data = df.to_dict(orient="records")
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        sp.table(schema).upsert(batch).execute()
        time.sleep(1)

def update(schema, df, batch_size=10000):
    sp = connect()
    if is_empty(sp, schema):
        print(f"Table {schema} is empty, inserting data")
        insert(sp, schema, df, batch_size)
    else:
        print(f"Table {schema} is not empty, upserting data")
        upsert(sp, schema, df, batch_size)

min_vote = 30
min_common_vote = 5

def describe(df, name="DataFrame"):
    print(f"{name} Description:")
    print(f"shape = {df.shape}")
    print(f"columns = {df.columns.tolist()}")
    print(f"types = {df.dtypes.to_dict()}")
    for col in df.columns:
        if df[col].dtype == "object":
            print(f"{col} = unique: {df[col].nunique()}, null: {df[col].isnull().sum()}")
            print(f"{df[col].unique()[:5]}")
        else:
            print(f"{col} = min: {df[col].min()}, max: {df[col].max()}, null: {df[col].isnull().sum()}")
            print(df[col].describe())

def nan_check(df, name="DataFrame"):
    print(f"{name} NaN Check:")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            print(f"{col} has NaN values as float")
        if df[col].dtype == "object":
            mask = df[col] == "\\N"
            if mask.sum() > 0:
                print(f"{col} has '\\N' values as string")

def parse_labels(label_str):
    try:
        return [int(x) for x in label_str.strip('{}').split(',') if x.strip().isdigit()]
    except:
        return []

def extract_min(state):
    _ = state[0]
    return _ if _ <= 5 else 0

def _vn():
    vn = load("vn") # id	image	c_image	olang	l_wikidata	c_votecount	c_rating	c_average	length	devstatus	alias	l_renai	description
    vn = vn[vn['c_votecount'] != '\\N']
    vn['c_votecount'] = vn['c_votecount'].astype(int)
    vn = vn[vn['c_votecount'] >= min_vote]
    vn['id'] = vn['id'].str[1:].astype(int)
    vn = vn[['id', 'olang', 'c_votecount', 'c_rating', 'c_average', 'length', 'alias']]
    vn.to_csv(os.path.join(tmp, "vn_min.csv"), index=False)

def _ulist_vns():
    vn = pd.read_csv(os.path.join(tmp, "vn_min.csv"))
    ids = vn['id'].tolist()
    ulist_vns = load("ulist_vns") # uid	vid	added	lastmod	vote_date	started	finished	vote	notes	labels
    ulist_vns['vid'] = ulist_vns['vid'].str[1:].astype(int)
    ulist_vns = ulist_vns[(ulist_vns['vid'].isin(ids)) & (ulist_vns['vote'] != '\\N')]
    ulist_vns['uid'] = ulist_vns['uid'].str[1:].astype(int)
    ulist_vns['vote'] = ulist_vns['vote'].astype(int)
    # ulist_vns['vote10'] = (ulist_vns['vote'] + 5) // 10
    ulist_vns['state'] = ulist_vns['labels'].apply(parse_labels).apply(extract_min)
    ulist_vns = ulist_vns[['uid', 'vid', 'lastmod', 'vote', 'notes', 'state']]
    ulist_vns.to_csv(os.path.join(tmp, "ulist_vns_min.csv"), index=False)

def setup_vn():
    vn = pd.read_csv(os.path.join(tmp, "vn_min.csv"))
    N = vn.shape[0]
    l_vid = vn['id'].to_numpy()
    vid2idx = {vid: idx for idx, vid in enumerate(l_vid)}
    return vn, N, l_vid, vid2idx

'''
performance
naive O(M^2) loop: 10min
triu_indices vectorization: 1.5min
'''
def partial_order():
    from scipy.stats import rankdata
    vn, N, l_vid, vid2idx = setup_vn()

    # ulist_vns = pd.read_csv(os.path.join(tmp, "ulist_vns_min.csv"))
    # u = ulist_vns.copy()
    # u = u[['uid', 'vid', 'vote']]
    # u['vid_idx'] = u['vid'].map(vid2idx)
    # u = u[['uid', 'vid_idx', 'vote']].to_numpy()
    # M = u.shape[0]
    # u = np.c_[u, np.zeros(M, dtype=float)]

    # pv = np.zeros((N, N), dtype=np.int16)
    # nv = np.zeros((N, N), dtype=np.int16)
    # tv = np.zeros((N, N), dtype=np.int16)

    # _begin, _end, _cur = 0, 1, u[0, 0]
    # while _end < M:
    #     if u[_end, 0] == _cur:
    #         _end += 1
    #     else:
    #         for i in range(_begin, _end - 1):
    #             v1, r1 = int(u[i, 1]), u[i, 2]
    #             for j in range(i + 1, _end):
    #                 v2, r2 = int(u[j, 1]), u[j, 2]
    #                 if r1 > r2:
    #                     pv[v1, v2] += 1
    #                     u[i, 3] += 1
    #                 elif r1 < r2:
    #                     nv[v1, v2] += 1
    #                     u[j, 3] += 1
    #                 else:
    #                     u[i, 3] += 0.5
    #                     u[j, 3] += 0.5
    #                 tv[v1, v2] += 1
    #         tot = _end - _begin
    #         u[_begin:_end, 3] = (u[_begin:_end, 3] + 0.5) / tot
    #         _begin = _end
    #         _end = _begin + 1
    #         _cur = u[_begin, 0]
    #         # print(_begin)
    # # the last group
    # if _begin < M:
    #     for i in range(_begin, M - 1):
    #         v1, r1 = int(u[i, 1]), u[i, 2]
    #         for j in range(i + 1, M):
    #             v2, r2 = int(u[j, 1]), u[j, 2]
    #             if r1 > r2:
    #                 pv[v1, v2] += 1
    #                 u[i, 3] += 1
    #             elif r1 < r2:
    #                 nv[v1, v2] += 1
    #                 u[j, 3] += 1
    #             else:
    #                 u[i, 3] += 0.5
    #                 u[j, 3] += 0.5
    #             tv[v1, v2] += 1
    #     tot = M - _begin
    #     u[_begin:M, 3] = (u[_begin:M, 3] + 0.5) / tot

    ulist_vns = pd.read_csv(os.path.join(tmp, "ulist_vns_min.csv"))
    u = ulist_vns.copy()
    u = u[['uid', 'vid', 'vote']]
    u.iloc[:, 1] = u.iloc[:, 1].map(vid2idx)
    u = u[['uid', 'vid', 'vote']].to_numpy()
    M = u.shape[0]
    u = np.c_[u, np.zeros(M, dtype=float)]

    pv, nv, tv = np.zeros((N, N), dtype=np.int16), np.zeros((N, N), dtype=np.int16), np.zeros((N, N), dtype=np.int16)
    uids, idx_start = np.unique(u[:, 0], return_index=True)
    idx_end = np.r_[idx_start[1:], M]

    for s, e in zip(idx_start, idx_end):
        arr = u[s:e]
        n = e - s
        if n < 2:
            u[s:e, 3] = 0.5
            continue
        indices = arr[:, 1].astype(int)
        votes = arr[:, 2]
        idx_i, idx_j = np.triu_indices(n, k=1)
        v1, v2 = indices[idx_i], indices[idx_j]
        r1, r2 = votes[idx_i], votes[idx_j]
        
        gt_mask = r1 > r2
        lt_mask = r1 < r2
        eq_mask = ~(gt_mask | lt_mask)

        pv[v1[gt_mask], v2[gt_mask]] += 1
        nv[v1[lt_mask], v2[lt_mask]] += 1
        tv[v1, v2] += 1

        win_count = np.zeros(n, dtype=float)
        np.add.at(win_count, idx_i[gt_mask], 1)
        np.add.at(win_count, idx_j[lt_mask], 1)
        np.add.at(win_count, idx_i[eq_mask], 0.5)
        np.add.at(win_count, idx_j[eq_mask], 0.5)

        u[s:e, 3] = rankdata(win_count, method='average') / (n + 1)

    ulist_vns['sp'] = np.round(u[:, 3] * 10000).astype(int)
    ulist_vns.to_csv(os.path.join(tmp, "ulist_vns_min.csv"), index=False)

    with open(os.path.join(tmp, "partial_order.csv"), "w") as f:
        f.write("i,j,pv,nv,tv\n")
        for i in range(N - 1):
            for j in range(i + 1, N):
                if tv[i, j] >= min_common_vote:
                    f.write(f"{l_vid[i]},{l_vid[j]},{pv[i, j]},{nv[i, j]},{tv[i, j]}\n")

def ari_geo():
    vn, N, l_vid, vid2idx = setup_vn()

    ulist_vns = pd.read_csv(os.path.join(tmp, "ulist_vns_min.csv"))
    ulist_vns.iloc[:, 1] = ulist_vns.iloc[:, 1].map(vid2idx)
    ulist_vns = ulist_vns[['uid', 'vid', 'vote', 'sp']].to_numpy()
    # ulist_vns[:, 3] = np.maximum(ulist_vns[:, 3], 1)
    M = ulist_vns.shape[0]

    ag3d = np.zeros((N, N, 8), dtype=np.float32)

    uids, idx_start = np.unique(ulist_vns[:, 0], return_index=True)
    idx_end = np.r_[idx_start[1:], M]

    for s, e in zip(idx_start, idx_end):
        arr = ulist_vns[s:e]
        n = e - s
        if n < 2:
            continue
        indices = arr[:, 1].astype(int)
        votes = arr[:, 2]
        sample_percentile = arr[:, 3]
        idx_i, idx_j = np.triu_indices(n, k=1)
        v1, v2 = indices[idx_i], indices[idx_j]
        ag3d[v1, v2, 0] += votes[idx_i]
        ag3d[v1, v2, 1] += votes[idx_j]
        ag3d[v1, v2, 2] += np.log10(votes[idx_i])
        ag3d[v1, v2, 3] += np.log10(votes[idx_j])
        ag3d[v1, v2, 4] += sample_percentile[idx_i]
        ag3d[v1, v2, 5] += sample_percentile[idx_j]
        ag3d[v1, v2, 6] += np.log10(sample_percentile[idx_i])
        ag3d[v1, v2, 7] += np.log10(sample_percentile[idx_j])

    partial_order = pd.read_csv(os.path.join(tmp, "partial_order.csv"))

    idx0 = partial_order.iloc[:, 0].map(vid2idx).to_numpy().astype(int)
    idx1 = partial_order.iloc[:, 1].map(vid2idx).to_numpy().astype(int)
    ag2d = ag3d[idx0, idx1]  # (n, 8)
    ag2d = pd.DataFrame(ag2d, columns=['ariX', 'ariY', 'geoX', 'geoY', 'sp_ariX', 'sp_ariY', 'sp_geoX', 'sp_geoY'])

    tv = partial_order['tv'].to_numpy()
    # vote is 100-scaled, sp is 10000-scaled. convert to a unified scale
    ag2d.iloc[:, 0] = ag2d.iloc[:, 0] / tv / 100
    ag2d.iloc[:, 1] = ag2d.iloc[:, 1] / tv / 100
    ag2d.iloc[:, 2] = np.power(10, ag2d.iloc[:, 2] / tv - 2)
    ag2d.iloc[:, 3] = np.power(10, ag2d.iloc[:, 3] / tv - 2)
    ag2d.iloc[:, 4] = ag2d.iloc[:, 4] / tv / 10000
    ag2d.iloc[:, 5] = ag2d.iloc[:, 5] / tv / 10000
    ag2d.iloc[:, 6] = np.power(10, ag2d.iloc[:, 6] / tv - 4)
    ag2d.iloc[:, 7] = np.power(10, ag2d.iloc[:, 7] / tv - 4)

    ag2d.to_csv(os.path.join(tmp, "ari_geo.csv"), index=False, float_format='%.6f')

def upload_ulist(offset=0):
    ulist_vns = pd.read_csv(os.path.join(tmp, "ulist_vns_min.csv"))
    if offset > 0:
        ulist_vns = ulist_vns.iloc[offset:]
    update("ulist", ulist_vns)

def partial_order_classical(data, N):
    appear = np.zeros(N, dtype=np.int16)
    appear += np.bincount(data[:, 0].astype(int), minlength=N)
    appear += np.bincount(data[:, 1].astype(int), minlength=N)
    scores = np.zeros((N, 4), dtype=np.float32)
    # for _ in range(data.shape[0]):
    #     i, j, pv, nv, tv = data[_]
    #     dv = pv - nv
    #     # total score = (X - Y)
    #     scores[i, 0] += dv
    #     scores[j, 0] -= dv
    #     # percentage score = (X - Y) / N
    #     scores[i, 1] += dv / tv
    #     scores[j, 1] -= dv / tv
    #     # simple score = sign(X - Y)
    #     scores[i, 2] += np.sign(dv)
    #     scores[j, 2] -= np.sign(dv)
    #     # weighted simple score = sign(X - Y) * sqrt(N)
    #     scores[i, 3] += np.sign(dv) * np.sqrt(tv)
    #     scores[j, 3] -= np.sign(dv) * np.sqrt(tv)
    scores[:, 0] = np.bincount(data[:, 0].astype(int), weights=data[:, 2] - data[:, 3], minlength=N)
    scores[:, 0] -= np.bincount(data[:, 1].astype(int), weights=data[:, 2] - data[:, 3], minlength=N)
    scores[:, 1] = np.bincount(data[:, 0].astype(int), weights=(data[:, 2] - data[:, 3]) / data[:, 4], minlength=N)
    scores[:, 1] -= np.bincount(data[:, 1].astype(int), weights=(data[:, 2] - data[:, 3]) / data[:, 4], minlength=N)
    scores[:, 2] = np.bincount(data[:, 0].astype(int), weights=np.sign(data[:, 2] - data[:, 3]), minlength=N)
    scores[:, 2] -= np.bincount(data[:, 1].astype(int), weights=np.sign(data[:, 2] - data[:, 3]), minlength=N)
    scores[:, 3] = np.bincount(data[:, 0].astype(int), weights=np.sign(data[:, 2] - data[:, 3]) * np.sqrt(data[:, 4]), minlength=N)
    scores[:, 3] -= np.bincount(data[:, 1].astype(int), weights=np.sign(data[:, 2] - data[:, 3]) * np.sqrt(data[:, 4]), minlength=N)
    with np.errstate(invalid='ignore'):
        for k in range(4):
            scores[:, k] /= appear
    return scores

def partial_order_bradley_terry(data, N, max_iter=100, eps=1e-6):
    skill = np.ones(N)
    for _ in range(max_iter):
        last_skill = skill.copy()
        n, d = np.zeros(N), np.zeros(N)
        for row in data:
            i, j, pv, nv, tv = row
            n[i] += pv
            d[i] += (pv + nv) / (skill[i] + skill[j])
            n[j] += nv
            d[j] += (pv + nv) / (skill[i] + skill[j])
        skill = n / (d + 1e-10)
        skill /= skill.sum()
        if np.all(np.abs(skill - last_skill) < eps):
            break
    return skill

def partial_order_random_walk(data, N, alpha=0.85, max_iter=100, eps=1e-6):
    mat = np.zeros((N, N))
    for row in data:
        i, j, pv, nv, tv = row
        n = pv + nv
        if n == 0:
            continue
        mat[i, j] += pv / n
        mat[j, i] += nv / n

    row_sums = mat.sum(axis=1, keepdims=True)
    dead_ends = (row_sums == 0)
    with np.errstate(invalid='ignore'):
        mat = mat / row_sums
    mat[dead_ends.flatten(), :] = 1.0 / N
    mat[np.isnan(mat)] = 0

    scores = np.ones(N) / N
    for _ in range(max_iter):
        last_scores = scores.copy()
        scores = alpha * mat.T.dot(scores) + (1 - alpha) / N
        if np.linalg.norm(scores - last_scores, 1) < eps:
            break
    # sum(scores) = 1, smaller score = better
    scores = 1 / scores
    return scores

def partial_order_elo(data, N, K=32, base=1500, divisor=400):
    rating = np.full(N, base)
    for row in data:
        i, j, pv, nv, tv = row
        for _ in range(pv):
            E0 = 1 / (1 + 10 ** ((rating[j] - rating[i]) / divisor))
            rating[i] += K * (1 - E0)
            rating[j] += K * (0 - (1 - E0))
        for _ in range(nv):
            E0 = 1 / (1 + 10 ** ((rating[j] - rating[i]) / divisor))
            rating[i] += K * (0 - E0)
            rating[j] += K * (1 - (1 - E0))
    return rating

def partial_order_elo_v2(data, N, K=32, base=1500, divisor=400, delta_thres=1e-3):
    rating = np.full(N, base)
    for row in data:
        i, j, pv, nv, tv = row
        E0 = 1 / (1 + 10 ** ((rating[j] - rating[i]) / divisor))
        if pv > 0:
            delta = K * (1 - E0)
            if abs(delta) > delta_thres:
                rating[i] += pv * delta
                rating[j] += pv * K * (0 - (1 - E0))
        if nv > 0:
            E0 = 1 / (1 + 10 ** ((rating[j] - rating[i]) / divisor))
            delta = K * (0 - E0)
            if abs(delta) > delta_thres:
                rating[i] += nv * delta
                rating[j] += nv * K * (1 - (1 - E0))
    return rating

def partial_order_entropy(data, N):
    scores = np.zeros((N, 2))
    n = data[:, 2] + data[:, 3]
    # n = np.where(n == 0, 1, n)
    zero_idx = n == 0
    data = data[~zero_idx]
    n = n[~zero_idx]
    p, q = data[:, 2] / n, data[:, 3] / n
    s = p - q
    ent = -(p * np.log2(p + 1e-10) + q * np.log2(q + 1e-10))
    # for idx, row in enumerate(data):
    #     i, j, pv, nv, tv = row
    #     if pv + nv == 0:
    #         continue
    #     scores[i, 0] += s[idx] * ent[idx]
    #     scores[j, 0] -= s[idx] * ent[idx]
    #     scores[i, 1] += ent[idx]
    #     scores[j, 1] += ent[idx]
    scores[:, 0] = np.bincount(data[:, 0].astype(int), weights=s * ent, minlength=N)
    scores[:, 0] -= np.bincount(data[:, 1].astype(int), weights=s * ent, minlength=N)
    scores[:, 1] = np.bincount(data[:, 0].astype(int), weights=ent, minlength=N)
    scores[:, 1] -= np.bincount(data[:, 1].astype(int), weights=ent, minlength=N)
    return scores[:, 0] / (scores[:, 1] + 1e-10)

def partial_order_spectral(data, N):
    W = np.zeros((N, N))
    for row in data:
        i, j, pv, nv, tv = row
        W[i, j] += pv / tv
        W[j, i] += nv / tv
    D = np.diag(W.sum(axis=1))
    L = D - W
    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]
    return fiedler

def rankit_wrapper(data, ranker='massey'):
    from rankit.Table import Table
    from rankit.Ranker import MasseyRanker, ColleyRanker, KeenerRanker, MarkovRanker, ODRanker, DifferenceRanker, EloRanker
    table = None
    if ranker == 'massey':
        ranker = MasseyRanker()
    elif ranker == 'colley':
        ranker = ColleyRanker()
    elif ranker == 'keener':
        ranker = KeenerRanker()
    elif ranker == 'markov_rv':
        hscore = data.iloc[:, 2] / (data.iloc[:, 2] + data.iloc[:, 3] + 1e-10)
        vscore = data.iloc[:, 3] / (data.iloc[:, 2] + data.iloc[:, 3] + 1e-10)
        data['hscore'] = hscore
        data['vscore'] = vscore
        table = Table(data)
        ranker = MarkovRanker()
    elif ranker == 'markov_rdv':
        hscore = (data.iloc[:, 2] - data.iloc[:, 3]) / (data.iloc[:, 2] + data.iloc[:, 3] + 1e-10)
        vscore = -hscore
        hscore = np.maximum(0, hscore)
        vscore = np.maximum(0, vscore)
        data['hscore'] = hscore
        data['vscore'] = vscore
        table = Table(data)
    elif ranker == 'markov_sdv':
        hscore = data.iloc[:, 2] - data.iloc[:, 3]
        vscore = -hscore
        hscore = np.maximum(0, hscore)
        vscore = np.maximum(0, vscore)
        data['hscore'] = hscore
        data['vscore'] = vscore
        table = Table(data)
    elif ranker == 'od':
        ranker = ODRanker()
    elif ranker == 'difference':
        ranker = DifferenceRanker()
    if table is None:
        table = Table(data)
    return ranker.rank(table)

def setup_po(lim=None):
    vn, N, l_vid, vid2idx = setup_vn()

    po = pd.read_csv(os.path.join(tmp, "partial_order.csv"))
    # np.vectorize: 5.5s
    # np.array(list(map)): 4.4s
    # pandas.map + to_numpy: 2.5s
    po.iloc[:, 0] = po.iloc[:, 0].map(vid2idx)
    po.iloc[:, 1] = po.iloc[:, 1].map(vid2idx)
    po = po.to_numpy()

    if lim is not None:
        '''
        performance test load(2000)
        np + classical (naive): 32s
        np + classical (bincount): 3s
        df + classical_df (naive): 66s
        np + random_walk: 4s
        df + random_walk_df: 4s
        np + elo_v2: 19s
        df + elo_v2_df: 13s
        np + entropy (naive): 5s
        np + entropy (bincount): 2.5s
        df + entropy_df (naive): 45s
        rankit.Massey: 8s (sys 35s)
        rankit.Colley: 15s (sys 68s)
        rankit.Keener, rankit.Markov, rankit.OD, rankit.Difference: 4s
        '''
        po = po[(po[:, 0] < lim) & (po[:, 1] < lim)]
        # po = po[(po.iloc[:, 0] < lim) & (po.iloc[:, 1] < lim)]
        # po = po.rename(columns={po.columns[0]: 'host', po.columns[1]: 'visit', po.columns[2]: 'hscore', po.columns[3]: 'vscore'})
        # po = po[['host', 'visit', 'hscore', 'vscore']]
        N = lim
        partial_order_spectral(po, N)
    else:
        return vn, N, l_vid, vid2idx, po

def create_rank():
    vn, N, l_vid, vid2idx = setup_vn()
    po = pd.read_csv(os.path.join(tmp, "partial_order.csv")) # i,j,pv,nv,tv
    po.iloc[:, 0] = po.iloc[:, 0].map(vid2idx)
    po.iloc[:, 1] = po.iloc[:, 1].map(vid2idx)
    po = po.to_numpy()

    scores = pd.DataFrame()
    scores['idx'] = np.arange(N)
    scores['vid'] = l_vid
    _classical = partial_order_classical(po, N)
    _random_walk = partial_order_random_walk(po, N)
    _elo = partial_order_elo_v2(po, N)
    _entropy = partial_order_entropy(po, N)

# _vn()
# _ulist_vns()
# partial_order()
# upload_ulist()
# ari_geo()
setup_po(2000)
# create_rank()