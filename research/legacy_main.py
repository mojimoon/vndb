import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import re
import csv

pwd = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(pwd)
dump = os.path.join(root, "db")
tmp = os.path.join(pwd, "tmp")

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")

if not os.path.exists(dump):
    raise FileNotFoundError(f"Database directory {dump} does not exist")

def load(table, dirty_quote=False):
    path = os.path.join(dump, "db", table)
    header_path = os.path.join(dump, "db", f"{table}.header")
    if not os.path.exists(path) or not os.path.exists(header_path):
        raise FileNotFoundError(f"Data file {path} or header file {header_path} does not exist")
    with open(header_path, "r") as f:
        header = f.read().strip().split("\t")
    if dirty_quote:
        df = pd.read_csv(path, sep="\t", header=None, names=header, quoting=csv.QUOTE_NONE)
    else:
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

def parse_min(label_str):
    if len(label_str) > 2 and label_str[1].isdigit() and not label_str[2].isdigit():
        return int(label_str[1])
    return 0
    

def _vn():
    vn = load("vn") # id	image	c_image	olang	l_wikidata	c_votecount	c_rating	c_average	length	devstatus	alias	l_renai	description
    vn = vn[vn['c_votecount'] != '\\N']
    vn['c_votecount'] = vn['c_votecount'].astype(int)
    vn = vn[vn['c_votecount'] >= min_vote]
    vn['id'] = vn['id'].str[1:].astype(int)
    vn = vn[['id', 'olang', 'c_votecount', 'c_rating', 'c_average', 'length', 'alias']]
    vn = vn.sort_values(by=['c_rating', 'c_average'], ascending=[False, False])
    vn['rank'] = np.arange(1, vn.shape[0] + 1)
    vn.reset_index(drop=True, inplace=True)
    vn.to_csv(os.path.join(tmp, "vn_min.csv"), index=False)

def _ulist_vns():
    vn = pd.read_csv(os.path.join(tmp, "vn_min.csv"))
    ids = vn['id'].tolist()
    ulist_vns = load("ulist_vns", dirty_quote=True) # uid	vid	added	lastmod	vote_date	started	finished	vote	notes	labels
    ulist_vns['vid'] = ulist_vns['vid'].str[1:].astype(int)
    ulist_vns = ulist_vns[(ulist_vns['vid'].isin(ids)) & (ulist_vns['vote'] != '\\N')]
    ulist_vns['uid'] = ulist_vns['uid'].str[1:].astype(int)
    ulist_vns['vote'] = ulist_vns['vote'].astype(int)
    ulist_vns['state'] = ulist_vns['labels'].apply(parse_min)
    ulist_vns = ulist_vns[['uid', 'vid', 'lastmod', 'vote', 'notes', 'state']]
    # describe(ulist_vns, "ulist_vns")
    ulist_vns.to_csv(os.path.join(tmp, "ulist_vns_min.csv"), index=False)

def _ulist_vns_full():
    ulist_vns = load("ulist_vns", dirty_quote=True)
    ulist_vns = ulist_vns[(ulist_vns['vote'] != '\\N') & (ulist_vns['notes'].notna())]
    ulist_vns['vid'] = ulist_vns['vid'].str[1:].astype(int)
    ulist_vns['uid'] = ulist_vns['uid'].str[1:].astype(int)
    ulist_vns['vote'] = ulist_vns['vote'].astype(int)
    ulist_vns['state'] = ulist_vns['labels'].apply(parse_min)
    ulist_vns = ulist_vns[['uid', 'vid', 'lastmod', 'vote', 'notes', 'state']]
    ulist_vns.to_csv(os.path.join(tmp, "ulist_vns_full.csv"), index=False)

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

# def partial_order_bradley_terry(data, N, max_iter=100, eps=1e-6):
#     skill = np.ones(N)
#     for _ in range(max_iter):
#         last_skill = skill.copy()
#         n, d = np.zeros(N), np.zeros(N)
#         for row in data:
#             i, j, pv, nv, tv = row
#             n[i] += pv
#             d[i] += (pv + nv) / (skill[i] + skill[j])
#             n[j] += nv
#             d[j] += (pv + nv) / (skill[i] + skill[j])
#         skill = n / (d + 1e-10)
#         skill /= skill.sum()
#         if np.all(np.abs(skill - last_skill) < eps):
#             break
#     return skill

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

# def partial_order_elo(data, N, K=32, base=1500, divisor=400):
#     rating = np.full(N, base)
#     for row in data:
#         i, j, pv, nv, tv = row
#         for _ in range(pv):
#             E0 = 1 / (1 + 10 ** ((rating[j] - rating[i]) / divisor))
#             rating[i] += K * (1 - E0)
#             rating[j] += K * (0 - (1 - E0))
#         for _ in range(nv):
#             E0 = 1 / (1 + 10 ** ((rating[j] - rating[i]) / divisor))
#             rating[i] += K * (0 - E0)
#             rating[j] += K * (1 - (1 - E0))
#     return rating

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

# def partial_order_spectral(data, N):
#     W = np.zeros((N, N))
#     for row in data:
#         i, j, pv, nv, tv = row
#         W[i, j] += pv / tv
#         W[j, i] += nv / tv
#     D = np.diag(W.sum(axis=1))
#     L = D - W
#     eigvals, eigvecs = np.linalg.eigh(L)
#     fiedler = eigvecs[:, 1]
#     return fiedler

# def partial_order_embedding(data, N):
#     import networkx as nx
#     from node2vec import Node2Vec

#     G = nx.DiGraph()
#     G.add_nodes_from(range(N))
#     for row in data:
#         i, j, pv, nv, tv = row
#         if pv > nv:
#             G.add_edge(i, j, weight=pv - nv)
#         elif nv > pv:
#             G.add_edge(j, i, weight=nv - pv)
#     node2vec = Node2Vec(G, dimensions=16, walk_length=30, num_walks=200)
#     model = node2vec.fit(window=10, min_count=1, batch_words=4)

#     scores = np.zeros(N)
#     for i in range(N):
#         if str(i) in model.wv:
#             scores[i] = np.mean(model.wv[str(i)])
#     return scores.reshape(-1, 1)

def partial_order_vi(data, N, iters=100, alpha=0.01):
    scores = np.random.normal(0, 1, N)
    for _ in range(iters):
        i, j, pv, nv, tv = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
        delta = scores[i] - scores[j]
        p_ab = 1 / (1 + np.exp(-delta))
        gradient = (pv - tv * p_ab)
        scores[i] += alpha * gradient
        scores[j] -= alpha * gradient
    
    return scores

def rankit_wrapper(data, ranker='massey'):
    from rankit.Table import Table
    from rankit.Ranker import MasseyRanker, ColleyRanker, KeenerRanker, MarkovRanker, ODRanker, DifferenceRanker, EloRanker
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
        partial_order_vi(po, N)
    else:
        return vn, N, l_vid, vid2idx, po

def create_rank():
    vn, N, l_vid, vid2idx = setup_vn()
    po = pd.read_csv(os.path.join(tmp, "partial_order.csv")) # i,j,pv,nv,tv
    po.iloc[:, 0] = po.iloc[:, 0].map(vid2idx)
    po.iloc[:, 1] = po.iloc[:, 1].map(vid2idx)
    po = po.to_numpy()

    scores = np.zeros((N, 8), dtype=np.float32)
    scores[:, 0:4] = partial_order_classical(po, N)
    scores[:, 4] = partial_order_random_walk(po, N)
    scores[:, 5] = partial_order_elo_v2(po, N)
    scores[:, 6] = partial_order_entropy(po, N)
    scores[:, 7] = partial_order_vi(po, N)
    scores = pd.DataFrame(scores, columns=['total@po', 'percent@po', 'simple@po', 'weight@po', 'rw@po', 'elo@po', 'entropy@po', 'vi@po'])
    scores['vid'] = l_vid
    scores.to_csv(os.path.join(tmp, "rank_po.csv"), index=False, float_format='%.4f')

def create_rankit():
    vn, N, l_vid, vid2idx = setup_vn()
    po = pd.read_csv(os.path.join(tmp, "partial_order.csv")) # i,j,pv,nv,tv
    po.iloc[:, 0] = po.iloc[:, 0].map(vid2idx)
    po.iloc[:, 1] = po.iloc[:, 1].map(vid2idx)
    po = po.to_numpy()
    ari_geo = pd.read_csv(os.path.join(tmp, "ari_geo.csv"))
    ari_geo = ari_geo.to_numpy()

    variables = ['prob', 'ari', 'geo', 'sp_ari', 'sp_geo']
    rankers = ['massey', 'colley', 'keener', 'markov_rv', 'markov_rdv', 'markov_sdv', 'od', 'difference']
    scores = np.zeros((N, len(variables) * len(rankers)), dtype=np.float32)
    for g, var in enumerate(variables):
        if g == 0:
            dt = po[:, 0:4]
        elif g == 1:
            dt = np.c_[po[:, 0:2], ari_geo[:, 0:2]]
        elif g == 2:
            dt = np.c_[po[:, 0:2], ari_geo[:, 2:4]]
        elif g == 3:
            dt = np.c_[po[:, 0:2], ari_geo[:, 4:6]]
        elif g == 4:
            dt = np.c_[po[:, 0:2], ari_geo[:, 6:8]]
        df = pd.DataFrame(dt, columns=['host', 'visit', 'hscore', 'vscore'])
        for h, ranker in enumerate(rankers):
            _ = g * len(rankers) + h
            res = rankit_wrapper(df, ranker)
            np.add.at(scores[:, _], res.iloc[:, 0].astype(int), res.iloc[:, 1])
            if ranker != 'od':
                scores[:, _] = scores[:, _] * 10000
    
    scores = pd.DataFrame(scores, columns=[f"{ranker}@{var}" for var in variables for ranker in rankers])
    scores['vid'] = l_vid
    scores.to_csv(os.path.join(tmp, "rank_rankit.csv"), index=False, float_format='%.2f')

# def borda_count_merge_from_ratings(names, ratings_list):
#     # adapted from rankit.Merge.borda_count_merge
#     # names = np.asarray(names)
#     # ratings_array = np.vstack([np.asarray(r) for r in ratings_list])
#     if isinstance(names, pd.DataFrame):
#         names = names.to_numpy() # shape (N,)
#     if isinstance(ratings_list, pd.DataFrame):
#         ratings_list = ratings_list.to_numpy() # shape (M, N)
#     ranks = np.apply_along_axis(lambda x: (-x).argsort().argsort() + 1, 1, ratings_list) # shape (M, N)
#     M, N = ratings_list.shape
#     borda_count = N * M - ranks.sum(axis=1) # shape (N,)
#     ranks_final = pd.Series(borda_count).rank(method='min', ascending=False).astype(int).values # shape (N,)
#     # print(f"shapes: names: {names.shape}, ratings_array: {ratings_array.shape}, ranks: {ranks.shape}, borda_count: {borda_count.shape}, ranks_final: {ranks_final.shape}")
#     result = pd.DataFrame({'name': names, 'BordaCount': borda_count, 'rank': ranks_final})
#     result = result.sort_values(by='BordaCount', ascending=False, ignore_index=True)
#     return result

def borda_count_rewritten(ratings):
    if not isinstance(ratings, pd.DataFrame):
        ratings = pd.DataFrame(ratings) # shape (n_items, n_methods)
    ranks = ratings.rank(method='min', ascending=False).astype(int) # shape (n_items, n_methods)
    n_items, n_methods = ranks.shape
    borda_scores = (n_items * n_methods) - ranks.sum(axis=1) # shape (n_items,)
    return borda_scores

def merge_rank():
    rank_po = pd.read_csv(os.path.join(tmp, "rank_po.csv"))
    rank_rankit = pd.read_csv(os.path.join(tmp, "rank_rankit.csv"))
    N = rank_po.shape[0]
    bcount = np.zeros((N, 8), dtype=np.int32)
    bcount[:, 0] = borda_count_rewritten(rank_po.iloc[:, :-1])
    for i in range(5):
        bcount[:, i+1] = borda_count_rewritten(rank_rankit.iloc[:, i*8:i*8+8])
    bcount[:, 6] = borda_count_rewritten(bcount[:, 1:4])
    bcount[:, 7] = borda_count_rewritten(rank_rankit.iloc[:, [0, 1, 4, 5, 6, 7, 24, 25, 28, 29, 30, 31, 32, 33, 36, 37, 38, 39]])
    bcount = pd.DataFrame(bcount, columns=['po', 'prob', 'ari', 'geo', 'sp_ari', 'sp_geo', 'sci', 'grand'])
    bcount['vid'] = rank_po['vid']
    bcount.to_csv(os.path.join(tmp, "borda_merge.csv"), index=False)

def compute_kendall_matrix(ratings):
    from scipy.stats import kendalltau
    methods = ratings.columns
    n = len(methods)
    kendall = np.zeros((n, n))
    idx_i, idx_j = np.tril_indices(n, k=-1)
    for i, j in zip(idx_i, idx_j):
        kendall[i, j] = kendalltau(ratings.iloc[:, i], ratings.iloc[:, j])[0]
    return pd.DataFrame(kendall, index=methods, columns=methods)

def render_value(val):
    val = round(val * 100)
    if val == 100:
        return "1"
    elif val > 0:
        return f".{val:02d}"
    elif val == 0:
        return "0"
    elif val < 0:
        return f"-.{-val:02d}"
    else:
        return "-1"

def visualize_rank():
    import seaborn as sns
    import matplotlib.pyplot as plt
    vn = pd.read_csv(os.path.join(tmp, "vn_min.csv"))
    vn.sort_values(by=['id'], inplace=True)
    vn.reset_index(drop=True, inplace=True)
    N = vn.shape[0]
    vndb = N - vn['rank']
    rank_po = pd.read_csv(os.path.join(tmp, "rank_po.csv"))
    rank_rankit = pd.read_csv(os.path.join(tmp, "rank_rankit.csv"))
    borda_merge = pd.read_csv(os.path.join(tmp, "borda_merge.csv"))

    ratings = pd.DataFrame()
    ratings[rank_po.columns[:-1]] = rank_po.iloc[:, :-1] # 8
    ratings[rank_rankit.columns[:-1]] = rank_rankit.iloc[:, :-1] # 40
    ratings[borda_merge.columns[:-1]] = borda_merge.iloc[:, :-1] # 8
    ratings['vndb'] = vndb # 0

    ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min())
    kendall = compute_kendall_matrix(ratings)
    M = kendall.shape[0]
    mask = np.zeros_like(kendall, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    fig, ax = plt.subplots(figsize=(20, 20))
    annot = kendall.applymap(render_value)
    sns.heatmap(kendall, mask=mask, annot=annot, fmt="", cmap="YlGnBu",
                cbar_kws={"shrink": .8}, linewidth=0.5, annot_kws={"size": 11}, ax=ax)

    for i in range(1, 7):
        idx = i * 8
        ax.plot([0, idx], [idx, idx], color='black', lw=1.5, clip_on=False)
        ax.plot([idx, idx], [idx, M], color='black', lw=1.5, clip_on=False)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.title("Kendall's Tau Correlation Coefficient over All Methods")
    plt.tight_layout()
    plt.savefig(os.path.join(root, "assets", "kendall_heatmap.png"))

    # [0:8] + [32:40] + [48:]
    my_ratings = ratings.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39, 48, 49, 50, 51, 52, 53, 54, 55, 56]]
    _kendall = compute_kendall_matrix(my_ratings)
    _M = _kendall.shape[0]
    _mask = np.zeros_like(_kendall, dtype=bool)
    _mask[np.triu_indices_from(_mask)] = True
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(_kendall, mask=_mask, annot=True, fmt=".2f", cmap="YlGnBu",
                cbar_kws={"shrink": .8}, linewidth=0.5, ax=ax)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.title("Kendall's Tau Correlation Coefficient over Selected Methods")
    plt.tight_layout()
    plt.savefig(os.path.join(root, "assets", "kendall_heatmap_selected.png"))

def purify(s):
    # only keep Alphanumeric, Chinese and Japanese characters
    s = s.replace('\\n', '')
    s = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\u3040-\u30ff]', '', s)
    s = s.lower()
    return s

def handle_vn_info():
    vn = pd.read_csv(os.path.join(tmp, "vn_min.csv"))
    vids = vn['id']
    vn_titles = load("vn_titles") # id	lang	official	title	latin
    vn_titles['id'] = vn_titles['id'].str[1:].astype(int)
    olang = vn['olang']
    _ja, _zh, _en = vn_titles[vn_titles['lang'] == 'ja'], vn_titles[vn_titles['lang'] == 'zh-Hans'], vn_titles[vn_titles['lang'] == 'en']
    vn['title_ja'] = vn['id'].map(_ja.set_index('id')['title'])
    vn['title_zh'] = vn['id'].map(_zh.set_index('id')['title'])
    vn['title_en'] = vn['id'].map(_en.set_index('id')['title'])

    for i in range(len(vn)):
        zh_q = pd.isna(vn['title_zh'][i])
        if zh_q and not pd.isna(vn['alias'][i]):
            alias = vn['alias'][i]
            _ = alias.split('\\n')
            _ = [a for a in _ if any('\u4e00' <= c <= '\u9fff' for c in a) and not any('\u3040' <= c <= '\u30ff' for c in a)]
            # _ = sorted(_, key=len, reverse=True)
            if len(_) > 0:
                vn.loc[i, 'title_zh'] = _[0]
                zh_q = False

        if pd.isna(vn['title_en'][i]) and pd.isna(vn['title_ja'][i]):
            olang_title = vn_titles[(vn_titles['id'] == vn['id'][i]) & (vn_titles['lang'] == olang[i])]
            if len(olang_title) > 0:
                vn.loc[i, 'title_en'] = olang_title.iloc[0]['latin']
                vn.loc[i, 'title_ja'] = olang_title.iloc[0]['title']

        # if pd.isna(vn['title_en'][i]):
        #     olang_title = vn_titles[(vn_titles['id'] == vn['id'][i]) & (vn_titles['lang'] == olang[i])]
        #     if len(olang_title) > 0:
        #         vn.loc[i, 'title_en'] = olang_title.iloc[0]['latin']
        
        if zh_q and not pd.isna(vn['title_en'][i]):
            vn.loc[i, 'title_zh'] = vn['title_en'][i]
    
    vn.loc[:, ['title_ja', 'title_zh', 'title_en', 'alias']] = vn.loc[:, ['title_ja', 'title_zh', 'title_en', 'alias']].replace('\\N', np.nan, inplace=True)
    vn.loc[:, ['title_ja', 'title_zh', 'title_en', 'alias']] = vn.loc[:, ['title_ja', 'title_zh', 'title_en', 'alias']].fillna('', inplace=True)
    vn['search'] = vn['title_ja'].astype(str) + vn['title_zh'].astype(str) + vn['title_en'].astype(str) + vn['alias'].astype(str)
    vn['search'] = vn['search'].apply(purify)

    releases_vn = load("releases_vn") # id	vid	rtype
    releases_vn['vid'] = releases_vn['vid'].str[1:].astype(int)
    releases_vn = releases_vn[releases_vn['vid'].isin(vids)]

    # For each vid, find the first id where rtype == 'complete'. If none, fallback to the first id for that vid.
    def get_first_release(group):
        complete = group[group['rtype'] == 'complete']
        if not complete.empty:
            return complete.iloc[0]['id']
        else:
            return group.iloc[0]['id']

    first_release_ids = releases_vn.groupby('vid', sort=False).apply(get_first_release)
    vn['rid'] = vn['id'].map(first_release_ids)
    releases = load("releases") # id	gtin	olang	released	... released = YYYYMMDD (int)
    vn['released'] = vn['rid'].map(releases.set_index('id')['released'])
    releases_producers = load("releases_producers") # id	pid	developer	publisher
    # releases_producers = releases_producers[releases_producers['developer'] == 't']
    # vn['developer'] = vn['rid'].map(releases_producers.set_index('id')['pid'])
    def get_first_producer(group):
        developer = group[group['developer'] == 't']
        if not developer.empty:
            return developer.iloc[0]['pid']
        else:
            return group.iloc[0]['pid']
    rp = releases_producers.groupby('id', sort=False).apply(get_first_producer)
    vn['pid'] = vn['rid'].map(rp)
    vn['pid'].fillna('p0', inplace=True)
    vn['pid'] = vn['pid'].str[1:].astype(int)
    vn.drop(columns=['rid'], inplace=True)

    vn.to_csv(os.path.join(tmp, "vn_min.csv"), index=False)

def handle_relations():
    vn = pd.read_csv(os.path.join(tmp, "vn_min.csv"))
    vn_relations = load("vn_relations") # id	vid	relation	official
    vids = vn['id']
    vn_relations['vid'] = vn_relations['vid'].str[1:].astype(int)
    vn_relations['id'] = vn_relations['id'].str[1:].astype(int)
    vn_relations = vn_relations[(vn_relations['vid'].isin(vids)) & (vn_relations['id'].isin(vids)) & (vn_relations['official'] == 't')]
    vn_relations.drop(columns=['official'], inplace=True)
    vn_relations.rename(columns={'id': 'i', 'vid': 'j'}, inplace=True)
    vn_relations.to_csv(os.path.join(tmp, "vn_relations_min.csv"), index=False)

def handle_producer():
    vn = pd.read_csv(os.path.join(tmp, "vn_min.csv"))
    producers = load("producers") # id	type	lang	name	latin	alias	description
    producers['id'] = producers['id'].str[1:].astype(int)
    producers['latin'] = producers['latin'].str.replace('\\N', '', regex=False)
    producers['alias'] = producers['alias'].fillna('')
    producers['search'] = producers['name'].astype(str) + producers['latin'].astype(str) + producers['alias'].astype(str)
    producers['search'] = producers['search'].apply(purify)
    producers = producers[['id', 'name', 'latin', 'search']]
    producers.to_csv(os.path.join(tmp, "producers_min.csv"), index=False)
    with open(os.path.join(tmp, "producers_min.csv"), 'a') as f:
        f.write("0,,,\n")

def generate_comparable():
    vn, N, l_vid, vid2idx = setup_vn()
    po = pd.read_csv(os.path.join(tmp, "partial_order.csv"))
    po.iloc[:, 0] = po.iloc[:, 0].map(vid2idx)
    po.iloc[:, 1] = po.iloc[:, 1].map(vid2idx)
    po['sim'] = po['pv'] / (po['pv'] + po['nv']) * np.log10(po['tv'])
    po['cont'] = (1 - np.abs(po['pv'] - po['nv']) / (po['pv'] + po['nv'])) * np.log10(po['tv'])
    po['pos'] = po['pv'] / (po['pv'] + po['nv']) * np.log10(po['tv'])
    po['neg'] = po['nv'] / (po['pv'] + po['nv']) * np.log10(po['tv'])
    '''
    i, j, pv, nv, tv -> copy to j, i, nv, pv, tv
    find up to 10 pairs for each i
    (1) most popular: tv
    (2) most similar: (tv - nv - pv) / tv * log10(tv)
    (3) most controversial: [1 - abs(pv - nv) / (pv + nv)] * log10(tv)
    (4) most positive: pv / tv * log10(tv)
    (5) most negative: nv / tv * log10(tv)
    (6) related work: vn_relations
    '''
    po = pd.concat([po, po.rename(columns={'i': 'j', 'j': 'i', 'pv': 'nv', 'nv': 'pv', 'pos': 'neg', 'neg': 'pos'})], ignore_index=True)

    po = po.sort_values(['i', 'tv'], ascending=[True, False]).groupby('i').head(10)
    po['rank'] = po.groupby('i').cumcount()
    r1 = po.pivot(index='i', columns='rank', values='j').fillna(-1).astype(int)
    r1.columns = [f"t{i}" for i in range(10)]
    po = po.sort_values(['i', 'sim'], ascending=[True, False]).groupby('i').head(10)
    po['rank'] = po.groupby('i').cumcount()
    r2 = po.pivot(index='i', columns='rank', values='j').fillna(-1).astype(int)
    r2.columns = [f"s{i}" for i in range(10)]
    po = po.sort_values(['i', 'cont'], ascending=[True, False]).groupby('i').head(10)
    po['rank'] = po.groupby('i').cumcount()
    r3 = po.pivot(index='i', columns='rank', values='j').fillna(-1).astype(int)
    r3.columns = [f"c{i}" for i in range(10)]
    po = po.sort_values(['i', 'pos'], ascending=[True, False]).groupby('i').head(10)
    po['rank'] = po.groupby('i').cumcount()
    r4 = po.pivot(index='i', columns='rank', values='j').fillna(-1).astype(int)
    r4.columns = [f"p{i}" for i in range(10)]
    po = po.sort_values(['i', 'neg'], ascending=[True, False]).groupby('i').head(10)
    po['rank'] = po.groupby('i').cumcount()
    r5 = po.pivot(index='i', columns='rank', values='j').fillna(-1).astype(int)
    r5.columns = [f"n{i}" for i in range(10)]
    # vn_relations = pd.read_csv(os.path.join(tmp, "vn_relations_min.csv")) # i,j,relation
    # vn_relations = vn_relations.sort_values(['i', 'j'], ascending=[True, True]).groupby('i').head(10)
    # vn_relations['rank'] = vn_relations.groupby('i').cumcount()
    # r6 = vn_relations.pivot(index='i', columns='rank', values='j').fillna(-1).astype(int)
    # r6.columns = [f"r{i}" for i in range(10)]
    # horizontally stack all the dataframes to N*60 matrix
    # print(r1.shape, r2.shape, r3.shape, r4.shape, r5.shape)
    res = pd.concat([r1, r2, r3, r4, r5], axis=1, ignore_index=True)
    res = res.applymap(lambda x: l_vid[x] if x >= 0 else -1)
    res.columns = r1.columns.tolist() + r2.columns.tolist() + r3.columns.tolist() + r4.columns.tolist() + r5.columns.tolist()
    res['id'] = l_vid
    res.to_csv(os.path.join(tmp, "comparable.csv"), index=False)

def general_statistics():
    import subprocess
    def get_file_linecount(filepath):
        result = subprocess.run(['wc', '-l', filepath], stdout=subprocess.PIPE, text=True)
        return int(result.stdout.split()[0])
    
    _dict = {}
    genstat = pd.DataFrame(columns=['key', 'count', 'mean', 'std'])
    for k in ['chars', 'images', 'producers', 'releases', 'staff', 'tags', 'traits', 'ulist_vns', 'users', 'vn']:
        _dict[k] = get_file_linecount(os.path.join(dump, "db", k))

    ulist_vns = load("ulist_vns")
    ulist_vns["label_set"] = ulist_vns["labels"].apply(parse_labels)
    for l in range(1, 6):
        ulist_vns[f"l{l}"] = ulist_vns["label_set"].apply(lambda s: l in s)
        _dict[f"ulist_state_{l}"] = ulist_vns[f"l{l}"].sum()

    genstat = pd.concat([genstat, pd.DataFrame(_dict.items(), columns=['key', 'count'])], ignore_index=True)

    ulist_vns = ulist_vns[(ulist_vns['vote'] != '\\N') & (ulist_vns['vote_date'] != '\\N')]
    vote_np = ulist_vns["vote"].to_numpy()
    vote_np = (vote_np.astype(np.float16) / 10).astype(np.int16)
    vsum = np.zeros(11, dtype=int)
    for a in range(1, 11):
        vsum[a] = np.sum(vote_np == a)
    _mean, _std = np.mean(vote_np), np.std(vote_np)
    genstat = pd.concat([genstat, pd.DataFrame({'key': 'vote', 'count': len(vote_np), 'mean': _mean, 'std': _std}, index=[0])], ignore_index=True)
    genstat = pd.concat([genstat, pd.DataFrame({'key': f'vote_{i}', 'count': vsum[i]} for i in range(11))], ignore_index=True)
    ulist_vns['vote_date'] = pd.to_datetime(ulist_vns['vote_date'], format='%Y-%m-%d')
    ulist_vns['year'] = ulist_vns['vote_date'].dt.year
    ulist_vns['vote'] = vote_np
    ulist_vns = ulist_vns.groupby('year').agg({'vote': ['count', 'mean', 'std']})
    year_data = ulist_vns['vote'].reset_index()
    year_data.columns = ['key', 'count', 'mean', 'std']
    genstat = pd.concat([genstat, year_data], ignore_index=True)
    genstat['key'] = genstat['key'].astype(str)
    genstat['count'] = genstat['count'].astype(int)
    genstat.to_csv(os.path.join(tmp, "genstat.csv"), index=False, float_format='%.3f')

# _vn()
# _ulist_vns()
_ulist_vns_full()
# partial_order()
# upload_ulist()
# ari_geo()
# create_rank()
# create_rankit()
# merge_rank()
# visualize_rank()
# handle_vn_info()
# handle_relations()
# handle_producer()
# generate_comparable()
# general_statistics()
