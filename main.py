import os
# import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from scipy.sparse import dok_matrix
# from tqdm import tqdm

db_dir = 'db/db'
out_dir = 'out'

if not os.path.exists(db_dir):
    raise FileNotFoundError(f"Database directory '{db_dir}' does not exist. Please run the setup script.")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def read(table_name):
    table_path = os.path.join(db_dir, table_name)
    table_header_path = os.path.join(db_dir, f"{table_name}.header")
    if not os.path.exists(table_path) or not os.path.exists(table_header_path):
        raise FileNotFoundError(f"Table file '{table_path}' does not exist.")
    df = pd.read_csv(table_path, sep='\t', encoding='utf-8', header=None)
    with open(table_header_path, 'r', encoding='utf-8') as f:
        header = f.read().strip().split('\t')
    df.columns = header
    return df

min_vote = 30
min_common_vote = 3
N = 0

def process_user(ulist_vns):
    pv, nv, tv = dok_matrix((N, N), dtype=np.int16), dok_matrix((N, N), dtype=np.int16), dok_matrix((N, N), dtype=np.int16)
    # print(ulist_vns.iloc[0, 0])
    ulist = ulist_vns[['idx', 'vote']].to_numpy()
    n_entries = len(ulist)
    for i in range(n_entries - 1):
        ri, si = ulist[i, 1], ulist[i, 0]
        for j in range(i + 1, n_entries):
            rj, sj = ulist[j, 1], ulist[j, 0]
            if ri > rj:
                pv[si, sj] += 1
            elif ri < rj:
                nv[si, sj] += 1
            tv[si, sj] += 1
    return pv, nv, tv

def parallel_partial_order(n_workers=cpu_count()):
    vn = read("vn")
    vn = vn[vn['c_rating'] != '\\N']
    vn['c_rating'] = vn['c_rating'].astype(int)
    vn = vn[vn['c_rating'] >= min_vote]
    print(f"# of vn: {len(vn)}")
    vid2idx = {vid: i for i, vid in enumerate(vn['id'])}
    with open(os.path.join(out_dir, "vid.txt"), "w") as f:
        for vid in vid2idx:
            f.write(f"{vid}\n")
    global N
    N = len(vn)

    ulist_vns = read("ulist_vns")
    ulist_vns = ulist_vns[ulist_vns['vid'].isin(vn['id']) & (ulist_vns['vote'] != '\\N')]
    ulist_vns['vote'] = ulist_vns['vote'].astype(int)
    ulist_vns['idx'] = ulist_vns['vid'].map(vid2idx)
    ulist_vns = ulist_vns[['uid', 'idx', 'vote']]
    print(f"# of ulist_vns: {len(ulist_vns)}")

    grouped = [group for _, group in ulist_vns.groupby('uid')]
    print(f"# of groups: {len(grouped)}")
    with Pool(n_workers) as pool:
        results = pool.imap(process_user, grouped)
        # results = list(tqdm(pool.imap(process_user, grouped), total=len(grouped), desc="Processing users"))
    
    pv, nv, tv = dok_matrix((N, N), dtype=np.int16), dok_matrix((N, N), dtype=np.int16), dok_matrix((N, N), dtype=np.int16)
    for pvi, nvi, tvi in results:
        pv += pvi
        nv += nvi
        tv += tvi
    with open(os.path.join(out_dir, "partial_order.csv"), "w") as f:
        f.write("x,y,pv,nv,tv\n")
        for i in range(N):
            for j in range(i + 1, N):
                # if tv[i, j] > 0:
                if tv[i, j] >= min_common_vote:
                    f.write(f"{i},{j},{pv[i, j]},{nv[i, j]},{tv[i, j]}\n")

def partial_order():
    vn = read("vn")
    vn = vn[vn['c_rating'] != '\\N']
    vn['c_rating'] = vn['c_rating'].astype(int)
    vn = vn[vn['c_rating'] >= min_vote]
    print(f"# of vn: {len(vn)}")
    vid2idx = {vid: i for i, vid in enumerate(vn['id'])}
    with open(os.path.join(out_dir, "vid.txt"), "w") as f:
        for vid in vid2idx:
            f.write(f"{vid}\n")
    global N
    N = len(vn)

    ulist_vns = read("ulist_vns")
    ulist_vns = ulist_vns[ulist_vns['vid'].isin(vn['id']) & (ulist_vns['vote'] != '\\N')]
    ulist_vns['vote'] = ulist_vns['vote'].astype(int)
    ulist_vns['idx'] = ulist_vns['vid'].map(vid2idx)
    ulist_vns = ulist_vns[['uid', 'idx', 'vote']].to_numpy()
    print(f"# of ulist_vns: {len(ulist_vns)}")

    pv, nv, tv = np.zeros((N, N), dtype=np.int16), np.zeros((N, N), dtype=np.int16), np.zeros((N, N), dtype=np.int16)
    _begin = 0
    _end = 1
    while _end < len(ulist_vns):
        if ulist_vns[_begin, 0] == ulist_vns[_end, 0]:
            _end += 1
        else:
            for i in range(_begin, _end - 1):
                ri, si = ulist_vns[i, 2], ulist_vns[i, 1]
                for j in range(i + 1, _end):
                    rj, sj = ulist_vns[j, 2], ulist_vns[j, 1]
                    if ri > rj:
                        pv[si, sj] += 1
                    elif ri < rj:
                        nv[si, sj] += 1
                    tv[si, sj] += 1
            _begin = _end
            _end += 1
            # print(_begin)
    for i in range(_begin, _end - 1):
        ri, si = ulist_vns[i, 2], ulist_vns[i, 1]
        for j in range(i + 1, _end):
            rj, sj = ulist_vns[j, 2], ulist_vns[j, 1]
            if ri > rj:
                pv[si, sj] += 1
            elif ri < rj:
                nv[si, sj] += 1
            tv[si, sj] += 1
    with open(os.path.join(out_dir, "partial_order.csv"), "w") as f:
        f.write("x,y,pv,nv,tv\n")
        for i in range(N):
            for j in range(i + 1, N):
                # if tv[i, j] > 0:
                if tv[i, j] >= min_common_vote:
                    f.write(f"{i},{j},{pv[i, j]},{nv[i, j]},{tv[i, j]}\n")

def po_reduce():
    po = pd.read_csv(os.path.join(out_dir, "partial_order.csv"))
    print(f"# of partial order: {len(po)}")
    po = po[po['tv'] >= min_common_vote]
    print(f"# of filtered partial order: {len(po)}")
    po.to_csv(os.path.join(out_dir, "partial_order.csv"), index=False)

def po_stat():
    po = pd.read_csv(os.path.join(out_dir, "partial_order.csv"))
    for b in (3, 5, 10, 20, 30, 50, 100):
        po_b = po[po['tv'] >= b]
        print(f">={b}: {len(po_b)}")

def po_load(id_limit=None):
    po = pd.read_csv(os.path.join(out_dir, "partial_order.csv"))
    # po = po[po['tv'] >= min_common_vote]
    if id_limit is not None:
        po = po[(po['x'] < id_limit) & (po['y'] < id_limit)]
    print(f"# of partial order: {len(po)}")
    po = po.to_numpy()
    return po

def vid_load():
    vid = np.loadtxt(os.path.join(out_dir, "vid.txt"), dtype=str)
    return vid

def classical_score(data, N):
    _dv = data[:, 2] - data[:, 3]  # dv = pv - nv
    scores = np.zeros((N, 4))
    for l in range(data.shape[0]):
        i, j, pv, nv, tv = data[l]
        dv = _dv[l]
        # total score = (X - Y)
        scores[i, 0] += dv
        scores[j, 0] -= dv
        # percentage score = (X - Y) / M
        scores[i, 1] += dv / tv
        scores[j, 1] -= dv / tv
        # simple score = sign(X - Y)
        scores[i, 2] += np.sign(dv)
        scores[j, 2] -= np.sign(dv)
        # weighted simple score = sign(X - Y) * sqrt(M)
        scores[i, 3] += np.sign(dv) * np.sqrt(tv)
        scores[j, 3] -= np.sign(dv) * np.sqrt(tv)
    return scores

def bradley_terry_score(data, N, max_iter=100, eps=1e-6):
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

def random_walk_score(data, N, alpha=0.85, max_iter=100, eps=1e-6):
    mat = np.zeros((N, N))
    for row in data:
        i, j, pv, nv, tv = row
        n = pv + nv
        mat[i, j] += pv / n
        mat[j, i] += nv / n
    mat = mat / mat.sum(axis=1, keepdims=True)
    mat[np.isnan(mat)] = 0
    scores = np.ones(N) / N
    for _ in range(max_iter):
        last_scores = scores.copy()
        scores = alpha * mat.T.dot(scores) + (1 - alpha) / N
        if np.linalg.norm(scores - last_scores, 1) < eps:
            break
    return scores

def elo_rating_score(data, N, K=32):
    rating = np.full(N, 1500.0)
    for row in data:
        i, j, pv, nv, tv = row
        for _ in range(pv):
            E0 = 1 / (1 + 10 ** ((rating[j] - rating[i]) / 400))
            rating[i] += K * (1 - E0)
            rating[j] += K * (0 - (1 - E0))
        for _ in range(nv):
            E0 = 1 / (1 + 10 ** ((rating[j] - rating[i]) / 400))
            rating[i] += K * (0 - E0)
            rating[j] += K * (1 - (1 - E0))
    return rating

def entropy_weighted_score(data, N):
    scores = np.zeros((N, 2))
    n = data[:, 2] + data[:, 3]
    n = np.where(n == 0, 1, n)
    p, q = data[:, 2] / n, data[:, 3] / n
    s = p - q
    ent = -(p * np.log2(p + 1e-10) + q * np.log2(q + 1e-10))
    for idx, row in enumerate(data):
        i, j, pv, nv, tv = row
        scores[i, 0] += s[idx] * ent[idx]
        scores[j, 0] -= s[idx] * ent[idx]
        scores[i, 1] += ent[idx]
        scores[j, 1] += ent[idx]
    return scores[:, 0] / (scores[:, 1] + 1e-10)

def performance_test():
    po = po_load(2000)
    vid = vid_load()
    N = len(vid)
    t0 = time.time()
    scores = classical_score(po, N)
    t1 = time.time()
    print(f"Classical score time: {t1 - t0:.2f}s")
    # t0 = time.time()
    # skill = bradley_terry_score(po, N)
    # t1 = time.time()
    # print(f"Bradley-Terry score time: {t1 - t0:.2f}s")
    t0 = time.time()
    pagerank = random_walk_score(po, N)
    t1 = time.time()
    print(f"Random walk score time: {t1 - t0:.2f}s")
    t0 = time.time()
    elo = elo_rating_score(po, N)
    t1 = time.time()
    print(f"Elo rating score time: {t1 - t0:.2f}s")
    t0 = time.time()
    entropy = entropy_weighted_score(po, N)
    t1 = time.time()
    print(f"Entropy weighted score time: {t1 - t0:.2f}s")

def full_order():
    pass

def main():
    # partial_order()
    performance_test()

if __name__ == "__main__":
    main()