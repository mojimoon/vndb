import os
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from scipy.sparse import dok_matrix
from tqdm import tqdm

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
                if tv[i, j] > 0:
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
                if tv[i, j] > 0:
                    f.write(f"{i},{j},{pv[i, j]},{nv[i, j]},{tv[i, j]}\n")

def main():
    partial_order()

if __name__ == "__main__":
    main()