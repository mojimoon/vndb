import os
import sys
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

db_dir = 'db/db'
out_dir = 'out'

if not os.path.exists(db_dir):
    raise FileNotFoundError(f"Database directory '{db_dir}' does not exist. Please run the setup script.")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# schemas = {
#     'ulist_vns': ['uid', 'vid', 'added', 'lastmod', 'vote_date', 'started', 'finished', 'vote', 'notes', 'labels'],
#     'vn': ['id', 'image', 'c_image', 'olang', 'l_wikidata', 'c_votecount', 'c_rating', 'c_average', 'length', 'devstatus', 'alias', 'l_renai', 'description'],
#     'vn_titles': ['id', 'lang', 'official', 'title', 'latin'],
# }

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

# def partial_order():
#     vn = read("vn")
#     vn['c_rating'] = vn['c_rating'].replace('\\N', 0).astype(int)
#     vn = vn[vn['c_rating'] >= min_vote]
#     vid2idx = {}
#     for i, vid in enumerate(vn['id']):
#         vid2idx[vid] = i
#     print(f"# of vn: {len(vn)}")
#     print(f"estimated memory usage: {len(vn) * len(vn) * 6 / 1024 / 1024:.3f} MB")
    
#     ulist_vns = read("ulist_vns")
#     ulist_vns = ulist_vns[ulist_vns['vid'].isin(vn['id']) & (ulist_vns['vote'] != '\\N')]
#     ulist_vns['vote'] = ulist_vns['vote'].astype(int)
#     ulist_vns['idx'] = ulist_vns['vid'].map(vid2idx)
#     # by default grouped by uid, vid ascendingly
#     ulist = ulist_vns[['uid', 'idx', 'vote']].to_numpy()
#     print(f"# of ulist: {len(ulist)}")

#     pv = np.zeros((len(vn), len(vn)), dtype=np.int16) # int16 = 2 bytes
#     nv = np.zeros((len(vn), len(vn)), dtype=np.int16)
#     tv = np.zeros((len(vn), len(vn)), dtype=np.int16)
#     _begin = 0
#     _end = 1

#     while _end < len(ulist):
#         if ulist[_end, 0] == ulist[_begin, 0]:
#             _end += 1
#         else:
#             for i in range(_begin, _end - 1):
#                 ri = ulist[i, 2]
#                 si = ulist[i, 1]
#                 for j in range(i + 1, _end):
#                     rj = ulist[j, 2]
#                     sj = ulist[j, 1]
#                     if ri > rj: # guaranteed that si < sj
#                         pv[si, sj] += 1
#                     elif ri < rj:
#                         nv[si, sj] += 1
#                     tv[si, sj] += 1
#             # print(f"(begin, end) = ({_begin}, {_end})")
#             _begin = _end
#             _end += 1
    
#     with open(os.path.join(out_dir, "partial_order.csv"), "w") as f:
#         f.write("x,y,pv,nv,tv\n")
#         for i in range(len(vn)):
#             for j in range(i + 1, len(vn)):
#                 if tv[i, j] > 0:
#                     print(f"({i}, {j}) {pv[i, j]} {nv[i, j]} {tv[i, j]}")
#                     f.write(f"{vn.iloc[i, 0]},{vn.iloc[j, 0]},{pv[i, j]},{nv[i, j]},{tv[i, j]}\n")

def xy_to_uppertrig(x, y, n):
    return (x * n) + y - ((x + 1) * (x + 2)) // 2

def uppertrig_to_xy(u, n):
    # max u(x) = ((n - 1) + (n - 2) + ... + (n - x)) / 2 = x * (2 * n - x - 1) / 2 >= u
    # -1/2 * x^2 + (n - 1/2) * x - u >= 0 => x^2 + (1 - 2n) * x + 2u <= 0
    # x = (-b + sqrt(b^2 - 4ac)) / 2a
    x = int((2 * n - 1 - np.sqrt((2 * n - 1) ** 2 - 8 * u)) / 2)
    # max_ux = (x * (2 * n - x - 1)) // 2
    # y = u - max_ux + x + 1
    y = u - (x * (2 * n - x - 1)) // 2 + x + 1
    return x, y

def test_uppertrig():
    n = 5
    for i in range(n):
        for j in range(i + 1, n):
            idx = xy_to_uppertrig(i, j, n)
            x, y = uppertrig_to_xy(idx, n)
            print(f"({i}, {j}) -> {idx} -> ({x}, {y})")

def process_user(ulist, vn_len):

def parallel_partial_order(n_workers=None):
    vn = read("vn")
    vn['c_rating'] = vn['c_rating'].replace('\\N', 0).astype(int)
    vn = vn[vn['c_rating'] >= min_vote]
    vid2idx = {}
    for i, vid in enumerate(vn['id']):
        vid2idx[vid] = i
    print(f"# of vn: {len(vn)}")
    
    ulist_vns = read("ulist_vns")
    ulist_vns = ulist_vns[ulist_vns['vid'].isin(vn['id']) & (ulist_vns['vote'] != '\\N')]
    ulist_vns['vote'] = ulist_vns['vote'].astype(int)
    ulist_vns['idx'] = ulist_vns['vid'].map(vid2idx)
    # by default grouped by uid, vid ascendingly
    ulist_vns = ulist_vns[['uid', 'idx', 'vote']]
    print(f"# of ulist: {len(ulist_vns)}")
    # group by uid
    grouped = [group for _, group in ulist_vns.groupby('uid')]
    if n_workers is None:
        n_workers = cpu_count()
    print(f"Using {n_workers} workers")
    print(f"# of groups: {len(grouped)}")
    # with Pool(n_workers) as pool:
    #     results = pool.starmap(process_user_group, [(group.to_numpy(), len(vn)) for group in grouped])
    # n

if __name__ == "__main__":
    parallel_partial_order()
