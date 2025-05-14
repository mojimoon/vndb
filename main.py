import os
import sys
import numpy as np
import pandas as pd

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

'''
def rela():
    df0 = pd.read_csv(INPUT_0)[["user_id", "subject_id", "rating"]].reset_index(drop=True)
    df0 = df0.sort_values(by=["user_id", "subject_id"]).reset_index(drop=True)
    df0 = df0[df0["subject_id"].isin(ids)]
    df0 = df0[df0["rating"] > 0]
    len0 = len(df0) # 7770854

    last = time.time()
    print("time elapsed: %.2f" % (last - timer))

    pv = {} # relative positive votes
    nv = {} # relative negative votes
    tv = {} # total votes
    arr = df0.to_numpy()
    cur_begin = 0
    cur_end = 1
    # use double pointer to get this current user's records in rows [cur_begin, cur_end-1]
    while cur_end < len0:
        if arr[cur_end, UID] == arr[cur_begin, UID]:
            cur_end += 1
        else:
            for i in range(cur_begin, cur_end - 1):
                ri = arr[i, RAT]
                if ri == 0:
                    continue
                si = arr[i, SID]
                for j in range(i + 1, cur_end):
                    # it is guaranteed that si < sj
                    rj = arr[j, RAT]
                    if rj == 0:
                        continue
                    sj = arr[j, SID]
                    if ri > rj:
                        pv[(si, sj)] = pv.get((si, sj), 0) + 1
                    elif ri < rj:
                        nv[(si, sj)] = nv.get((si, sj), 0) + 1
                    tv[(si, sj)] = tv.get((si, sj), 0) + 1
            cur_begin = cur_end
            cur_end += 1
            # print cur_begin every minute
            if time.time() - last > 59:
                print("%d %.0f" % (cur_begin, time.time() - timer))
                last = time.time()

    print("time elapsed: %.2f" % (time.time() - timer))

    with open(TMP_2, "w") as f:
        for (si, sj), v in tv.items():
            f.write("%d,%d,%d,%d\n" % (si, sj, v, pv.get((si, sj), 0) - nv.get((si, sj), 0)))
'''

def partial_order():
    vn = read("vn")
    vn['c_rating'] = vn['c_rating'].replace('\\N', 0).astype(int)
    vn = vn[vn['c_rating'] >= min_vote]
    vid2idx = {}
    for i, vid in enumerate(vn['id']):
        vid2idx[vid] = i
    print(f"# of vn: {len(vn)}")
    print(f"estimated memory usage: {len(vn) * len(vn) * 6 / 1024 / 1024} MB")
    
    ulist_vns = read("ulist_vns")
    ulist_vns = ulist_vns[ulist_vns['vid'].isin(vn['id']) & (ulist_vns['vote'] != '\\N')]
    ulist_vns['vote'] = ulist_vns['vote'].astype(int)
    ulist_vns['idx'] = ulist_vns['vid'].map(vid2idx)
    # by default grouped by uid, vid ascendingly
    ulist = ulist_vns[['uid', 'idx', 'vote']].to_numpy()
    print(f"# of ulist: {len(ulist)}")

    pv = np.zeros((len(vn), len(vn)), dtype=np.int16) # int16 = 2 bytes
    nv = np.zeros((len(vn), len(vn)), dtype=np.int16)
    tv = np.zeros((len(vn), len(vn)), dtype=np.int16)
    _begin = 0
    _end = 1

    while _end < len(ulist):
        if ulist[_end, 0] == ulist[_begin, 0]:
            _end += 1
        else:
            for i in range(_begin, _end - 1):
                ri = ulist[i, 2]
                # if ri == 0:
                #     continue
                si = ulist[i, 1]
                for j in range(i + 1, _end):
                    rj = ulist[j, 2]
                    # if rj == 0:
                    #     continue
                    sj = ulist[j, 1]
                    if ri > rj: # guaranteed that si < sj
                        pv[si, sj] += 1
                    elif ri < rj:
                        nv[si, sj] += 1
                    tv[si, sj] += 1
            _begin = _end
            _end += 1
    
    with open(os.path.join(out_dir, "partial_order.txt"), "w") as f:
        for i in range(len(vn)):
            for j in range(i + 1, len(vn)):
                if tv[i, j] > 0:
                    f.write(f"{i},{j},{tv[i, j]},{pv[i, j]},{nv[i, j]}\n")

if __name__ == "__main__":
    partial_order()
