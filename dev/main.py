import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client

pwd = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(pwd)
dump = os.path.join(root, "db")
tmp = os.path.join(pwd, "tmp")

load_dotenv()
SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

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
    if SUPABASE_URL is None or SUPABASE_KEY is None:
        raise ValueError("Supabase URL or key is not set in environment variables")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

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
    ulist_vns['vote10'] = (ulist_vns['vote'] + 5) // 10
    ulist_vns['state'] = ulist_vns['labels'].apply(parse_labels).apply(extract_min)
    ulist_vns = ulist_vns[['uid', 'vid', 'lastmod', 'vote', 'vote10', 'notes', 'state']]
    ulist_vns.to_csv(os.path.join(tmp, "ulist_vns_min.csv"), index=False)

def partial_order():
    vn = pd.read_csv(os.path.join(tmp, "vn_min.csv"))
    N = vn.shape[0]
    vid_array = vn['id'].to_numpy()
    vid2idx = {vid: idx for idx, vid in enumerate(vid_array)}

    ulist_vns = pd.read_csv(os.path.join(tmp, "ulist_vns_min.csv"))
    u = ulist_vns.copy()
    u = u[['uid', 'vid', 'vote']]
    u['vid_idx'] = u['vid'].map(vid2idx)
    u = u[['uid', 'vid_idx', 'vote']].to_numpy()
    M = u.shape[0]
    u = np.c_[u, np.zeros(M, dtype=float)]

    pv = np.zeros((N, N), dtype=np.int16)
    nv = np.zeros((N, N), dtype=np.int16)
    tv = np.zeros((N, N), dtype=np.int16)

    _begin, _end, _cur = 0, 1, u[0, 0]
    while _end < M:
        if u[_end, 0] == _cur:
            _end += 1
        else:
            for i in range(_begin, _end - 1):
                v1, r1 = int(u[i, 1]), u[i, 2]
                for j in range(i + 1, _end):
                    v2, r2 = int(u[j, 1]), u[j, 2]
                    if r1 > r2:
                        pv[v1, v2] += 1
                        u[i, 3] += 1
                    elif r1 < r2:
                        nv[v1, v2] += 1
                        u[j, 3] += 1
                    else:
                        u[i, 3] += 0.5
                        u[j, 3] += 0.5
                    tv[v1, v2] += 1
            tot = _end - _begin
            u[_begin:_end, 3] = (u[_begin:_end, 3] + 0.5) / tot
            _begin = _end
            _end = _begin + 1
            _cur = u[_begin, 0]
            print(_begin)
    # the last group
    if _begin < M:
        for i in range(_begin, M - 1):
            v1, r1 = int(u[i, 1]), u[i, 2]
            for j in range(i + 1, M):
                v2, r2 = int(u[j, 1]), u[j, 2]
                if r1 > r2:
                    pv[v1, v2] += 1
                    u[i, 3] += 1
                elif r1 < r2:
                    nv[v1, v2] += 1
                    u[j, 3] += 1
                else:
                    u[i, 3] += 0.5
                    u[j, 3] += 0.5
                tv[v1, v2] += 1
        tot = M - _begin
        u[_begin:M, 3] = (u[_begin:M, 3] + 0.5) / tot

    ulist_vns['norm'] = np.round(u[:, 3], 3)
    ulist_vns.to_csv(os.path.join(tmp, "ulist_vns_min.csv"), index=False)

    with open(os.path.join(tmp, "partial_order.csv"), "w") as f:
        f.write("x0,x1,pv,nv,tv\n")
        for i in range(N - 1):
            for j in range(i + 1, N):
                if tv[i, j] >= min_common_vote:
                    f.write(f"{vid_array[i]},{vid_array[j]},{pv[i, j]},{nv[i, j]},{tv[i, j]}\n")

# _vn()
# _ulist_vns()
partial_order()