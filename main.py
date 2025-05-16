import os
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from scipy.sparse import dok_matrix
import re
# from tqdm import tqdm

db_dir = 'db/db'
out_dir = 'web/public/out'
web_data_dir = 'web/public/out'

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
min_common_vote = 5
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
    vn = vn[vn['c_votecount'] != '\\N']
    vn['c_votecount'] = vn['c_votecount'].astype(int)
    vn = vn[vn['c_votecount'] >= min_vote]
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

def compute_dist(ulist_vns):
    plain = np.zeros((N, 11), dtype=np.int16)
    ulist = ulist_vns.copy()
    # vote divided by 10 then rounded
    ulist[:, 2] = np.round((ulist[:, 2].astype(np.float16) / 10)).astype(np.int16)
    # describe ulist[:, 2]
    for i in range(ulist.shape[0]):
        plain[ulist[i, 1], ulist[i, 2]] += 1
    votes = np.sum(plain, axis=1)
    avg = np.sum(plain * np.arange(11), axis=1) / votes
    std = np.sqrt(np.sum(plain * (np.arange(11) - avg[:, None]) ** 2, axis=1) / votes)
    # rank = np.empty_like(avg, dtype=np.int16)
    # rank[np.argsort(avg)] = np.arange(N, 0, -1)
    df = pd.DataFrame({'idx': np.arange(N), **{f'_{i}': plain[:, i] for i in range(1, 11)}, 'votes': votes, 'avg': avg, 'std': std})
    df.to_csv(os.path.join(out_dir, "dist.csv"), index=False, float_format='%.4f')

def partial_order():
    vn = read("vn")
    vn = vn[vn['c_votecount'] != '\\N']
    vn['c_votecount'] = vn['c_votecount'].astype(int)
    vn = vn[vn['c_votecount'] >= min_vote]
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
    compute_dist(ulist_vns)
    
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
    # vid = pd.read_csv(os.path.join(out_dir, "vid.csv"))
    # vid, appear = vid['vid'].to_numpy(), vid['appear'].to_numpy()
    # return vid, appear

# def vid_reduce():
#     vid = np.loadtxt(os.path.join(out_dir, "vid.txt"), dtype=str)
#     print(f"# of vid: {len(vid)}")
#     appear = np.zeros(len(vid), dtype=np.int16)
#     po = pd.read_csv(os.path.join(out_dir, "partial_order.csv"))
#     for i, j, pv, nv, tv in po.to_numpy():
#         appear[i] += 1
#         appear[j] += 1
#     indices = np.where(appear > 0)[0]
#     print(f"# of filtered vid: {len(indices)}")
#     vid, appear = vid[indices], appear[indices]
#     df = pd.DataFrame({'vid': vid, 'appear': appear})
#     df.to_csv(os.path.join(out_dir, "vid.csv"), index=False)

def classical_score(data, N, appear):
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
    with np.errstate(invalid='ignore'):
        scores[:, 0] /= appear
        scores[:, 1] /= appear
        scores[:, 2] /= appear
        scores[:, 3] /= appear
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

def elo_rating_score(data, N, K=32, base=1500, divisor=400):
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

def elo_rating_score_fast(data, N, K=32, base=1500, divisor=400, delta_thres=1e-3):
    rating = np.full(N, base, dtype=np.float64)
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

def entropy_weighted_score(data, N):
    scores = np.zeros((N, 2))
    n = data[:, 2] + data[:, 3]
    n = np.where(n == 0, 1, n)
    p, q = data[:, 2] / n, data[:, 3] / n
    s = p - q
    ent = -(p * np.log2(p + 1e-10) + q * np.log2(q + 1e-10))
    for idx, row in enumerate(data):
        i, j, pv, nv, tv = row
        if pv + nv == 0:
            continue
        scores[i, 0] += s[idx] * ent[idx]
        scores[j, 0] -= s[idx] * ent[idx]
        scores[i, 1] += ent[idx]
        scores[j, 1] += ent[idx]
    return scores[:, 0] / (scores[:, 1] + 1e-10)

'''
performance
condition: po_load(2000), 725557 partial orders
classical_score: 11.58s
bradley_terry_score: 315.08s
random_walk_score: 5.08s
elo_rating_score: 21.44s
entropy_weighted_score: 2.07s
'''

def full_order():
    po = po_load()
    vid = vid_load()
    N = len(vid)
    vn = read("vn")
    vn = vn[vn['id'].isin(vid)]
    appear = np.zeros(N, dtype=np.int16)
    for i, j, pv, nv, tv in po:
        appear[i] += 1
        appear[j] += 1

    scores = classical_score(po, N, appear)
    pagerank = random_walk_score(po, N)
    elo = elo_rating_score_fast(po, N)
    entropy = entropy_weighted_score(po, N)

    res = pd.DataFrame({ 'idx': np.arange(N), 'vid': vid, 'total': scores[:, 0], 'percentage': scores[:, 1], 'simple': scores[:, 2], 'weighted_simple': scores[:, 3], 'pagerank': pagerank, 'elo': elo, 'entropy': entropy, 'appear': appear })

    res.to_csv(os.path.join(out_dir, "full_order.csv"), index=False, float_format='%.4f')

def purify(s):
    # only keep Alphanumeric, Chinese and Japanese characters
    s = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff\u3040-\u30ff]', '', s)
    s = s.lower()
    return s

def postfix():
    vid = vid_load()
    vn = read("vn")
    vn = vn[vn['id'].isin(vid)]
    
    res = pd.read_csv(os.path.join(out_dir, "full_order.csv"))
    res = res[['idx', 'vid', 'total', 'percentage', 'simple', 'weighted_simple', 'pagerank', 'elo', 'entropy']]
    assert vn.shape[0] == res.shape[0]
    vn.reset_index(drop=True, inplace=True)

    releases_vn = read("releases_vn") # id	vid	rtype
    releases_producers = read("releases_producers") # id	pid	developer	publisher
    producers = read("producers") # id	type	lang	name	latin	alias	description

    # 1 match res.vid with releases_vn.vid (first match)
    releases_vn_first = releases_vn.drop_duplicates(subset=['vid'], keep='first')
    res = res.merge(releases_vn_first[['vid', 'id']], on='vid', how='left')
    # 2 match releases_vn.id with releases_producers.id where releases_producers.developer is true (first match)
    releases_producers_dev = releases_producers[releases_producers['developer'] == 1]
    releases_producers_dev_first = releases_producers_dev.drop_duplicates(subset=['id'], keep='first')
    res = res.merge(releases_producers_dev_first[['id', 'pid']], on='id', how='left')
    # 3 match releases_producers.pid with producers.id (should be unique)
    producers = producers.rename(columns={'id': 'pid', 'name': 'p_name', 'latin': 'p_latin', 'alias': 'p_alias'})
    res = res.merge(producers[['pid', 'p_name', 'p_latin', 'p_alias']], on='pid', how='left')
    res.drop(columns=['id'], inplace=True)

    res = res.merge(vn[['id', 'alias', 'c_votecount', 'c_rating', 'c_average']], left_on='vid', right_on='id', how='left')
    vn_titles = read("vn_titles") # id	lang	official	title	latin
    olang = vn['olang']
    _ja, _zh, _en = vn_titles[vn_titles['lang'] == 'ja'], vn_titles[vn_titles['lang'] == 'zh-Hans'], vn_titles[vn_titles['lang'] == 'en']
    res['title_ja'] = res['vid'].map(_ja.set_index('id')['title'])
    res['title_en'] = res['vid'].map(_en.set_index('id')['title'])
    res['title_zh'] = res['vid'].map(_zh.set_index('id')['title'])
    
    for i in range(len(res)):
        zh_q = pd.isna(res['title_zh'][i])
        if zh_q and not pd.isna(res['alias'][i]):
            alias = res['alias'][i]
            _ = alias.split('\\n')
            _ = [a for a in _ if any('\u4e00' <= c <= '\u9fff' for c in a) and not any('\u3040' <= c <= '\u30ff' for c in a)]
            # _ = sorted(_, key=len, reverse=True)
            if len(_) > 0:
                res.loc[i, 'title_zh'] = _[0]
                zh_q = False

        if pd.isna(res['title_en'][i]) and pd.isna(res['title_ja'][i]):
            olang_title = vn_titles[(vn_titles['id'] == res['vid'][i]) & (vn_titles['lang'] == olang[i])]
            if len(olang_title) > 0:
                res.loc[i, 'title_en'] = olang_title.iloc[0]['latin']
                res.loc[i, 'title_ja'] = olang_title.iloc[0]['title']

        # if pd.isna(res['title_en'][i]):
        #     olang_title = vn_titles[(vn_titles['id'] == res['vid'][i]) & (vn_titles['lang'] == olang[i])]
        #     if len(olang_title) > 0:
        #         res.loc[i, 'title_en'] = olang_title.iloc[0]['latin']
        
        if zh_q and not pd.isna(res['title_en'][i]):
            res.loc[i, 'title_zh'] = res['title_en'][i]
    
    res[['p_name', 'p_latin', 'p_alias', 'title_ja', 'title_en', 'title_zh']] = res[['p_name', 'p_latin', 'p_alias', 'title_ja', 'title_en', 'title_zh']].fillna('')
    res['search'] = res['title_ja'].astype(str) + res['title_en'].astype(str) + res['title_zh'].astype(str) + res['alias'].astype(str) + res['p_name'].astype(str) + res['p_latin'].astype(str) + res['p_alias'].astype(str)
    res['search'] = res['search'].apply(purify)
    res.drop(columns=['id', 'p_latin', 'p_alias'], inplace=True)
    res['c_rating'] = res['c_rating'].astype(np.int16) / 100
    res['c_average'] = res['c_average'].astype(np.int16) / 100
    res.sort_values(by=['c_rating', 'c_average'], ascending=[False, False], inplace=True)
    res['rank'] = np.arange(1, len(res) + 1)
    res.to_csv(os.path.join(out_dir, "full_order.csv"), index=False, float_format='%.3f')

def _ulist_vns():
    ulist_vns = read("ulist_vns") # uid	vid	added	lastmod	vote_date	started	finished	vote	notes	labels
    print(f"ulist_vns.shape: {ulist_vns.shape}")
    vid = vid_load()
    ulist_vns = ulist_vns[(ulist_vns['vid'].isin(vid))]
    # labels is a int array, e.g. {2,7}
    dist = pd.read_csv(os.path.join(out_dir, "dist.csv"))
    dist['vid'] = vid
    # for each work; for labels in 1,2,3,4,5; count the appearance of each label (Playing, Finished, Stalled, Dropped, Wishlist)
    def parse_labels(label_str):
        try:
            return [int(x) for x in label_str.strip('{}').split(',') if x.strip().isdigit()]
        except:
            return []
    ulist_vns['label_set'] = ulist_vns['labels'].apply(parse_labels)
    for l in range(1, 6):
        ulist_vns[f'l{l}'] = ulist_vns['label_set'].apply(lambda s: l in s)
    def extract_min(label_set):
        _ = label_set[0]
        return _ if _ <= 5 else 0
    ulist_vns['state'] = ulist_vns['label_set'].apply(extract_min)
    # aggregate l1, l2, l3, l4, l5 and total collection sum and save to dist
    ulist_vid = ulist_vns.groupby('vid')
    dist['collection'] = ulist_vid['vid'].transform('count')
    dist[['l1', 'l2', 'l3', 'l4', 'l5']] = ulist_vid[['l1', 'l2', 'l3', 'l4', 'l5']].transform('sum')
    dist[['collection', 'l1', 'l2', 'l3', 'l4', 'l5']] = dist[['collection', 'l1', 'l2', 'l3', 'l4', 'l5']].fillna(0).astype(np.int16)

    ulist_vns = ulist_vns[ulist_vns['vote'] != '\\N']
    ulist_vns = ulist_vns[['uid', 'vid', 'vote_date', 'vote', 'notes', 'state']]
    ulist_cpy = ulist_vns.copy()
    ulist_cpy = ulist_cpy[(~ulist_cpy['notes'].isna())]
    print(f"ulist_cpy.shape: {ulist_cpy.shape}")
    uid_list = ulist_cpy['uid'].unique()
    ulist_vns = ulist_vns[ulist_vns['uid'].isin(uid_list)]
    print(f"ulist_vns.shape: {ulist_vns.shape}")

    dist.to_csv(os.path.join(out_dir, "dist.csv"), index=False, float_format='%.3f')
    ulist_vns.to_csv(os.path.join(out_dir, "ulist_vns.csv"), index=False)

def minimize_dataset():
    _ulist_vns()

def main():
    # partial_order()
    # full_order()
    # postfix()
    minimize_dataset()

if __name__ == "__main__":
    main()