import os
import pandas as pd
import numpy as np
import re
import csv

pwd = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(pwd)
dump = os.path.join(root, "db")
tmp = os.path.join(pwd, "tmp")

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

def intid(s):
    try:
        return int(s[1:])
    except:
        return np.nan

def translate_role(s):
    if s[0] == 'm': # main
        return 0
    elif s[0] == 'p': # primary
        return 1
    elif s[0] == 's': # side
        return 2
    elif s[0] == 'a': # appears
        return 3
    return 4

vn = load("vn", dirty_quote=True)
print(f"Loaded vn with shape: {vn.shape}")
vn['id'] = vn['id'].apply(intid)
# id	image	c_image	olang	c_votecount	c_rating	c_average	length	devstatus	alias	description

traits = load("traits")
print(f"Loaded traits with shape: {traits.shape}")
traits['id'] = traits['id'].apply(intid)
# id	gid	gorder	defaultspoil	sexual	searchable	applicable	name	alias	description

chars = load("chars", dirty_quote=True)
print(f"Loaded chars with shape: {chars.shape}")
chars['id'] = chars['id'].apply(intid)
# id	image	bloodt	cup_size	sex	spoil_sex	gender	spoil_gender	main	main_spoil	s_bust	s_waist	s_hip	birthday	height	weight	age	name	latin	alias	description

chars_traits = load("chars_traits")
print(f"Loaded chars_traits with shape: {chars_traits.shape}")
chars_traits['id'] = chars_traits['id'].apply(intid)
chars_traits['tid'] = chars_traits['tid'].apply(intid)
# id	tid	spoil	lie

chars_vns = load("chars_vns")
print(f"Loaded chars_vns with shape: {chars_vns.shape}")
chars_vns['id'] = chars_vns['id'].apply(intid)
chars_vns['vid'] = chars_vns['vid'].apply(intid)
chars_vns['role'] = chars_vns['role'].apply(translate_role)
# id	vid	rid	role	spoil

vn_titles = load("vn_titles")
print(f"Loaded vn_titles with shape: {vn_titles.shape}")
vn_titles['id'] = vn_titles['id'].apply(intid)
# id	lang	official	title	latin

tid = traits[traits['name'] == 'Childhood Friend']['id'].values[0]
print(f'tid: {tid}')
blacklist_tids = traits[traits['name'].isin(['Netorare (they are the stolen SO)', 'Netorase (they are the shared SO)', 'Netori (their SO is stolen)'])]['id'].values
print(f'blacklist_tids: {blacklist_tids}')
chars_t = chars_traits[(chars_traits['tid'] == tid) & (chars_traits['lie'] == 'f')]['id'].values
print(f'chars_t.shape: {chars_t.shape}')
chars_blacklist = chars_traits[chars_traits['tid'].isin(blacklist_tids)]['id'].values
print(f'chars_blacklist.shape: {chars_blacklist.shape}')
chars = chars[chars['id'].isin(chars_t)]
chars = chars[~chars['id'].isin(chars_blacklist)]
print(f'chars.shape: {chars.shape}')
chars_f = chars[chars['sex'] == 'f']
print(f'chars_f.shape: {chars_f.shape}')
chars_f_v = chars_vns[chars_vns['id'].isin(chars_f['id'].values)]
print(f'chars_f_v.shape: {chars_f_v.shape}')
chars_f_v_p = chars_f_v[chars_f_v['role'] == 1]
print(f'chars_f_v_p.shape: {chars_f_v_p.shape}')
# chars_f_v_p_cnt = chars_f_v_p['vid'].value_counts()
# chars_f_v_p_cnt_2 = chars_f_v_p_cnt[chars_f_v_p_cnt >= 2]
# print(f'chars_f_v_p_cnt_2.shape: {chars_f_v_p_cnt_2.shape}')

# rejoin chars_f_v_p with chars to get character names
chars_f_v_p = chars_f_v_p.merge(chars[['id', 'name']], on='id', how='left')
chars_f_v_p_unique = chars_f_v_p.drop_duplicates(subset=['vid', 'name'], keep='first')
chars_f_v_p_cnt = chars_f_v_p_unique['vid'].value_counts()
chars_f_v_p_cnt_2 = chars_f_v_p_cnt[chars_f_v_p_cnt >= 2]
print(f'chars_f_v_p_cnt_2.shape: {chars_f_v_p_cnt_2.shape}')
# use unique list but reset index

chars_m = chars[chars['sex'] == 'm']
print(f'chars_m.shape: {chars_m.shape}')
chars_m_v = chars_vns[chars_vns['id'].isin(chars_m['id'].values)]
print(f'chars_m_v.shape: {chars_m_v.shape}')
chars_m_v_m = chars_m_v[chars_m_v['role'] == 0]
print(f'chars_m_v_m.shape: {chars_m_v_m.shape}')
mm_vids = chars_m_v_m['vid'].unique()
print(f'mm_vids.shape: {mm_vids.shape}')

vids = np.intersect1d(chars_f_v_p_cnt_2.index.values, mm_vids)
print(f'vids.shape: {vids.shape}')
# only japanese vns
vids = vn[(vn['id'].isin(vids)) & (vn['olang'] == 'ja')]['id'].values
print(f'japanese vids.shape: {vids.shape}')

# a nested array to store female character names for each vn
# so for each vn we first find corresponding chars in chars_f_v_p

def array_to_str(arr):
    # remove space in each array element
    arr = [re.sub(r'\s+', '', x) for x in arr]
    return ";".join(arr)

target = []
for vid in vids:
    chars_cur = chars_f_v_p_unique[chars_f_v_p_unique['vid'] == vid]
    chars_names = chars_cur['name'].values
    target.append(array_to_str(chars_names))

# merge ja and zh-Hans titles
result = pd.DataFrame({
    'vid': vids,
    'char_names': target
})
vn_titles_ja = vn_titles[(vn_titles['lang'] == 'ja')][['id', 'title']].rename(columns={'id': 'vid', 'title': 'title_ja'})
vn_titles_zh = vn_titles[(vn_titles['lang'] == 'zh-Hans')][['id', 'title']].rename(columns={'id': 'vid', 'title': 'title_zh'})
result = result.merge(vn_titles_ja, on='vid', how='left')
result = result.merge(vn_titles_zh, on='vid', how='left')
result['cnt'] = result['char_names'].apply(lambda x: len(x.split(';')))
result = result.merge(vn[['id', 'c_votecount', 'c_rating']], left_on='vid', right_on='id', how='left').drop(columns=['id'])
result['c_rating'] = result['c_rating'].apply(lambda x: round(int(x) / 100, 2) if x != '\\N' else np.nan)
result = result[['vid', 'title_ja', 'title_zh', 'c_votecount', 'c_rating', 'cnt', 'char_names']]
# sort by cnt desc
result = result.sort_values(by='cnt', ascending=False).reset_index(drop=True)

result.to_csv(os.path.join(tmp, "multiple_childhood_friends.csv"), index=False, encoding='utf-8-sig', sep=',')
print(f"Saved result with shape: {result.shape}")
