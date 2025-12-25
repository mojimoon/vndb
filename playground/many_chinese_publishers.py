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

# 移除一个id字段的前缀字母，并转换为整数
def prune_id(df, columns=['id']):
    for column in columns:
        df[column] = df[column].str[1:].astype(int)
    return df

vn = prune_id(load("vn", dirty_quote=True))
# id	image	c_image	olang	c_votecount	c_rating	c_average	c_length	c_lengthnum	length	devstatus	alias	description

releases_vn = prune_id(load("releases_vn"), columns=['id', 'vid'])
# id	vid	rtype

releases = prune_id(load("releases", dirty_quote=True))
# id	gtin	olang	released	voiced	reso_x	reso_y	minage	ani_story	ani_ero	ani_story_sp	ani_story_cg	ani_cutscene	ani_ero_sp	ani_ero_cg	ani_bg	ani_face	has_ero	patch	freeware	uncensored	official	catalog	notes	engine

releases_titles = prune_id(load("releases_titles"))
# id	lang	mtl	title	latin

releases_producers = prune_id(load("releases_producers"), columns=['id', 'pid'])
# id	pid	developer	publisher

producers = prune_id(load("producers"))
# id	type	lang	name	latin	alias	description

'''
Task
找到所有 >= 4 个简体中文发行商 (即汉化组) 的作品
生成 csv，列出作品 id，所有标题，发行商名称
'''

THRES = 4

vn = vn[vn['olang'] == 'ja']
vn_ids = set(vn['id'].tolist())

# 1. 筛选出简中 releases_titles
releases_titles_zh = releases_titles[
    (releases_titles['lang'] == 'zh-Hans') &
    (releases_titles['mtl'] == 'f')
]
releases_titles_zh = releases_titles_zh[['id', 'title']]
print(f"shape of releases_titles_zh: {releases_titles_zh.shape}")
# 2. 和 releases_vn 连接，得到 ['id', 'title', 'vid']
releases_titles_zh = releases_titles_zh.merge(
    releases_vn[['id', 'vid']],
    left_on='id',
    right_on='id',
    how='inner'
)
print(f"shape of releases_titles_zh after merge with releases_vn: {releases_titles_zh.shape}")
# 3. 和 releases_producers 连接，得到 ['id', 'title', 'vid', 'pid']
releases_producers_zh = releases_producers[
    (releases_producers['id'].isin(releases_titles_zh['id'])) &
    (releases_producers['publisher'] == 't')
]
releases_titles_producers_zh = releases_titles_zh.merge(
    releases_producers_zh[['id', 'pid']],
    left_on='id',
    right_on='id',
    how='inner'
)
print(f"shape of releases_titles_producers_zh: {releases_titles_producers_zh.shape}")
# 4. 按照 vid 分组，统计 count(pid)
grouped = releases_titles_producers_zh.groupby('vid').agg({
    'pid': 'nunique',
    'id': 'first',
    'title': lambda x: list(x)
}).reset_index()
# 仅日文游戏
grouped = grouped[grouped['vid'].isin(vn_ids)]
grouped = grouped.rename(columns={'pid': 'publisher_count'})
# 5. 筛选出 count(pid) >= THRES 的记录
grouped_filtered = grouped[grouped['publisher_count'] >= THRES]
print(f"shape of grouped_filtered: {grouped_filtered.shape}")
# 6. 展开 pid，获取发行商名称
output_rows = []
for _, row in grouped_filtered.iterrows():
    vid = row['vid']
    release_id = row['id']
    titles = row['title']
    # 获取对应的 pid 列表
    pids = releases_titles_producers_zh[
        releases_titles_producers_zh['vid'] == vid
    ]['pid'].unique().tolist()
    # 获取发行商名称
    publisher_names = producers[
        producers['id'].isin(pids)
    ]['name'].tolist()
    output_rows.append({
        'release_id': release_id,
        'titles': "; ".join(titles),
        'publishers': "; ".join(publisher_names)
    })

output_df = pd.DataFrame(output_rows)
output_csv_path = os.path.join(tmp, "many.csv")
output_df.to_csv(output_csv_path, index=False)
print(f"Output written to {output_csv_path}")