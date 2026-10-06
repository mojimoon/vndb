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

# 检查一个记录是否存在
def debug_exists(df, column, value):
    print(f"{value} exists in {column}: {not df[df[column] == value].empty}")

vn = prune_id(load("vn", dirty_quote=True))
# id	image	c_image	olang	c_votecount	c_rating	c_average	c_length	c_lengthnum	length	devstatus	alias	description

releases = prune_id(load("releases", dirty_quote=True))
# id	gtin	olang	released	voiced	reso_x	reso_y	minage	ani_story	ani_ero	ani_story_sp	ani_story_cg	ani_cutscene	ani_ero_sp	ani_ero_cg	ani_bg	ani_face	has_ero	patch	freeware	uncensored	official	catalog	notes	engine

releases_platforms = prune_id(load("releases_platforms"))
# id	platform

releases_producers = prune_id(load("releases_producers"), columns=['id', 'pid'])
# id	pid	developer	publisher

releases_titles = prune_id(load("releases_titles"))
# id	lang	mtl	title	latin

releases_vn = prune_id(load("releases_vn"), columns=['id', 'vid'])
# id	vid	rtype

producers = prune_id(load("producers"))
# id	type	lang	name	latin	alias	description

# 找到所有有官中的游戏
vn = vn[(vn['olang'] == 'ja') & (vn['devstatus'] == 0)] # 日文原版 + 已发售
vn_ids = vn['id'].tolist()
# print(f'vn_ids: {len(vn_ids)}') # 33733
releases = releases[(releases['official'] == 't')] # 官方
releases_platforms = releases_platforms[releases_platforms['platform'] == 'win'] # 仅限Windows平台
rp_ids = releases_platforms['id'].unique().tolist()
releases = releases[releases['id'].isin(rp_ids)]
# print(f'releases after platform filter: {len(releases)}') # 94020
releases_vn = releases_vn[releases_vn['rtype'] == 'complete'] # 完整版
# 对 releases_titles 进行预填充：如果一个 zh-Hans 记录的标题是 \N，使用同一个 id 的第一个非 \N 标题进行填充
# def fill_title(group):
#     if group.empty:
#         return group
#     first_valid_title = group[group['title'] != r'\N']['title'].iloc[0] if not group[group['title'] != r'\N'].empty else None
#     if first_valid_title is not None:
#         group.loc[group['title'] == r'\N', 'title'] = first_valid_title
#     return group
# releases_titles = releases_titles.groupby('id').apply(fill_title).reset_index(drop=True)
releases_titles = releases_titles[(releases_titles['lang'] == 'zh-Hans') & (releases_titles['mtl'] == 'f')] # 简中 + 非机器翻译
# 只保留对应一个 vn 的 releases
# 把 vid 和 title 合并到 releases 上，drop 掉不具有这两个字段的记录
releases = releases.merge(releases_vn[['id', 'vid']], on='id', how='inner')
releases = releases[releases['vid'].isin(vn_ids)]
releases = releases.merge(releases_titles[['id', 'title']], on='id', how='inner')
# print(f'releases after title and vn merge: {len(releases)}') # 1669
releases = releases[releases['minage'] != 18]
releases_base = releases[releases['patch'] == 'f']
releases_patch = releases[releases['patch'] == 't']
with open(os.path.join(tmp, "base.txt"), "w", encoding="utf-8") as f:
    for _, row in releases_base.iterrows():
        f.write(f"{row['id']}\t{row['vid']}\t{row['title']}\n")
with open(os.path.join(tmp, "patch.txt"), "w", encoding="utf-8") as f:
    for _, row in releases_patch.iterrows():
        f.write(f"{row['id']}\t{row['vid']}\t{row['title']}\n")
