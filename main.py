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

if __name__ == "__main__":
    read("vn")