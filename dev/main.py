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
