curl -L -o db.tar.zst https://dl.vndb.org/dump/vndb-db-latest.tar.zst
mkdir -p db
tar -I zstd -xf db.tar.zst -C db/
rm db.tar.zst

pip install -r requirements.txt
python main.py

mkdir -p web/data
mv out/dist.csv web/data/
mv out/full_order.csv web/data/
