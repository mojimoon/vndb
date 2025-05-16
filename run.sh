curl -L -o db.tar.zst https://dl.vndb.org/dump/vndb-db-latest.tar.zst
if [ -d "db" ]; then
    rm -rf db
fi
mkdir db
tar -I zstd -xf db.tar.zst -C db/
rm db.tar.zst
