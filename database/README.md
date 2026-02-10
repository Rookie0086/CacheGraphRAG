## database tool

> database info

```bash
# kg database info
python -m database.db-tool --dim 1024 --db test --info kg

# vector database info
python db-tool.py --dim 1024 --db test --info vector

# both kg and vector database info
python db-tool.py --dim 1024 --db test --info all
```

> clear database

```bash
# clear kg database
python db-tool.py --dim 1024 --db test --clear kg

# clear vector database
python db-tool.py --dim 1024 --db test --clear vector

# clear both kg and vector database
python db-tool.py --dim 1024 --db test --clear all
```

> delete database

```bash
# delete kg database
python db-tool.py --dim 1024 --db test --delete kg

# delete vector database
python db-tool.py --dim 1024 --db test --delete vector

# delete both kg and vector database
python db-tool.py --dim 1024 --db test --delete all
```


> save kg triplets

```bash
 python db-tool.py --dim 1024 --db test --save_kg triplets
```


## insert triplet in NebulaGraph

use `tqa` dataset as an example.

> Method 1: use nebulagraph api



```bash
# craete a new database if not exist
python db-tool.py --dim 1024 --db tqa  --create kg


# Method 1: use nebulagraph api
# python insert_triples.py --data tqa --db tqa
python insert_triplets.py  --db rgb_fact --proc 1 --input ../triplet/rgb_fact/rgb_en_fact_triplets.json
```

> Method 2: NebulaGraph importer

https://docs.nebula-graph.io/3.4.0/nebula-importer/use-importer/

```bash
## Convert triplets to CSV format
python triplet_csv.py --data tqa

## Load triplets with nebula-importer
./nebula-importer-linux-amd64-v3.4.0 --config nebula_import.yaml
```
