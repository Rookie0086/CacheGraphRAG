
# extract triplets

```bash
# Multi-process triplet extraction
python extract_triplet.py --start 0 --end 100 &
python extract_triplet.py --start 100 --end 200 &
python extract_triplet.py --start 200 --end 300 &

# Merge all triplet files
python merge_triplets.py

# Clean special characters from triplets
python filter_triplets.py --input tqa_2wiki_triplets0-10000.json --output ./triplets/dep_triplets.json
```
