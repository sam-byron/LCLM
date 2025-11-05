python lexicon_compress.py \
        --candidates data/clusters.json \
        --in datasets/packed_bnc_blocks.txt \
        --out datasets/embed_packed_bnc_blocks.txt \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --abstain_margin 0.05 \
        --min_score 0.25