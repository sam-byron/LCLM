python lexicon_compress.py \
        --candidates data/clusters.json \
        --in datasets/bnc_sentences_bnc_sentences_sampled_1.0.txt \
        --out datasets/embed_bnc_sentences_bnc_sentences_sampled_1.0.txt \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --abstain_margin 0.05 \
        --min_score 0.25 \
        --devices auto --nlp_processes 96