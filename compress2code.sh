python lexicon_compress.py \
        --candidates data/clusters_enriched.json \
        --in datasets/bnc_sentences.txt \
        --out datasets/code_synset_bnc_sentences.txt \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --synset2id data/synset2ids.json \
        --debug-json debug_report.json \
        # --model "answerdotai/ModernBERT-large" \
        --context_window_sents 3 --context_max_tokens 256