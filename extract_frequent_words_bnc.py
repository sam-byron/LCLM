import os
import pickle
from load_packed_bnc import load_packed_blocks_cache

MIN_TOTAL_WORD_COVERAGE_PERCENT = 0.96
packed_blocks = load_packed_blocks_cache()
def extract_frequent_words(packed_blocks, min_coverage=MIN_TOTAL_WORD_COVERAGE_PERCENT):
    word_freq = {}
    for block in packed_blocks:
        for word in block['text'].lower().split():
            if not word.isalpha():
                continue
            word_freq[word] = word_freq.get(word, 0) + 1
    word_freq_items = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    total_word_count = sum(word_freq.values())
    cumulative_count = 0
    frequent_words = set()
    for word, count in word_freq_items:
        cumulative_count += count
        if cumulative_count / total_word_count >= MIN_TOTAL_WORD_COVERAGE_PERCENT:
            break
        else:
            frequent_words.add(word)
    return frequent_words

def main():
    frequent_words = extract_frequent_words(packed_blocks)
    print(f"Extracted {len(frequent_words)} frequent words covering at least {MIN_TOTAL_WORD_COVERAGE_PERCENT*100}% of total word occurrences.")
    # Optionally, save the frequent words to a file
    with open("frequent_words.txt", "w") as f:
        for word in sorted(frequent_words):
            f.write(f"{word}\n")

if __name__ == "__main__":
    main()
