import os

MIN_TOTAL_WORD_COVERAGE_PERCENT = 0.96
def load_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

# mapped_bnc = load_dataset('datasets/mapped_bnc_sentences.txt')
mapped_bnc = load_dataset('datasets/embed_synset_bnc_sentences.txt')
# mapped_bnc = load_dataset('datasets/mapped_packed_bnc_blocks.txt')
def count_frequent_words(mapped_bnc, min_coverage=MIN_TOTAL_WORD_COVERAGE_PERCENT):
    word_freq = {}
    for line in mapped_bnc:
        for word in line.lower().split():
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
    print(f"Selected {len(frequent_words)} frequent words covering {cumulative_count}/{total_word_count} words ({cumulative_count/total_word_count:.2%})")

def main():
    count_frequent_words(mapped_bnc)

if __name__ == "__main__":
    main()
