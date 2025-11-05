import os
import pickle

PACKED_BLOCKS_CACHE = "datasets/bnc_packed_blocks_cache.pkl"  # cache of packed blocks to skip re-packing

def load_packed_blocks_cache():
    if os.path.exists(PACKED_BLOCKS_CACHE):
        with open(PACKED_BLOCKS_CACHE, 'rb') as f:
            packed_blocks = pickle.load(f)
        return packed_blocks['blocks']
    else:
        print("No packed blocks cache found.")
        return None

def save_packed_blocks_as_txt(blocks):
    with open("packed_bnc_blocks.txt", 'w', encoding='utf-8') as f:
        for block in blocks:
            f.write(f"{block}\n")

def main():
    packed_blocks = load_packed_blocks_cache()
    if packed_blocks is not None:
        # Use the loaded packed blocks
        print("Loaded packed blocks from cache.")
    else:
        # Handle the case where no cached blocks are found
        print("No cached packed blocks found.")
    save_packed_blocks_as_txt(packed_blocks)

if __name__ == "__main__":
    main()