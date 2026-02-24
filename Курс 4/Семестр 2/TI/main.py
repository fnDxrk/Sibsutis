import math
from collections import Counter
import sys

def shannon_entropy(probs):
    return -sum(p * math.log2(p) for p in probs if p > 0)

def estimate_entropy(filename, n=1):
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    
    if len(text) < n:
        return 0.0
    
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    counts = Counter(ngrams)
    total = len(ngrams)
    
    probs = [count / total for count in counts.values()]
    H_n = shannon_entropy(probs)
    return H_n / n

def process_file(filename):
    print(f"\n--- {filename} ---")
    H1 = estimate_entropy(filename, 1)
    H2 = estimate_entropy(filename, 2)
    H3 = estimate_entropy(filename, 3)
    print(f"H1 (1-gram): {H1:.4f}")
    print(f"H2 (2-gram): {H2:.4f}")
    print(f"H3 (3-gram): {H3:.4f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for fname in sys.argv[1:]:
            process_file(fname)
    else:
        print("Использование: python entropy.py file1.txt file2.txt file3.txt")

