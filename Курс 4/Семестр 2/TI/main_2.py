import os
import heapq
from collections import Counter
import math


class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(freq):
    heap = [HuffmanNode(char, f) for char, f in freq.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)
    return heap[0]


def generate_huffman_codes(node, code="", codes=None):
    if codes is None:
        codes = {}
    if node.char is not None:
        codes[node.char] = code
        return codes
    generate_huffman_codes(node.left, code + "0", codes)
    generate_huffman_codes(node.right, code + "1", codes)
    return codes


def huffman_encode(text):
    freq = Counter(text)
    tree = build_huffman_tree(freq)
    codes = generate_huffman_codes(tree)
    encoded = ''.join(codes[char] for char in text)
    return codes, encoded, freq


def shannon_encode(text):
    freq_count = Counter(text)
    symbols = sorted(freq_count.items(), key=lambda x: x[1], reverse=True)
    codes = {}
    stack = [(symbols, "")]

    while stack:
        group, code = stack.pop()
        if len(group) == 1:
            codes[group[0][0]] = code
            continue

        best_split = 1
        min_diff = float('inf')
        for split in range(1, len(group)):
            left_p = sum(f for _, f in group[:split])
            right_p = sum(f for _, f in group[split:])
            diff = abs(left_p - right_p)
            if diff < min_diff:
                min_diff = diff
                best_split = split

        stack.append((group[:best_split], code + "0"))
        stack.append((group[best_split:], code + "1"))

    encoded = ''.join(codes[char] for char in text)
    return codes, encoded, freq_count


def entropy_n_grams(encoded, n):
    if len(encoded) < n:
        return 0.0

    ngrams = [encoded[i:i + n] for i in range(len(encoded) - n + 1)]
    freq = Counter(ngrams)
    total = len(ngrams)

    H = 0.0
    for count in freq.values():
        p = count / total
        H -= p * math.log2(p)

    return H / n


def average_code_length(codes, freq):
    total = sum(freq.values())
    return sum(len(codes[char]) * freq[char] / total for char in freq)


def redundancy(h, lcp):
    return lcp / h - 1 if h > 0 else 0


def process_file(filename):
    if not os.path.exists(filename):
        print(f"Файл {filename} не найден!")
        return None

    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"\n{'=' * 80}")
    print(f"ОБРАБОТКА: {filename}")
    print(f"Длина текста: {len(text)} символов")

    huff_codes, huff_encoded, huff_freq = huffman_encode(text)
    sf_codes, sf_encoded, sf_freq = shannon_encode(text)

    huff_lcp = average_code_length(huff_codes, huff_freq)
    sf_lcp = average_code_length(sf_codes, sf_freq)
    text_H1 = entropy_n_grams(text, 1)
    huff_gamma = redundancy(text_H1, huff_lcp)
    sf_gamma = redundancy(text_H1, sf_lcp)

    huff_H1 = entropy_n_grams(huff_encoded, 1)
    huff_H2 = entropy_n_grams(huff_encoded, 2)
    huff_H3 = entropy_n_grams(huff_encoded, 3)

    sf_H1 = entropy_n_grams(sf_encoded, 1)
    sf_H2 = entropy_n_grams(sf_encoded, 2)
    sf_H3 = entropy_n_grams(sf_encoded, 3)

    with open(filename + '_huff.txt', 'w') as f:
        f.write(huff_encoded)
    with open(filename + '_sf.txt', 'w') as f:
        f.write(sf_encoded)

    result = {
        'file': os.path.splitext(os.path.basename(filename))[0],
        'huff_lcp': huff_lcp, 'huff_gamma': huff_gamma,
        'huff_H1': huff_H1, 'huff_H2': huff_H2, 'huff_H3': huff_H3,
        'sf_lcp': sf_lcp, 'sf_gamma': sf_gamma,
        'sf_H1': sf_H1, 'sf_H2': sf_H2, 'sf_H3': sf_H3
    }

    print(f"Хаффман: L_cp={huff_lcp:.4f}, γ={huff_gamma:.4f}")
    print(f"  H₁={huff_H1:.4f}, H₂={huff_H2:.4f}, H₃={huff_H3:.4f}")
    print(f"Шеннон:  L_cp={sf_lcp:.4f}, γ={sf_gamma:.4f}")
    print(f"  H₁={sf_H1:.4f}, H₂={sf_H2:.4f}, H₃={sf_H3:.4f}")

    return result

files = ['onegin_clean.txt', 'random_biased.txt', 'random_equal.txt']
results = []

for filename in files:
    result = process_file(filename)
    if result:
        results.append(result)

print("\n" + "Метод        Текст                      L_cp    γ       H₁      H₂      H₃")
print("-" * 130)

for r in results:
    print(f"Хаффман     {r['file']:<25} {r['huff_lcp']:<7.4f} {r['huff_gamma']:<7.4f} "
          f"{r['huff_H1']:<7.4f} {r['huff_H2']:<7.4f} {r['huff_H3']:<7.4f}")
    print(f"Шеннон      {r['file']:<25} {r['sf_lcp']:<7.4f} {r['sf_gamma']:<7.4f} "
          f"{r['sf_H1']:<7.4f} {r['sf_H2']:<7.4f} {r['sf_H3']:<7.4f}")
    print()
