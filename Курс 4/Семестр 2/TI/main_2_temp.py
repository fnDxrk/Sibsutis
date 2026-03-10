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
    total = len(text)
    codes = {}
    
    for char, count in freq_count.items():
        p = count / total
        code_length = math.ceil(-math.log2(p) if p > 0 else 0)
        
        code = '0' * (code_length - 1) + '1' if code_length > 0 else ''
        codes[char] = code
    
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
    shan_codes, shan_encoded, shan_freq = shannon_encode(text)

    huff_lcp = average_code_length(huff_codes, huff_freq)
    shan_lcp = average_code_length(shan_codes, shan_freq)
    text_H1 = entropy_n_grams(text, 1)
    huff_gamma = redundancy(text_H1, huff_lcp)
    shan_gamma = redundancy(text_H1, shan_lcp)

    huff_H1 = entropy_n_grams(huff_encoded, 1)
    huff_H2 = entropy_n_grams(huff_encoded, 2)
    huff_H3 = entropy_n_grams(huff_encoded, 10)

    shan_H1 = entropy_n_grams(shan_encoded, 1)
    shan_H2 = entropy_n_grams(shan_encoded, 2)
    shan_H3 = entropy_n_grams(shan_encoded, 10)

    with open(filename + '_huff.txt', 'w') as f:
        f.write(huff_encoded)
    with open(filename + '_shan.txt', 'w') as f:
        f.write(shan_encoded)

    result = {
        'file': os.path.splitext(os.path.basename(filename))[0],
        'huff_lcp': huff_lcp, 'huff_gamma': huff_gamma,
        'huff_H1': huff_H1, 'huff_H2': huff_H2, 'huff_H3': huff_H3,
        'shan_lcp': shan_lcp, 'shan_gamma': shan_gamma,
        'shan_H1': shan_H1, 'shan_H2': shan_H2, 'shan_H3': shan_H3
    }

    print(f"Хаффман: L_cp={huff_lcp:.4f}, γ={huff_gamma:.4f}")
    print(f"  H₁={huff_H1:.4f}, H₂={huff_H2:.4f}, H₃={huff_H3:.4f}")
    print(f"Шеннон:  L_cp={shan_lcp:.4f}, γ={shan_gamma:.4f}")
    print(f"  H₁={shan_H1:.4f}, H₂={shan_H2:.4f}, H₃={shan_H3:.4f}")

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
    print(f"Шеннон      {r['file']:<25} {r['shan_lcp']:<7.4f} {r['shan_gamma']:<7.4f} "
          f"{r['shan_H1']:<7.4f} {r['shan_H2']:<7.4f} {r['shan_H3']:<7.4f}")
    print()

