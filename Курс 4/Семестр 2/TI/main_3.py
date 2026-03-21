import random
import math
from collections import Counter
import heapq

random.seed(42)
chars = list('ABCD')
probs_theoretical = [0.1, 0.2, 0.3, 0.4]
with open('random_biased.txt', 'w') as f:
    for _ in range(10000):
        f.write(random.choices(chars, weights=probs_theoretical)[0])

print("=== Файл сгенерирован ===")
print(f"random_biased.txt: {len(open('random_biased.txt').read())} символов")
print("Распределение: P(A)=0.1, P(B)=0.2, P(C)=0.3, P(D)=0.4")

class Node:
    def __init__(self, sym, freq):
        self.sym = sym
        self.freq = freq
        self.left = self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_codes(freq_dict):
    heap = [Node(s, f) for s, f in freq_dict.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        return {heap[0].sym: '0'}
    while len(heap) > 1:
        l, r = heapq.heappop(heap), heapq.heappop(heap)
        parent = Node(None, l.freq + r.freq)
        parent.left, parent.right = l, r
        heapq.heappush(heap, parent)
    codes = {}
    def traverse(node, code=''):
        if node.sym is not None:
            codes[node.sym] = code or '0'
            return
        traverse(node.left, code + '0')
        traverse(node.right, code + '1')
    traverse(heap[0])
    return codes

def block_encode_analysis(text, n):
    num_blocks = len(text) // n
    blocks = [text[i*n:(i+1)*n] for i in range(num_blocks)]

    block_counts = Counter(blocks)
    block_probs = {b: c / num_blocks for b, c in block_counts.items()}

    H_blocks = -sum(p * math.log2(p) for p in block_probs.values())
    codes = build_huffman_codes(block_counts)
    avg_len_per_block = sum(block_probs[b] * len(codes[b]) for b in block_probs)

    avg_len_per_sym = avg_len_per_block / n
    H_per_sym       = H_blocks / n
    redundancy      = avg_len_per_sym - H_per_sym

    return {
        'n':               n,
        'num_unique':      len(block_counts),
        'H_per_sym':       H_per_sym,
        'avg_len_per_sym': avg_len_per_sym,
        'redundancy':      redundancy,
        'codes':           codes
    }

H_theory = -sum(p * math.log2(p) for p in probs_theoretical)

text = open('random_biased.txt').read()

results = []
for n in [1, 2, 3, 4]:
    r = block_encode_analysis(text, n)
    results.append(r)

print("\n=== ИТОГОВАЯ ТАБЛИЦА ===")
print(f"{'Метрика':<35} | {'n=1':>8} | {'n=2':>8} | {'n=3':>8} | {'n=4':>8}")
print("-" * 75)
print(f"{'Уникальных блоков':<35} | " + " | ".join(f"{r['num_unique']:>8}" for r in results))
print(f"{'H на символ (бит)':<35} | " + " | ".join(f"{r['H_per_sym']:>8.4f}" for r in results))
print(f"{'Ср. длина кода на символ (бит)':<35} | " + " | ".join(f"{r['avg_len_per_sym']:>8.4f}" for r in results))
print(f"{'Избыточность на символ (бит)':<35} | " + " | ".join(f"{r['redundancy']:>8.4f}" for r in results))
print(f"\nТеоретическая H = {H_theory:.4f} бит/символ")
