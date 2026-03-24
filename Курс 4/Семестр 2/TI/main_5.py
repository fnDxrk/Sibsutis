import random
from collections import Counter
import heapq

def get_huffman_codes(text):
    if not text:
        return {}
    
    freq = Counter(text)
    
    class Node:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None
        def __lt__(self, other):
            return self.freq < other.freq
    
    heap = [Node(char, f) for char, f in freq.items()]
    heapq.heapify(heap)

    if len(heap) == 1:
        node = heapq.heappop(heap)
        return {node.char: "0"}

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    root = heap[0]
    codes = {}
    
    def generate_codes(node, current_code):
        if node.char is not None:
            codes[node.char] = current_code
            return
        if node.left:
            generate_codes(node.left, current_code + "0")
        if node.right:
            generate_codes(node.right, current_code + "1")

    generate_codes(root, "")
    return codes

def encode_huffman(text):
    codes = get_huffman_codes(text)
    encoded = "".join(codes[ch] for ch in text)
    return encoded, codes

def hamming_encode_7_4(data_bits):
    if len(data_bits) != 4:
        raise ValueError("Нужно 4 бита данных")
    
    d = [int(b) for b in data_bits]
    
    p1 = d[0] ^ d[1] ^ d[3]
    p2 = d[0] ^ d[2] ^ d[3]
    p3 = d[1] ^ d[2] ^ d[3]
    
    codeword = [p1, p2, d[0], p3, d[1], d[2], d[3]]
    return "".join(map(str, codeword))

def hamming_decode_7_4(codeword):
    if len(codeword) != 7:
        raise ValueError("Нужно 7 бит")
    
    r = [int(b) for b in codeword]
    
    s1 = r[0] ^ r[2] ^ r[4] ^ r[6]
    s2 = r[1] ^ r[2] ^ r[5] ^ r[6]
    s3 = r[3] ^ r[4] ^ r[5] ^ r[6]
    
    error_pos = s1 * 1 + s2 * 2 + s3 * 4
    
    if error_pos != 0:
        r[error_pos - 1] ^= 1
    
    data = [r[2], r[4], r[5], r[6]]
    return "".join(map(str, data))

def hamming_encode_file(bits):
    padding = (4 - len(bits) % 4) % 4
    bits_padded = bits + "0" * padding
    
    encoded = ""
    for i in range(0, len(bits_padded), 4):
        block = bits_padded[i:i+4]
        encoded += hamming_encode_7_4(block)
    
    return encoded, padding

def hamming_decode_file(encoded_bits, padding):
    decoded = ""
    for i in range(0, len(encoded_bits), 7):
        block = encoded_bits[i:i+7]
        if len(block) == 7:
            decoded += hamming_decode_7_4(block)
    
    if padding > 0:
        decoded = decoded[:-padding]
    
    return decoded

def add_noise(bits, p):
    noisy = ""
    for b in bits:
        if random.random() < p:
            noisy += "1" if b == "0" else "0"
        else:
            noisy += b
    return noisy

def count_errors(original, decoded):
    errors = 0
    min_len = min(len(original), len(decoded))
    for i in range(min_len):
        if original[i] != decoded[i]:
            errors += 1
    errors += abs(len(original) - len(decoded))
    return errors

if __name__ == "__main__":
    with open("file2.txt", "r", encoding="utf-8") as f:
        original_text = f.read().replace('\n', '').replace('\r', '')

    huffman_bits, codes = encode_huffman(original_text)
    print(f"Исходный текст: {len(original_text)} символов")
    print(f"После Хаффмана: {len(huffman_bits)} бит")

    hamming_encoded, padding = hamming_encode_file(huffman_bits)
    print(f"После Хэмминга (7,4): {len(hamming_encoded)} бит")
    print()



    probabilities = [0.0001, 0.001, 0.01, 0.1]
    
    print(f"{'Вероятность ошибки':<25} | ", end="")
    for p in probabilities:
        print(f"{'p = ' + str(p):<15} | ", end="")
    print()
    print("-" * 95)
    
    print(f"{'Количество ошибок':<25} | ", end="")
    
    results = []
    for p in probabilities:
        noisy = add_noise(hamming_encoded, p)
        
        decoded_hamming = hamming_decode_file(noisy, padding)
        
        errors = count_errors(huffman_bits, decoded_hamming)
        results.append(errors)
        print(f"{errors:<15} | ", end="")
    
    print()
