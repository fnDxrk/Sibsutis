import random

print("Генерация random_equal.txt...")
with open('random_equal.txt', 'w') as f:
    for _ in range(20000):
        char = random.choice('ABCD')
        f.write(char)

print("Генерация random_biased.txt...")
chars = list('ABCD')
probs = [0.1, 0.2, 0.3, 0.4]

with open('random_biased.txt', 'w') as f:
    for _ in range(20000):
        char = random.choices(chars, weights=probs)[0]
        f.write(char)

print("Файлы созданы: random_equal.txt, random_biased.txt")
print("Размеры:")
print(f"random_equal.txt: {len(open('random_equal.txt').read())} символов")
print(f"random_biased.txt: {len(open('random_biased.txt').read())} символов")
