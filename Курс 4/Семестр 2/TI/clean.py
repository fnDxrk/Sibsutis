def normalize_russian(text):
    trans = str.maketrans('ёЁЪъ', 'еееь', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    result = text.translate(trans).lower()
    return ''.join(c for c in result if 'а' <= c <= 'я' or c == ' ')

input_file = 'onegin_clean.txt'

with open(input_file, 'r', encoding='utf-8') as f:
    raw_text = f.read()

cleaned = normalize_russian(raw_text)
multiplier = max(1, 15000 // len(cleaned) + 1)
final_text = cleaned * multiplier

with open('onegin_clean.txt', 'w', encoding='utf-8') as f:
    f.write(final_text)

