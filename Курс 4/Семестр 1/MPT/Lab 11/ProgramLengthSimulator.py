import random
import math
import statistics
import tokenize
import io
import keyword
import builtins


class ProgramLengthSimulator:
    def __init__(self):
        self.results = {}
        self.python_keywords = set(keyword.kwlist)
        self.builtin_names = set(dir(builtins))

    def simulate_program_writing(self, eta, num_trials=10000):
        """
        Моделирование процесса написания программы
        eta - размер словаря программы
        num_trials - количество испытаний
        """
        lengths = []

        for _ in range(num_trials):
            unique_items = set()
            program_length = 0

            while len(unique_items) < eta:
                item = random.randint(1, eta)
                program_length += 1
                unique_items.add(item)

            lengths.append(program_length)

        mean_length = statistics.mean(lengths)
        variance = statistics.variance(lengths)
        std_dev = statistics.stdev(lengths)
        relative_error = std_dev / mean_length if mean_length > 0 else 0

        self.results[eta] = {
            "experimental": {
                "mean_length": mean_length,
                "variance": variance,
                "std_dev": std_dev,
                "relative_error": relative_error,
            },
            "theoretical": self.calculate_theoretical_values(eta),
        }

        return lengths

    def calculate_theoretical_values(self, eta):
        """Вычисление теоретических значений по формулам"""
        theoretical_length = 0.9 * eta * math.log2(eta)
        theoretical_variance = (math.pi**2 * eta**2) / 6
        theoretical_std_dev = math.sqrt(theoretical_variance)
        theoretical_relative_error = 1 / (2 * math.log2(eta)) if eta > 1 else 0
        alternative_length = eta * math.log2(eta)

        return {
            "mean_length": theoretical_length,
            "variance": theoretical_variance,
            "std_dev": theoretical_std_dev,
            "relative_error": theoretical_relative_error,
            "alternative_length": alternative_length,
        }

    def analyze_program_text(self, program_text):
        """
        Корректный лексический анализ текста программы
        Возвращает словарь и длину в терминах операторов и операндов
        """
        operators = set()
        operands = set()
        total_tokens = 0

        try:
            f = io.BytesIO(program_text.encode("utf-8"))
            for tok in tokenize.tokenize(f.readline):
                if tok.type in (
                    tokenize.COMMENT,
                    tokenize.NL,
                    tokenize.NEWLINE,
                    tokenize.ENCODING,
                    tokenize.ENDMARKER,
                ):
                    continue

                total_tokens += 1
                token_str = tok.string

                if tok.type == tokenize.OP:
                    operators.add(token_str)
                elif tok.type == tokenize.NAME:
                    if token_str in self.python_keywords:
                        operators.add(token_str)
                    elif token_str in self.builtin_names:
                        operands.add(token_str)
                    else:
                        operands.add(token_str)
                elif tok.type in (tokenize.NUMBER, tokenize.STRING):
                    operands.add(token_str)

        except tokenize.TokenError:
            pass

        eta = len(operators) + len(operands)
        actual_length = total_tokens

        if eta > 1:
            predicted_length_1 = 0.9 * eta * math.log2(eta)
            predicted_length_2 = eta * math.log2(eta)
        else:
            predicted_length_1 = predicted_length_2 = 0

        error_1 = (
            abs(actual_length - predicted_length_1) / actual_length * 100
            if actual_length > 0
            else 0
        )
        error_2 = (
            abs(actual_length - predicted_length_2) / actual_length * 100
            if actual_length > 0
            else 0
        )

        return {
            "eta": eta,
            "actual_length": actual_length,
            "predicted_length_1": predicted_length_1,
            "predicted_length_2": predicted_length_2,
            "error_1": error_1,
            "error_2": error_2,
            "operators_count": len(operators),
            "operands_count": len(operands),
        }

    def analyze_eta2_star(self, program_text):
        """
        Определение η₂* — уникальные операнды
        (переменные, параметры, константы),
        исключая ключевые слова и встроенные функции.
        """
        operands = set()

        try:
            f = io.BytesIO(program_text.encode("utf-8"))
            for tok in tokenize.tokenize(f.readline):
                if tok.type in (
                    tokenize.COMMENT,
                    tokenize.NL,
                    tokenize.NEWLINE,
                    tokenize.ENCODING,
                    tokenize.ENDMARKER,
                ):
                    continue

                if tok.type == tokenize.NAME:
                    if (
                        tok.string not in self.python_keywords
                        and tok.string not in self.builtin_names
                    ):
                        operands.add(tok.string)
                elif tok.type in (tokenize.NUMBER, tokenize.STRING):
                    operands.add(tok.string)

        except tokenize.TokenError:
            pass

        return operands


def main():
    simulator = ProgramLengthSimulator()

    eta_values = [16, 32, 64, 128]

    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №1")
    print("Вероятностное моделирование метрических характеристик программ")
    print("=" * 80)

    for eta in eta_values:
        print(f"\n--- МОДЕЛИРОВАНИЕ ДЛЯ η = {eta} ---")
        lengths = simulator.simulate_program_writing(eta, 10000)

        exp = simulator.results[eta]["experimental"]
        theory = simulator.results[eta]["theoretical"]

        print("ЭКСПЕРИМЕНТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        print(f"Средняя длина программы L: {exp['mean_length']:.2f}")
        print(f"Дисперсия D(Lη): {exp['variance']:.2f}")
        print(f"СКО √D(Lη): {exp['std_dev']:.2f}")
        print(
            f"Относительная погрешность δ: {exp['relative_error']:.4f} ({exp['relative_error'] * 100:.2f}%)"
        )

        print("\nТЕОРЕТИЧЕСКИЕ ЗНАЧЕНИЯ:")
        print(f"Средняя длина программы L: {theory['mean_length']:.2f}")
        print(f"Дисперсия D(Lη): {theory['variance']:.2f}")
        print(f"СКО √D(Lη): {theory['std_dev']:.2f}")
        print(
            f"Относительная погрешность δ: {theory['relative_error']:.4f} ({theory['relative_error'] * 100:.2f}%)"
        )
        print(f"Альтернативная формула L: {theory['alternative_length']:.2f}")

        length_error = (
            abs(exp["mean_length"] - theory["mean_length"])
            / theory["mean_length"]
            * 100
        )
        print(f"\nОшибка предсказания длины: {length_error:.2f}%")

    print("\n" + "=" * 80)
    print("АНАЛИЗ ТЕКСТА ПРОГРАММЫ (ЭТОГО СКРИПТА)")
    print("=" * 80)

    with open(__file__, "r", encoding="utf-8") as f:
        program_text = f.read()

    analysis = simulator.analyze_program_text(program_text)

    print(f"Размер словаря программы η: {analysis['eta']}")
    print(f" из них операторов: {analysis['operators_count']}")
    print(f" из них операндов: {analysis['operands_count']}")
    print(f"Фактическая длина программы (токенов): {analysis['actual_length']}")
    print(f"Прогнозируемая длина (формула 1): {analysis['predicted_length_1']:.2f}")
    print(f"Прогнозируемая длина (формула 2): {analysis['predicted_length_2']:.2f}")
    print(f"Ошибка прогноза (формула 1): {analysis['error_1']:.2f}%")
    print(f"Ошибка прогноза (формула 2): {analysis['error_2']:.2f}%")

    print("\n" + "=" * 80)
    print("ОПРЕДЕЛЕНИЕ η₂* - ЧИСЛА УНИКАЛЬНЫХ ОПЕРАНДОВ")
    print("=" * 80)

    operands = simulator.analyze_eta2_star(program_text)
    eta2_star = len(operands)

    print(f"Число уникальных операндов η₂*: {eta2_star}")
    examples = list(operands)[:10]
    print(f"Примеры операндов: {examples}{'...' if len(operands) > 10 else ''}")

    if eta2_star > 1:
        predicted_by_eta2 = 0.9 * eta2_star * math.log2(eta2_star)
        print(f"\nПрогноз длины по η₂* (условно): {predicted_by_eta2:.2f}")
        print(f"Фактическая длина: {analysis['actual_length']}")
    else:
        print("\nНедостаточно операндов для прогноза по η₂*")


if __name__ == "__main__":
    main()
