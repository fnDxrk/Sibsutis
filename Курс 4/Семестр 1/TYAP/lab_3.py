from typing import Dict, Set, Tuple, List, Optional, Union
from collections import deque
import os

class DMPProcessor:
    def __init__(self, states: Set[str], input_alphabet: Set[str], stack_alphabet: Set[str],
                 transitions: Dict[Tuple[str, str, str], List[Tuple[str, str]]],
                 start_state: str, start_stack: str, final_states: Set[str]):
        self.states = states
        self.input_alphabet = input_alphabet
        self.stack_alphabet = stack_alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.start_stack = start_stack
        self.final_states = final_states

    def process_string(self, input_string: str):
        print(f"\n{'='*60}")
        print(f"ПОШАГОВОЕ ВЫПОЛНЕНИЕ ДЛЯ ЦЕПОЧКИ: '{input_string}'")
        print(f"{'='*60}")

        initial_config = (self.start_state, 0, [self.start_stack])
        queue = deque([(initial_config, [])])
        visited = set()
        step_counter = 0

        print(f"Шаг {step_counter}: Начальная конфигурация")
        print(f"  Состояние: {self.start_state}")
        print(f"  Позиция в строке: 0")
        print(f"  Стек: [{self.start_stack}]")
        print(f"  Оставшаяся строка: '{input_string}'")

        while queue:
            (state, pos, stack), path = queue.popleft()

            config_key = (state, pos, tuple(stack))
            if config_key in visited:
                continue
            visited.add(config_key)

            if pos == len(input_string):
                step_counter += 1
                print(f"\nШаг {step_counter}: Достигнут конец строки")
                print(f"  Состояние: {state}")
                print(f"  Стек: {stack if stack else '[пустой]'}")

                epsilon_transitions = self.get_epsilon_transitions(state, stack)
                for next_state, new_stack in epsilon_transitions:
                    step_counter += 1
                    print(f"\nШаг {step_counter}: ε-переход")
                    print(f"  δ({state}, ε, {stack[0] if stack else 'пустой'}) -> ({next_state}, {new_stack if new_stack else '[пустой]'})")
                    print(f"  Новое состояние: {next_state}")
                    print(f"  Новый стек: {new_stack if new_stack else '[пустой]'}")

                    if next_state in self.final_states and not new_stack:
                        print(f"\n{'='*60}")
                        print("РЕЗУЛЬТАТ: ЦЕПОЧКА ПРИНЯТА!")
                        print("Достигнуто финальное состояние с пустым стеком")
                        print(f"{'='*60}")
                        return True, "Цепочка принята", []
                    queue.append(((next_state, pos, new_stack), path + [f"ε-переход в {next_state}"]))

                if state in self.final_states and not stack:
                    print(f"\n{'='*60}")
                    print("РЕЗУЛЬТАТ: ЦЕПОЧКА ПРИНЯТА!")
                    print("Достигнуто финальное состояние с пустым стеком")
                    print(f"{'='*60}")
                    return True, "Цепочка принята", []
                continue

            current_symbol = input_string[pos]
            remaining_string = input_string[pos+1:] if pos+1 < len(input_string) else ""

            step_counter += 1
            print(f"\nШаг {step_counter}: Обработка символа '{current_symbol}'")
            print(f"  Текущее состояние: {state}")
            print(f"  Текущий стек: {stack if stack else '[пустой]'}")
            print(f"  Оставшаяся строка: '{remaining_string}'")

            possible_transitions = self.get_transitions(state, current_symbol, stack)

            if not possible_transitions:
                print(f"  Нет доступных переходов для символа '{current_symbol}'")
                continue

            for next_state, new_stack in possible_transitions:
                print(f"  Применяем переход: δ({state}, '{current_symbol}', {stack[0] if stack else 'пустой'}) -> ({next_state}, {new_stack if new_stack else 'ε'})")
                print(f"  Новое состояние: {next_state}")
                print(f"  Новый стек: {new_stack if new_stack else '[пустой]'}")
                queue.append(((next_state, pos + 1, new_stack), path + [f"Переход по '{current_symbol}'"]))

            epsilon_transitions = self.get_epsilon_transitions(state, stack)
            # for next_state, new_stack in epsilon_transitions:
                # print(f"  Доступен ε-переход: δ({state}, ε, {stack[0] if stack else 'пустой'}) -> ({next_state}, {new_stack if new_stack else 'ε'})")
                # queue.append(((next_state, pos, new_stack), path + [f"ε-переход в {next_state}"]))

        print(f"\n{'='*60}")
        print("РЕЗУЛЬТАТ: ЦЕПОЧКА ОТКЛОНЕНА!")
        print("Нет пути к финальному состоянию с пустым стеком")
        print(f"{'='*60}")
        return False, "Цепочка отклонена", []
    
    def get_transitions(self, state: str, symbol: str, stack: List[str]):
        if not stack:
            return []
        
        top = stack[0]
        key = (state, symbol, top)
        
        if key not in self.transitions:
            return []
        
        results = []
        for next_state, stack_operation in self.transitions[key]:
            new_stack = stack[1:] 
            
            if stack_operation == 'ε':
                pass 
            else:
                for char in reversed(stack_operation):
                    new_stack.insert(0, char)
            
            results.append((next_state, new_stack))
        
        return results

    def get_epsilon_transitions(self, state: str, stack: List[str]):
        if not stack:
            return []
        
        top = stack[0]
        key = (state, 'ε', top)
        
        if key not in self.transitions:
            return []
        
        results = []
        for next_state, stack_operation in self.transitions[key]:
            new_stack = stack[1:] 
            
            if stack_operation == 'ε':
                pass 
            else:
                for char in reversed(stack_operation):
                    new_stack.insert(0, char)
            
            results.append((next_state, new_stack))
        
        return results
    
    def print_automaton_info(self):
        print("\n" + "╔" + "═" * 70 + "╗")
        print("║" + " " * 27 + "ДМП-АВТОМАТ" + " " * 32 + "║")
        print("╠" + "═" * 70 + "╣")
        print(f"║ Состояния:           {str(sorted(self.states)):<47} ║")
        print(f"║ Входной алфавит:     {str(sorted(self.input_alphabet)):<47} ║")
        print(f"║ Стековый алфавит:    {str(sorted(self.stack_alphabet)):<47} ║")
        print(f"║ Начальное состояние: {self.start_state:<47} ║")
        print(f"║ Начальный символ:    {self.start_stack:<47} ║")
        print(f"║ Финальные состояния: {str(sorted(self.final_states)):<47} ║")
        print("╠" + "═" * 70 + "╣")
        print("║" + " " * 29 + "ПЕРЕХОДЫ" + " " * 33 + "║")
        print("╠" + "═" * 70 + "╣")
        
        for (state, symbol, stack_top), transitions in sorted(self.transitions.items()):
            for next_state, stack_op in transitions:
                transition_str = f"δ({state}, {symbol}, {stack_top}) = ({next_state}, {stack_op})"
                print(f"║ {transition_str:<68} ║")
        
        print("╚" + "═" * 70 + "╝")


def create_task_dmp() -> DMPProcessor:
    states = {'q0', 'q1', 'q2'}
    input_alphabet = {'a', 'b'}
    stack_alphabet = {'Z', 'a', 'b'}
    
    transitions = {
        ('q0', 'a', 'Z'): [('q0', 'aZ')],
        ('q0', 'b', 'Z'): [('q0', 'bZ')],
        ('q0', 'a', 'a'): [('q0', 'aa'), ('q1', 'ε')],
        ('q0', 'a', 'b'): [('q0', 'ab')],
        ('q0', 'b', 'a'): [('q0', 'ba')],
        ('q0', 'b', 'b'): [('q0', 'bb'), ('q1', 'ε')],
        ('q1', 'a', 'a'): [('q1', 'ε')],
        ('q1', 'b', 'b'): [('q1', 'ε')],
        ('q1', 'ε', 'Z'): [('q2', 'ε')]
    }
    
    start_state = 'q0'
    start_stack = 'Z'
    final_states = {'q2'}
    
    return DMPProcessor(states, input_alphabet, stack_alphabet, transitions, 
                       start_state, start_stack, final_states)

def load_dmp_from_file(filename: str) -> Union[DMPProcessor, None]:
    states = set()
    input_alphabet = set()
    stack_alphabet = set()
    start_state = ""
    start_stack = ""
    final_states = set()
    transitions = {}
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if line.startswith('STATES:'):
                    states = set(line.split()[1:])
                elif line.startswith('INPUT_ALPHABET:'):
                    input_alphabet = set(line.split()[1:])
                elif line.startswith('STACK_ALPHABET:'):
                    stack_alphabet = set(line.split()[1:])
                elif line.startswith('START:'):
                    start_state = line.split()[1]
                elif line.startswith('START_STACK:'):
                    start_stack = line.split()[1]
                elif line.startswith('FINAL:'):
                    final_states = set(line.split()[1:])
                elif line.startswith('TRANSITIONS:'):
                    continue
                else:
                    parts = line.split()
                    if len(parts) == 5:
                        from_state, input_symbol, stack_symbol, to_state, new_stack = parts
                        key = (from_state, input_symbol, stack_symbol)
                        if key not in transitions:
                            transitions[key] = []
                        transitions[key].append((to_state, new_stack))
    
    except FileNotFoundError:
        print(f"Файл '{filename}' не найден!")
        return None
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None
    
    return DMPProcessor(states, input_alphabet, stack_alphabet, transitions, 
                       start_state, start_stack, final_states)


def analyze_language():
    print("\nВЫВОД: Автомат принимает язык L = {w | w = xy, где y = reverse(x)}")
    print("Строки вида: префикс + обращенный_префикс")


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "ДМП-АВТОМАТ АНАЛИЗАТОР" + " " * 21 + "║")
    print("║" + " " * 10 + "Проверка принадлежности цепочек языку" + " " * 11 + "║")
    print("╚" + "═" * 58 + "╝")

def print_menu():
    print("\n┌─────────────────────────────────────────┐")
    print("│             ВЫБОР АВТОМАТА              │")
    print("├─────────────────────────────────────────┤")
    print("│ 1. Загрузить ДМП из файла               │")
    print("│ 2. ДМП из задания 19                    │")
    print("└─────────────────────────────────────────┘")

def main():
    clear_screen()
    print_header()
    print_menu()
    
    choice = input("\nВаш выбор (1): ").strip()
    
    automaton = None
    
    if choice == '1':
        filename = input("Введите имя файла ДМП: ").strip()
        automaton = load_dmp_from_file(filename)
        if automaton is None:
            print("Используем ДМП из задания 19...")
            automaton = create_task_dmp()
    
    elif choice == '2':
        automaton = create_task_dmp()
        print("Загружен ДМП из задания 19")
        analyze_language()
        
        test_strings = ['abab', 'aabaa', 'abaab', 'aabaab', 'aabbaa', 'baab', 'baba']
        
        print("\n" + "╔" + "═" * 70 + "╗")
        print("║" + " " * 23 + "ПРОВЕРКА ЦЕПОЧЕК" + " " * 23 + "║")
        print("╚" + "═" * 70 + "╝")
        
        for test_string in test_strings:
            print(f"\n{'='*70}")
            print(f"ТЕСТИРОВАНИЕ ЦЕПОЧКИ: '{test_string}'")
            print('='*70)
            
            accepted, reason, path = automaton.process_string(test_string)
        
        print(f"\n{'='*70}")
        print("ИНТЕРАКТИВНАЯ ПРОВЕРКА")
        print("Введите цепочки для проверки (Enter для завершения):")
        
        while True:
            input_string = input("\nЦепочка: ").strip()
            if not input_string:
                break
            
            accepted, reason, path = automaton.process_string(input_string)
        
        print("\nПрограмма завершена.")
        return
    
    else:
        print("Неверный выбор. Используем ДМП из задания 19...")
        automaton = create_task_dmp()
    
    automaton.print_automaton_info()
    
    print("\n" + "┌" + "─" * 58 + "┐")
    print("│" + " " * 19 + "ПРОВЕРКА ЦЕПОЧЕК" + " " * 23 + "│")
    print("└" + "─" * 58 + "┘")
    print("Введите цепочки для проверки (Enter для завершения)")
    
    while True:
        print("\n" + "─" * 40)
        input_string = input("Цепочка: ").strip()
        
        if not input_string:
            break
        
        accepted, reason, path = automaton.process_string(input_string)
        
        print(f"\nАнализ цепочки: '{input_string}'")
        
        if accepted:
            print(f"ПРИНЯТА: {reason}")
        else:
            print(f"ОТКЛОНЕНА: {reason}")
    
    print("\nПрограмма завершена.")


if __name__ == "__main__":
    main()
