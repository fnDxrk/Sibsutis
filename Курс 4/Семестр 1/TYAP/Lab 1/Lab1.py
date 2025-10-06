import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List


class TokenType(Enum):
    NUMBER = 'number'
    ID = 'id'
    PLUS = '+'
    MINUS = '-'
    MULTIPLY = '*'
    DIVIDE = '/'
    LPAREN = '('
    RPAREN = ')'
    EOF = 'EOF'


@dataclass
class Token:
    type: TokenType
    value: str
    position: int


fork_string = '├──'
corner_string = '└──'
wall_string = '│  '
space_string = '   '


class ASTNode:
    """Класс для узлов синтаксического дерева"""

    def __init__(self, node_type: str, value: str = "", children: List['ASTNode'] = None):
        self.node_type = node_type
        self.value = value
        self.children = children if children is not None else []

    def add_child(self, child: 'ASTNode'):
        self.children.append(child)

    def draw_tree(self, prefix: str = "", is_last: bool = True) -> str:
        """Рекурсивное рисование дерева с вертикальными линиями"""
        # Текущий узел
        if self.value:
            current_line = f"{self.node_type}: {self.value}"
        else:
            current_line = self.node_type

        if prefix == "":  # Корневой узел
            result = current_line + "\n"
        else:
            connector = corner_string if is_last else fork_string
            result = prefix + connector + current_line + "\n"

        # Новый префикс для дочерних
        if is_last:
            new_prefix = prefix + space_string
        else:
            new_prefix = prefix + wall_string

        # Дети
        for i, child in enumerate(self.children):
            is_last_child = i == len(self.children) - 1
            result += child.draw_tree(new_prefix, is_last_child)

        return result

    def to_bracket_notation(self) -> str:
        """Скобочная нотация"""
        if not self.children:
            return f"{self.node_type}:{self.value}" if self.value else self.node_type

        children_str = " ".join(child.to_bracket_notation() for child in self.children)
        return f"({self.node_type}{':' + self.value if self.value else ''} {children_str})"

    def display_tree(self):
        """Выводит дерево в консоль"""
        print("🌳 Синтаксическое дерево:")
        print("─" * 50)
        tree_str = self.draw_tree()
        print(tree_str)

    def __str__(self) -> str:
        if self.value:
            return f"{self.node_type}: {self.value}"
        return self.node_type


class ParserError(Exception):
    def __init__(self, message, position):
        super().__init__(message)
        self.message = message
        self.position = position


class ArithmeticParser:
    def __init__(self, expression: str):
        self.expression = expression
        self.tokens = []
        self.current_token_index = 0
        self.current_token = None

    def tokenize(self):
        """Лексический анализатор - разбивает строку на токены"""
        token_specification = [
            (TokenType.NUMBER, r'\d+'),
            (TokenType.ID, r'[a-zA-Z_][a-zA-Z0-9_]*'),
            (TokenType.PLUS, r'\+'),
            (TokenType.MINUS, r'-'),
            (TokenType.MULTIPLY, r'\*'),
            (TokenType.DIVIDE, r'/'),
            (TokenType.LPAREN, r'\('),
            (TokenType.RPAREN, r'\)'),
            (TokenType.EOF, r'$')
        ]

        tokens = []
        position = 0
        expression = self.expression.strip() #удаляем пробелы

        while position < len(expression):
            if expression[position].isspace(): #проверка на пробелы в начале
                position += 1
                continue

            match = None
            for token_type, pattern in token_specification:
                regex = re.compile(pattern)
                match = regex.match(expression, position) #ищем совпадение
                if match:
                    value = match.group(0)
                    if token_type != TokenType.EOF:
                        tokens.append(Token(token_type, value, position))
                    position = match.end()
                    break

            if not match:
                raise ParserError(f"Неизвестный символ: '{expression[position]}'", position)

        tokens.append(Token(TokenType.EOF, '', len(expression)))
        return tokens

    def get_next_token(self) -> Token:
        """Возвращает следующий токен"""
        if self.current_token_index < len(self.tokens):
            token = self.tokens[self.current_token_index]
            self.current_token_index += 1
            return token
        return Token(TokenType.EOF, '', len(self.expression))

    def match(self, expected_types) -> Token:
        """Проверяет соответствие текущего токена с ожидаемым типом"""
        if not isinstance(expected_types, list):
            expected_types = [expected_types]

        if self.current_token.type in expected_types:
            token = self.current_token
            self.current_token = self.get_next_token()
            return token

        expected_str = ", ".join([f"'{t.value}'" for t in expected_types])
        raise ParserError(
            f"Ошибка! Ожидалось: {expected_str}, но получено: '{self.current_token.value}'",
            self.current_token.position
        )

    def parse(self) -> ASTNode:
        """Основной метод парсинга
           Вызываем parse_S - начало грамматики
        """
        try:
            self.tokens = self.tokenize()
            self.current_token_index = 0
            self.current_token = self.get_next_token()

            tree = self.parse_S()
            self.match(TokenType.EOF)
            return tree

        except ParserError as e:
            error_message = f"{e.message}\n"
            error_message += f"Позиция: {e.position}\n"
            error_message += f"Строка: {self.expression}\n"
            error_message += " " * (e.position + 8) + "^\n"
            raise ParserError(error_message, e.position)

    def parse_S(self) -> ASTNode:
        """S -> T E"""
        node = ASTNode("   S")
        node.add_child(self.parse_T())
        node.add_child(self.parse_E())
        return node

    def parse_E(self) -> ASTNode:
        """E -> + T E | - T E | ε"""
        node = ASTNode("E")

        if self.current_token.type in [TokenType.PLUS, TokenType.MINUS]:
            operator_token = self.match([TokenType.PLUS, TokenType.MINUS])
            node.add_child(ASTNode("OPERATOR", operator_token.value))
            node.add_child(self.parse_T())
            node.add_child(self.parse_E())
        else:
            node.add_child(ASTNode("EPSILON", "ε"))

        return node

    def parse_T(self) -> ASTNode:
        """T -> F T'"""
        node = ASTNode("T")
        node.add_child(self.parse_F())
        node.add_child(self.parse_T_prime())
        return node

    def parse_T_prime(self) -> ASTNode:
        """T' -> * F T' | / F T' | ε"""
        node = ASTNode("T'")

        if self.current_token.type in [TokenType.MULTIPLY, TokenType.DIVIDE]:
            operator_token = self.match([TokenType.MULTIPLY, TokenType.DIVIDE])
            node.add_child(ASTNode("OPERATOR", operator_token.value))
            node.add_child(self.parse_F())
            node.add_child(self.parse_T_prime())
        else:
            node.add_child(ASTNode("EPSILON", "ε"))

        return node

    def parse_F(self) -> ASTNode:
        """F -> ( S ) | number | id"""
        node = ASTNode("F")

        if self.current_token.type == TokenType.LPAREN:
            self.match(TokenType.LPAREN)
            node.add_child(self.parse_S())
            self.match(TokenType.RPAREN)
        elif self.current_token.type == TokenType.NUMBER:
            number_token = self.match(TokenType.NUMBER)
            node.add_child(ASTNode("NUMBER", number_token.value))
        elif self.current_token.type == TokenType.ID:
            id_token = self.match(TokenType.ID)
            node.add_child(ASTNode("IDENTIFIER", id_token.value))
        else:
            expected = [TokenType.LPAREN, TokenType.NUMBER, TokenType.ID]
            expected_str = ", ".join([f"'{t.value}'" for t in expected])
            raise ParserError(f"Ошибка! Ожидалось: {expected_str}", self.current_token.position)

        return node


def main():
    print("Синтаксический анализатор арифметических выражений")
    print("Грамматика:")
    print("S -> T E")
    print("E -> + T E | - T E | ε")
    print("T -> F T'")
    print("T' -> * F T' | / F T' | ε")
    print("F -> ( S ) | number | id")
    print("=" * 60)

    while True:
        try:
            expression = input("\nВведите выражение (или 'quit' для выхода): ").strip()

            if expression.lower() == 'quit':
                break

            if not expression:
                continue

            parser = ArithmeticParser(expression)
            syntax_tree = parser.parse()

            print("✅ Выражение корректно.")

            # Отображаем дерево в консоли
            syntax_tree.display_tree()

            print("\nСкобочная нотация:")
            print("─" * 30)
            print(syntax_tree.to_bracket_notation())

        except ParserError as e:
            print(f"❌ {e.message}")
        except KeyboardInterrupt:
            print("\nПрограмма завершена.")
            break
        except Exception as e:
            print(f"Неожиданная ошибка: {e}")


def run_demo():
    """Демонстрация работы с разными выражениями"""
    demo_expressions = [
        "2 + 3 * 4",
        "a * (b - 10)",
        "(x + y) * 2 - z / 3",
        "42",
        "simple_variable",
        "a + b * c - d / (e + f)"
    ]

    print("Демонстрация работы парсера:")
    print("=" * 50)

    for expr in demo_expressions:
        print(f"\nВыражение: {expr}")
        print("─" * (len(expr) + 12))

        try:
            parser = ArithmeticParser(expr)
            tree = parser.parse()
            print("✅ Корректно")
            tree.display_tree()
            print()

        except ParserError as e:
            print(f"❌ Ошибка: {e.message.splitlines()[0]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo()
    else:
        main()
