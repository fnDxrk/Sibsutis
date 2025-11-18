import re
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
from enum import Enum
from typing import List
import numpy as np


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


class ASTNode:
    """Класс для узлов абстрактного синтаксического дерева"""

    def __init__(self, node_type: str, value: str = "", children: List['ASTNode'] = None):
        self.node_type = node_type
        self.value = value
        self.children = children if children is not None else []
        self.x = 0  # Позиция X для визуализации
        self.y = 0  # Позиция Y для визуализации
        self.width = 0  # Ширина поддерева

    def add_child(self, child: 'ASTNode'):
        self.children.append(child)

    def calculate_layout(self, x=0, y=0, level=0):
        """Вычисляет позиции всех узлов для визуализации"""
        # Уменьшаем масштаб
        vertical_spacing = 1.2
        horizontal_spacing = 1.0

        self.y = y

        if not self.children:
            # Листовой узел
            self.x = x
            self.width = 0.6  # Минимальная ширина для листа
            return x + 0.8, y  # Возвращаем следующую позицию

        # Сначала вычисляем layout для всех детей
        child_x = x
        max_child_y = y - vertical_spacing

        child_widths = []
        for child in self.children:
            child_x, child_y = child.calculate_layout(child_x, y - vertical_spacing, level + 1)
            child_widths.append(child.width)
            # Добавляем расстояние между детьми
            child_x += horizontal_spacing * 0.3

        # Вычисляем общую ширину поддерева
        total_children_width = sum(child_widths) + horizontal_spacing * 0.3 * (len(self.children) - 1)

        # Центрируем текущий узел над детьми
        if self.children:
            first_child = self.children[0]
            last_child = self.children[-1]
            self.x = (first_child.x + last_child.x) / 2
            self.width = max(0.8, total_children_width)  # Минимальная ширина

        return x + total_children_width, y

    def get_tree_bounds(self):
        """Возвращает границы дерева (min_x, max_x, min_y, max_y)"""
        if not self.children:
            return self.x - 0.3, self.x + 0.3, self.y - 0.2, self.y + 0.2

        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')

        # Границы текущего узла
        min_x = min(min_x, self.x - 0.4)
        max_x = max(max_x, self.x + 0.4)
        min_y = min(min_y, self.y - 0.2)
        max_y = max(max_y, self.y + 0.2)

        # Границы детей
        for child in self.children:
            c_min_x, c_max_x, c_min_y, c_max_y = child.get_tree_bounds()
            min_x = min(min_x, c_min_x)
            max_x = max(max_x, c_max_x)
            min_y = min(min_y, c_min_y)
            max_y = max(max_y, c_max_y)

        return min_x, max_x, min_y, max_y

    def plot_tree(self, ax, level=0):
        """Рекурсивно рисуем дерево"""

        # Цвета
        colors = {
            'S': '#FF6B6B',  # Красный
            'E': '#4ECDC4',  # Бирюзовый
            'T': '#45B7D1',  # Синий
            "T'": '#96CEB4',  # Зеленый
            'F': '#FECA57',  # Желтый
            'UNARY_MINUS': '#FA003F',
            'OPERATOR': '#FF9FF3',  # Розовый
            'NUMBER': '#54A0FF',  # Голубой
            'IDENTIFIER': '#5F27CD',  # Фиолетовый
            'EPSILON': '#C8D6E5'  # Серый
        }

        node_color = colors.get(self.node_type, '#E0E0E0')
        text_color = 'white' if self.node_type in ['S', 'E', 'T', "T'", 'F'] else 'black'

        # Текст узла
        display_value = self.value
        if display_value and len(display_value) > 8:
            display_value = display_value[:8] + "..."

        node_text = f"{self.node_type}"
        if display_value:
            node_text += f"\n{display_value}"

        # Рисуем узел - уменьшаем размер
        rect_width = 0.7
        rect_height = 0.35

        rect = patches.Rectangle(
            (self.x - rect_width / 2, self.y - rect_height / 2), rect_width, rect_height,
            linewidth=1.5, edgecolor='black', facecolor=node_color, alpha=0.9,
            zorder=3  # Узлы поверх линий
        )
        ax.add_patch(rect)

        # Текст узла
        font_size = 8 if display_value else 9
        ax.text(self.x, self.y, node_text, ha='center', va='center',
                fontsize=font_size, fontweight='bold', color=text_color, zorder=4)

        # Рисуем связи с детьми
        for i, child in enumerate(self.children):
            # Линия к ребенку
            ax.plot([self.x, child.x], [self.y - rect_height / 2, child.y + rect_height / 2],
                    'black', linewidth=1.5, alpha=0.8, zorder=2)

            # Рекурсивно рисуем детей
            child.plot_tree(ax, level + 1)

    def display_matplotlib_tree(self, expression):
        """Отображает дерево"""
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(12, 8))

        # Вычисляем layout
        self.calculate_layout()

        # Получаем границы дерева
        min_x, max_x, min_y, max_y = self.get_tree_bounds()

        # Добавляем отступы
        x_padding = 1.0
        y_padding = 1.0

        # Настройки графика с уменьшенным масштабом
        ax.set_xlim(min_x - x_padding, max_x + x_padding)
        ax.set_ylim(min_y - y_padding, max_y + y_padding)
        ax.set_aspect('equal')
        ax.axis('off')  # Скрываем оси

        # Рисуем дерево
        self.plot_tree(ax)

        # Заголовок
        plt.title(f'Синтаксическое дерево для: "{expression}"',
                  fontsize=14, fontweight='bold', pad=20)

        # Легенда
        legend_elements = [
            patches.Patch(color='#FF6B6B', label='S - Выражение'),
            patches.Patch(color='#4ECDC4', label='E - Операции +/-'),
            patches.Patch(color='#45B7D1', label='T - Терм'),
            patches.Patch(color='#96CEB4', label="T' - Операции */"),
            patches.Patch(color='#FECA57', label='F - Фактор'),
            patches.Patch(color='#FF9FF3', label='OPERATOR - Оператор'),
            patches.Patch(color='#54A0FF', label='NUMBER - Число'),
            patches.Patch(color='#5F27CD', label='IDENTIFIER - Идентификатор'),
            patches.Patch(color='#C8D6E5', label='EPSILON - Пусто')
        ]

        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1),
                  fontsize=8, framealpha=0.9)

        # Улучшаем layout
        plt.tight_layout()

        # ax.grid(True, alpha=0.3)

        plt.show()

    def to_bracket_notation(self) -> str:
        """Скобочная нотация"""
        if not self.children:
            return f"{self.node_type}:{self.value}" if self.value else self.node_type

        children_str = " ".join(child.to_bracket_notation() for child in self.children)
        return f"({self.node_type}{':' + self.value if self.value else ''} {children_str})"

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

    def tokenize(self) -> List[Token]:
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
        expression = self.expression.strip()

        while position < len(expression):
            if expression[position].isspace():
                position += 1
                continue

            match = None
            for token_type, pattern in token_specification:
                regex = re.compile(pattern)
                match = regex.match(expression, position)
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
        """Проверяет соответствие текущего токена ожидаемому типу"""
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
        """Основной парсинг"""
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
        node = ASTNode("S")
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

        if self.current_token.type == TokenType.MINUS:
            operator_token = self.match(TokenType.MINUS)
            minus_node = ASTNode("UNARY_MINUS", operator_token.value)
            minus_node.add_child(self.parse_F())
            node.add_child(minus_node)
        elif self.current_token.type == TokenType.LPAREN:
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

            print("\nСкобочная нотация:")
            print("─" * 30)
            print(syntax_tree.to_bracket_notation())

            # Отображаем графическое дерево
            print("Строим графическое дерево...")
            syntax_tree.display_matplotlib_tree(expression)

        except ParserError as e:
            print(f"❌ {e.message}")
        except KeyboardInterrupt:
            print("\nПрограмма завершена.")
            break
        except Exception as e:
            print(f"Неожиданная ошибка: {e}")


if __name__ == "__main__":
    main()
