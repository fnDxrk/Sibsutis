#ifndef SIMPLEX_HPP
#define SIMPLEX_HPP

#include "Fraction.hpp"
#include <vector>
#include <string>
#include <utility>

class Simplex {
private:
    std::vector<Fraction> original_objective_coeffs_;               // Исходные коэффициенты целевой функции
    std::vector<Fraction> objective_coeffs_;                        // Коэффициенты целевой функции после преобразования в каноническую форму
    std::vector<std::vector<Fraction>> constraint_coeffs_;          // Коэффициенты системы ограничений
    std::vector<std::string> constraint_signs_;                     // Знаки ограничений ("<=", ">=", "=")
    std::vector<Fraction> constraint_rhs_;                          // Правая часть ограничений
    std::string optimization_goal_;                                 // Цель оптимизации: "min" или "max"
    size_t num_original_vars_;                                      // Количество переменных в изначальной постановке задачи
    std::vector<std::vector<Fraction>> simplex_tableau_;            // Симплекс-таблица
    std::vector<size_t> basis_indices_;                             // Индексы базисных переменных
    std::vector<size_t> artificial_var_indices_;                    // Индексы искусственных переменных
    bool has_big_m_row_;                                            // Флаг наличия строки большого M (метод искусственного базиса)
    size_t objective_row_index_;                                    // Индекс строки целевой функции (Z-строки)
    size_t big_m_row_index_;                                        // Индекс строки большого M
    size_t current_iteration_;                                      // Номер текущей итерации
    Fraction big_m_value_;                                          // Значение большого M

    void transform_to_canonical_form();                             // Преобразует задачу в каноническую форму (максимизация, равенства)
    void add_artificial_variables();                                // Добавляет искусственные переменные для начального базиса
    void create_initial_simplex_tableau();                          // Создаёт начальную симплекс-таблицу
    bool is_solution_optimal() const;                               // Проверяет, является ли решение оптимальным
    std::pair<size_t, size_t> find_pivot_element() const;           // Находит ведущий элемент (строка, столбец)
    void execute_simplex_pivot(size_t row, size_t col);             // Выполняет симплекс-поворот для выбранного элемента
    void execute_phase_two_pivot(size_t row, size_t col);           // Выполняет симплекс-поворот для второй фазы
    void update_big_m_row();                                        // Обновляет строку с большим M после итерации
    void restore_objective_row();                                   // Восстанавливает строку целевой функции для второй фазы
    void remove_artificial_variables();                             // Удаляет искусственные переменные из таблицы
    bool is_infeasible_due_to_big_m_row() const;                    // Проверяет неосуществимость задачи из-за строки M
    void print_equation_term(const Fraction& coef, size_t index, bool& first_term) const;  // Выводит член уравнения
    void print_constraint_equation(const std::vector<Fraction>& coefs, const std::string& sign, const Fraction& rhs) const;  // Выводит уравнение ограничения
    void print_original_problem_form() const;                       // Выводит исходную форму задачи
    void print_canonical_problem_form() const;                      // Выводит каноническую форму задачи
    void print_canonical_form_with_artificial_variables() const;    // Выводит каноническую форму с искусственными переменными
    void print_simplex_tableau(const std::pair<size_t, size_t>* pivot = nullptr) const;   // Выводит текущую симплекс-таблицу с подсветкой ведущего элемента
    void print_current_solution() const;                            // Выводит текущее решение задачи
    void find_alternative_optimal_solutions();                      // Ищет альтернативные оптимальные решения
    std::vector<Fraction> extract_current_solution() const;         // Извлекает текущие значения переменных решения
    std::string format_solution_as_string(const std::vector<Fraction>& solution) const;  // Форматирует решение в строку ("(1, 0, 2)")

public:
    Simplex(const std::vector<Fraction>& objective_coeffs,
            const std::vector<std::vector<Fraction>>& constraint_coeffs,
            const std::vector<std::string>& constraint_signs,
            const std::vector<Fraction>& constraint_rhs,
            const std::string& optimization_goal);

    bool solve_linear_program();
};

#endif // SIMPLEX_HPP
