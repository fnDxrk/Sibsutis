#ifndef SIMPLEX_HPP
#define SIMPLEX_HPP

#include "Fraction.hpp"
#include <vector>
#include <string>

class SimplexBigM {
private:
    std::vector<Fraction> obj_func_; // Коэффициенты целевой функции
    std::vector<std::vector<Fraction>> constraints_; // Коэффициенты ограничений
    std::vector<std::string> signs_; // Знаки ограничений (<=, >=, =)
    std::vector<Fraction> rhs_; // Правая часть ограничений
    std::string goal_; // Цель (min или max)
    size_t original_vars_; // Количество исходных переменных
    std::vector<std::vector<Fraction>> tableau_; // Симплекс-таблица
    std::vector<size_t> basis_; // Базисные переменные
    std::vector<size_t> artificial_vars_; // Индексы искусственных переменных
    bool has_m_row_; // Наличие M-строки
    size_t z_row_index_; // Индекс Z-строки
    size_t m_row_index_; // Индекс M-строки
    int iteration_; // Текущая итерация
    Fraction M_; // Большое число M

    void to_canonical();
    void add_artificial_vars();
    void build_initial_tableau();
    bool is_optimal() const;
    std::pair<size_t, size_t> get_pivot() const;
    void pivot(size_t row, size_t col);
    void pivot2(size_t row, size_t col);
    void update_m_row();
    void restore_original_z_row();
    void remove_artificial_vars();
    bool is_infeasible_due_to_m_row() const;
    void print_term(const Fraction& coef, size_t index, bool& first_term) const;
    void print_equation(const std::vector<Fraction>& coefs, const std::string& sign_or_eq, const Fraction& rhs) const;
    void print_default_form() const;
    void print_canonical_form() const;
    void print_canonical_with_artificial() const;
    void print_tableau(const std::pair<size_t, size_t>* pivot = nullptr) const;
    void print_solution() const;
    std::vector<Fraction> get_current_solution() const;
    std::string format_solution(const std::vector<Fraction>& solution) const;

public:
    SimplexBigM(const std::vector<Fraction>& obj_func,
                const std::vector<std::vector<Fraction>>& constraints,
                const std::vector<std::string>& signs,
                const std::vector<Fraction>& rhs,
                const std::string& goal);

    void solve();
};

#endif // SIMPLEX_HPP