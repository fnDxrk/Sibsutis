#include "Simplex.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <sstream>

SimplexBigM::SimplexBigM(const std::vector<Fraction>& obj_func,
                         const std::vector<std::vector<Fraction>>& constraints,
                         const std::vector<std::string>& signs,
                         const std::vector<Fraction>& rhs,
                         const std::string& goal)
    : original_obj_func_(obj_func),
      obj_func_(obj_func),
      constraints_(constraints),
      signs_(signs),
      rhs_(rhs),
      goal_(goal),
      original_vars_(obj_func.size()),
      has_m_row_(true),
      z_row_index_(0),
      m_row_index_(0),
      iteration_(1),
      M_(1000) {
    print_default_form();
    to_canonical();
    print_canonical_form();
    add_artificial_vars();
    print_canonical_with_artificial();
    build_initial_tableau();
}

void SimplexBigM::to_canonical() {
    std::vector<Fraction> new_obj_func = obj_func_;
    if (goal_ == "min") {
        for (auto& coef : new_obj_func) {
            coef = -coef;
        }
    }

    std::vector<std::vector<Fraction>> new_constraints;
    size_t slack_vars = 0;

    for (size_t i = 0; i < constraints_.size(); ++i) {
        std::vector<Fraction> row = constraints_[i];
        Fraction b = rhs_[i];
        std::string sign = signs_[i];

        if (b < Fraction(0)) {
            for (auto& coef : row) {
                coef = -coef;
            }
            b = -b;
            if (sign == "<=") sign = ">=";
            else if (sign == ">=") sign = "<=";
        }

        row.resize(row.size() + slack_vars, Fraction(0));

        if (sign == "<=") {
            row.push_back(Fraction(1));
            new_obj_func.push_back(Fraction(0));
            ++slack_vars;
        } else if (sign == ">=") {
            row.push_back(Fraction(-1));
            new_obj_func.push_back(Fraction(0));
            ++slack_vars;
        }

        new_constraints.push_back(row);
        rhs_[i] = b;
    }

    size_t max_len = new_obj_func.size();
    for (const auto& row : new_constraints) {
        max_len = std::max(max_len, row.size());
    }
    for (auto& row : new_constraints) {
        row.resize(max_len, Fraction(0));
    }
    new_obj_func.resize(max_len, Fraction(0));

    obj_func_ = new_obj_func;
    constraints_ = new_constraints;
    signs_ = std::vector<std::string>(constraints_.size(), "=");
}

void SimplexBigM::add_artificial_vars() {
    size_t num_vars = constraints_[0].size();
    size_t num_constraints = constraints_.size();
    basis_.resize(num_constraints, 0);

    for (size_t i = 0; i < num_constraints; ++i) {
        bool has_basis = false;
        size_t basis_col = 0;
        for (size_t j = 0; j < num_vars; ++j) {
            bool is_unit = constraints_[i][j] == Fraction(1);
            bool is_unique = true;
            for (size_t k = 0; k < num_constraints; ++k) {
                if (k != i && constraints_[k][j] != Fraction(0)) {
                    is_unique = false;
                    break;
                }
            }
            if (is_unit && is_unique) {
                has_basis = true;
                basis_col = j;
                break;
            }
        }
        if (!has_basis) {
            for (auto& row : constraints_) {
                row.push_back(Fraction(0));
            }
            constraints_[i].back() = Fraction(1);
            obj_func_.push_back(Fraction(0));
            artificial_vars_.push_back(num_vars);
            basis_[i] = num_vars;
            ++num_vars;
        } else {
            basis_[i] = basis_col;
        }
    }
}

void SimplexBigM::build_initial_tableau() {
    size_t num_constraints = constraints_.size();
    size_t num_vars = constraints_[0].size();

    // Всегда добавляем M-строку
    size_t num_rows = num_constraints + 2; // Ограничения + Z-строка + M-строка
    tableau_.resize(num_rows, std::vector<Fraction>(num_vars + 1, Fraction(0)));

    for (size_t i = 0; i < num_constraints; ++i) {
        for (size_t j = 0; j < num_vars; ++j) {
            tableau_[i][j] = constraints_[i][j];
        }
        tableau_[i][num_vars] = rhs_[i];
    }

    z_row_index_ = num_constraints;
    m_row_index_ = num_constraints + 1;
    has_m_row_ = true;

    // Устанавливаем Z-строку как c_j (для максимизации)
    for (size_t j = 0; j < num_vars; ++j) {
        tableau_[z_row_index_][j] = obj_func_[j]; // Без инверсии
    }
    tableau_[z_row_index_][num_vars] = Fraction(0);

    // Инициализируем M-строку как -∑(строки с искусственными переменными)
    std::vector<Fraction> m_row(num_vars + 1, Fraction(0));
    for (size_t i = 0; i < basis_.size(); ++i) {
        if (std::find(artificial_vars_.begin(), artificial_vars_.end(), basis_[i]) != artificial_vars_.end()) {
            for (size_t j = 0; j < num_vars + 1; ++j) {
                m_row[j] = m_row[j] - tableau_[i][j];
            }
        }
    }
    tableau_[m_row_index_] = m_row;
}

void SimplexBigM::print_term(const Fraction& coef, size_t index, bool& first_term) const {
    if (first_term) {
        if (coef == Fraction(1)) std::cout << "x" << (index + 1);
        else if (coef == Fraction(-1)) std::cout << "-x" << (index + 1);
        else std::cout << coef.to_string() << " * x" << (index + 1);
    } else {
        std::cout << (coef >= Fraction(0) ? " + " : " - ");
        Fraction abs_coef = coef >= Fraction(0) ? coef : -coef;
        if (abs_coef == Fraction(1)) std::cout << "x" << (index + 1);
        else std::cout << abs_coef.to_string() << " * x" << (index + 1);
    }
    first_term = false;
}

void SimplexBigM::print_equation(const std::vector<Fraction>& coefs, const std::string& sign_or_eq, const Fraction& rhs) const {
    bool first_term = true;
    bool has_non_zero = false;
    for (size_t j = 0; j < coefs.size(); ++j) {
        if (coefs[j] != Fraction(0)) {
            print_term(coefs[j], j, first_term);
            has_non_zero = true;
        }
    }
    if (!has_non_zero) std::cout << "0";
    std::cout << " " << sign_or_eq << " " << rhs.to_string() << "\n";
}

void SimplexBigM::print_default_form() const {
    std::cout << "Исходная форма задачи:\n";
    std::cout << "Z = ";
    bool first_term = true;
    for (size_t i = 0; i < original_obj_func_.size(); ++i) {
        if (original_obj_func_[i] != Fraction(0)) {
            print_term(original_obj_func_[i], i, first_term);
        }
    }
    std::cout << " -> " << goal_ << "\n\n";
    std::cout << "При ограничениях:\n";
    for (size_t i = 0; i < constraints_.size(); ++i) {
        print_equation(constraints_[i], signs_[i], rhs_[i]);
    }
    std::cout << "\n";
}

void SimplexBigM::print_canonical_form() const {
    std::cout << "Каноническая форма задачи:\n";
    std::cout << "Z = ";
    bool first_term = true;
    for (size_t i = 0; i < obj_func_.size(); ++i) {
        if (obj_func_[i] != Fraction(0)) {
            print_term(obj_func_[i], i, first_term);
        }
    }
    std::cout << " -> max\n\n";
    std::cout << "При ограничениях:\n";
    for (size_t i = 0; i < constraints_.size(); ++i) {
        print_equation(constraints_[i], "=", rhs_[i]);
    }
    std::cout << "\n";
}

void SimplexBigM::print_canonical_with_artificial() const {
    std::cout << "Каноническая форма с искусственными переменными:\n";
    std::cout << "При ограничениях:\n";
    for (size_t i = 0; i < constraints_.size(); ++i) {
        print_equation(constraints_[i], "=", rhs_[i]);
    }
    std::cout << "\n";
}

void SimplexBigM::print_tableau(const std::pair<size_t, size_t>* pivot) const {
    std::cout << "Симплекс-таблица (итерация " << iteration_ << "):\n";
    size_t num_vars = tableau_[0].size() - 1;

    std::cout << "     ";
    for (size_t j = 0; j < num_vars; ++j) {
        std::cout << std::setw(8) << ("x" + std::to_string(j + 1));
    }
    std::cout << std::setw(8) << "1" << "\n";

    for (size_t i = 0; i < tableau_.size(); ++i) {
        if (i == z_row_index_) {
            std::cout << "Z    ";
        } else if (i == m_row_index_ && has_m_row_) {
            std::cout << "M    ";
        } else {
            std::cout << "x" << (basis_[i] + 1) << "   ";
        }
        for (size_t j = 0; j < tableau_[0].size(); ++j) {
            std::cout << std::setw(8) << tableau_[i][j].to_string();
        }
        std::cout << "\n";
    }
    if (pivot && pivot->first != std::numeric_limits<size_t>::max()) {
        std::cout << "\nВедущий столбец: x" << (pivot->second + 1)
                  << ", ведущая строка: " << (pivot->first + 1) << "\n";
    }
    std::cout << "\n";
}

bool SimplexBigM::is_optimal() const {
    // Проверяем M-строку, если она есть
    if (has_m_row_) {
        for (size_t j = 0; j < tableau_[m_row_index_].size() - 1; ++j) {
            if (tableau_[m_row_index_][j] < Fraction(0)) {
                return false;
            }
        }
        // Если M-строка нулевая, считаем её обработанной
        if (std::all_of(tableau_[m_row_index_].begin(), tableau_[m_row_index_].end() - 1,
                        [](const Fraction& x) { return x == Fraction(0); })) {
            return false; // Переходим к Фазе II
        }
    }
    // Проверяем Z-строку
    for (size_t j = 0; j < tableau_[z_row_index_].size() - 1; ++j) {
        if (tableau_[z_row_index_][j] < Fraction(0)) {
            return false;
        }
    }
    return true;
}

std::pair<size_t, size_t> SimplexBigM::get_pivot() const {
    // Если есть M-строка (Фаза I)
    if (has_m_row_) {
        const auto& m_row = tableau_[m_row_index_];
        Fraction min_val = Fraction(0);
        size_t col = std::numeric_limits<size_t>::max();
        for (size_t j = 0; j < m_row.size() - 1; ++j) {
            if (m_row[j] < min_val) {
                min_val = m_row[j];
                col = j;
            }
        }
        if (col == std::numeric_limits<size_t>::max()) {
            return {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
        }
        std::vector<std::pair<size_t, Fraction>> ratios;
        for (size_t i = 0; i < tableau_.size() - 2; ++i) {
            if (tableau_[i][col] > Fraction(0)) {
                ratios.emplace_back(i, tableau_[i].back() / tableau_[i][col]);
            } else if (tableau_[i][col] == Fraction(0) && tableau_[i].back() == Fraction(0)) {
                ratios.emplace_back(i, Fraction(0)); // Учитываем случай b_i = 0, a_{i,j} = 0
            }
        }
        if (ratios.empty()) {
            return {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
        }
        auto min_it = std::min_element(ratios.begin(), ratios.end(),
                                      [](const auto& a, const auto& b) { return a.second < b.second; });
        return {min_it->first, col};
    }

    // Фаза II: выбор столбца с наиболее отрицательным коэффициентом в Z-строке
    const auto& z_row = tableau_[z_row_index_];
    Fraction min_val = Fraction(0);
    size_t col = std::numeric_limits<size_t>::max();
    for (size_t j = 0; j < z_row.size() - 1; ++j) {
        if (z_row[j] < min_val) {
            min_val = z_row[j];
            col = j;
        }
    }
    if (col == std::numeric_limits<size_t>::max()) {
        return {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
    }

    // Выбор ведущей строки
    std::vector<std::pair<size_t, Fraction>> ratios;
    for (size_t i = 0; i < tableau_.size() - (has_m_row_ ? 2 : 1); ++i) {
        if (tableau_[i][col] > Fraction(0)) {
            ratios.emplace_back(i, tableau_[i].back() / tableau_[i][col]);
        } else if (tableau_[i][col] == Fraction(0) && tableau_[i].back() == Fraction(0)) {
            ratios.emplace_back(i, Fraction(0)); // Учитываем случай b_i = 0, a_{i,j} = 0
        }
    }
    if (ratios.empty()) {
        return {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
    }
    auto min_it = std::min_element(ratios.begin(), ratios.end(),
                                   [](const auto& a, const auto& b) { return a.second < b.second; });
    return {min_it->first, col};
}

void SimplexBigM::pivot(size_t row, size_t col) {
    Fraction pivot_val = tableau_[row][col];
    for (auto& val : tableau_[row]) {
        val = val / pivot_val;
    }
    for (size_t i = 0; i < tableau_.size(); ++i) {
        if (i != row) {
            Fraction factor = tableau_[i][col];
            for (size_t j = 0; j < tableau_[i].size(); ++j) {
                tableau_[i][j] = tableau_[i][j] - factor * tableau_[row][j];
            }
        }
    }
    basis_[row] = col;
    update_m_row();
}

void SimplexBigM::pivot_phase2(size_t row, size_t col) {
    Fraction pivot_val = tableau_[row][col];
    for (auto& val : tableau_[row]) {
        val = val / pivot_val;
    }
    for (size_t i = 0; i < tableau_.size(); ++i) {
        if (i != row) {
            Fraction factor = tableau_[i][col];
            for (size_t j = 0; j < tableau_[i].size(); ++j) {
                tableau_[i][j] = tableau_[i][j] - factor * tableau_[row][j];
            }
        }
    }
    basis_[row] = col;
    restore_original_z_row();
}

void SimplexBigM::update_m_row() {
    if (!has_m_row_) return;
    std::vector<Fraction> m_row(tableau_[0].size(), Fraction(0));
    for (size_t i = 0; i < basis_.size(); ++i) {
        if (std::find(artificial_vars_.begin(), artificial_vars_.end(), basis_[i]) != artificial_vars_.end()) {
            for (size_t j = 0; j < m_row.size(); ++j) {
                m_row[j] = m_row[j] - tableau_[i][j];
            }
        }
    }
    for (size_t var : artificial_vars_) {
        if (var < m_row.size()) {
            m_row[var] = Fraction(0);
        }
    }
    tableau_[m_row_index_] = m_row;
}

void SimplexBigM::restore_original_z_row() {
    std::vector<Fraction> z_row(tableau_[0].size(), Fraction(0));
    size_t num_vars = std::min(obj_func_.size(), z_row.size() - 1);
    for (size_t j = 0; j < num_vars; ++j) {
        z_row[j] = -obj_func_[j]; // Для максимизации используем -c_j
    }
    for (size_t i = 0; i < basis_.size(); ++i) {
        size_t var_idx = basis_[i];
        if (var_idx < obj_func_.size()) {
            Fraction coef = -obj_func_[var_idx];
            for (size_t j = 0; j < z_row.size(); ++j) {
                z_row[j] = z_row[j] - coef * tableau_[i][j];
            }
        }
    }
    tableau_[z_row_index_] = z_row;
}

void SimplexBigM::remove_artificial_vars() {
    std::vector<size_t> cols_to_remove;
    for (size_t var : artificial_vars_) {
        if (std::find(basis_.begin(), basis_.end(), var) == basis_.end()) {
            cols_to_remove.push_back(var);
        }
    }
    std::sort(cols_to_remove.begin(), cols_to_remove.end(), std::greater<size_t>());
    for (size_t col : cols_to_remove) {
        for (auto& row : tableau_) {
            row.erase(row.begin() + col);
        }
        for (auto& b : basis_) {
            if (b > col) --b;
        }
        for (auto& av : artificial_vars_) {
            if (av > col) --av;
        }
    }
    if (has_m_row_ && std::all_of(tableau_[m_row_index_].begin(), tableau_[m_row_index_].end() - 1,
                                  [](const Fraction& x) { return x == Fraction(0); })) {
        std::cout << "Удаляем M-строку\n\n";
        tableau_.erase(tableau_.begin() + m_row_index_);
        has_m_row_ = false;
        m_row_index_ = 0;
        z_row_index_ = tableau_.size() - 1;
    }
    artificial_vars_.erase(
        std::remove_if(artificial_vars_.begin(), artificial_vars_.end(),
                       [this](size_t var) { return std::find(basis_.begin(), basis_.end(), var) == basis_.end(); }),
        artificial_vars_.end());
}

bool SimplexBigM::is_infeasible_due_to_m_row() const {
    if (!has_m_row_) return false;
    const auto& m_row = tableau_[m_row_index_];
    bool all_non_negative = std::all_of(m_row.begin(), m_row.end() - 1,
                                        [](const Fraction& x) { return x >= Fraction(0); });
    if (!all_non_negative) return false;
    for (size_t i = 0; i < basis_.size(); ++i) {
        if (std::find(artificial_vars_.begin(), artificial_vars_.end(), basis_[i]) != artificial_vars_.end() &&
            tableau_[i].back() != Fraction(0)) {
            return true;
        }
    }
    return false;
}

std::vector<Fraction> SimplexBigM::get_current_solution() const {
    std::vector<Fraction> solution(tableau_[0].size() - 1, Fraction(0));
    for (size_t i = 0; i < basis_.size(); ++i) {
        size_t var_idx = basis_[i];
        if (var_idx < solution.size()) {
            solution[var_idx] = tableau_[i].back();
        }
    }
    return std::vector<Fraction>(solution.begin(), solution.begin() + original_vars_);
}

std::string SimplexBigM::format_solution(const std::vector<Fraction>& solution) const {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < solution.size(); ++i) {
        oss << solution[i].to_string();
        if (i < solution.size() - 1) {
            oss << ", ";
        }
    }
    oss << ")";
    return oss.str();
}

void SimplexBigM::print_solution() const {
    std::cout << "Оптимальное решение:\n";
    std::vector<Fraction> solution = get_current_solution();
    size_t num_vars = tableau_[0].size() - 1;
    std::vector<Fraction> full_solution(num_vars, Fraction(0));
    for (size_t i = 0; i < basis_.size(); ++i) {
        if (basis_[i] < num_vars) {
            full_solution[basis_[i]] = tableau_[i].back();
        }
    }
    for (size_t i = 0; i < num_vars; ++i) {
        std::cout << "x" << (i + 1) << " = " << full_solution[i].to_string() << "\n";
    }
    Fraction z_value = Fraction(0);
    for (size_t i = 0; i < original_obj_func_.size() && i < full_solution.size(); ++i) {
        z_value = z_value + original_obj_func_[i] * full_solution[i];
    }
    std::cout << "Z = " << z_value.to_string() << "\n\n";
    std::cout << "Z_" << goal_ << " = Z" << format_solution(full_solution) << " = " << z_value.to_string() << "\n";
}

void SimplexBigM::find_alternative_solutions() {
    //std::cout << "\nПроверка альтернативных оптимальных решений:\n";
    size_t z_row_idx = has_m_row_ ? z_row_index_ : tableau_.size() - 1;
    
    // Находим не базисные переменные с нулевым коэффициентом в Z-строке
    std::vector<size_t> non_basis_zero;
    for (size_t j = 0; j < tableau_[z_row_idx].size() - 1; ++j) {
        if (std::find(basis_.begin(), basis_.end(), j) == basis_.end() && tableau_[z_row_idx][j] == Fraction(0)) {
            non_basis_zero.push_back(j);
        }
    }

    if (non_basis_zero.empty()) {
        //std::cout << "Альтернативные оптимальные решения отсутствуют.\n";
        return;
    }

    // Сохраняем текущее состояние для восстановления
    auto original_tableau = tableau_;
    auto original_basis = basis_;
    std::vector<std::vector<Fraction>> solutions;
    solutions.push_back(get_current_solution()); // Первое решение

    // Проверяем каждую не базисную переменную с Z_j = 0
    for (size_t col : non_basis_zero) {
        std::vector<std::pair<size_t, Fraction>> ratios;
        for (size_t i = 0; i < tableau_.size() - 1; ++i) {
            if (tableau_[i][col] > Fraction(0)) {
                ratios.emplace_back(i, tableau_[i].back() / tableau_[i][col]);
            }
        }
        if (!ratios.empty()) {
            auto min_it = std::min_element(ratios.begin(), ratios.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
            size_t row = min_it->first;
            
            // Выполняем вращение
            pivot_phase2(row, col);
            std::cout << "Альтернативное оптимальное решение:\n";
            print_solution();
            solutions.push_back(get_current_solution());
            
            // Восстанавливаем таблицу
            tableau_ = original_tableau;
            basis_ = original_basis;
        }
    }

    // Вывод общего вида решения (для первых двух решений)
    if (solutions.size() >= 2) {
        std::cout << "\nСуществует бесконечно много оптимальных решений.\n";
        std::cout << "Общий вид:\n λ * X₁ + (1-λ) * X₂, где 0 ≤ λ ≤ 1\n";
        std::cout << "X₁ = " << format_solution(solutions[0]) << "\n";
        std::cout << "X₂ = " << format_solution(solutions[1]) << "\n\n";

        // Вычисляем общее решение в раскрытом виде
        std::cout << "Общее решение в раскрытом виде:\n(";
        for (size_t i = 0; i < solutions[0].size(); ++i) {
            Fraction coef_lambda = solutions[0][i] - solutions[1][i];
            Fraction const_term = solutions[1][i];
            if (coef_lambda != Fraction(0)) {
                std::cout << const_term.to_string();
                if (coef_lambda > Fraction(0)) {
                    std::cout << " + " << coef_lambda.to_string() << "λ";
                } else {
                    std::cout << " - " << (-coef_lambda).to_string() << "λ";
                }
            } else {
                std::cout << const_term.to_string();
            }
            if (i < solutions[0].size() - 1) std::cout << ", ";
        }
        std::cout << ")\n";

        // Проверяем значение Z
        Fraction z_value = Fraction(0);
        for (size_t i = 0; i < original_obj_func_.size() && i < solutions[0].size(); ++i) {
            z_value = z_value + original_obj_func_[i] * solutions[0][i];
        }
        std::cout << "Z = " << z_value.to_string() << "\n";
    }
}

void SimplexBigM::solve() {
    // Выводим начальную таблицу
    print_tableau();

    // Проверка искусственных переменных в базисе
    bool has_nonzero_artificial = false;
    for (size_t i = 0; i < basis_.size(); ++i) {
        if (std::find(artificial_vars_.begin(), artificial_vars_.end(), basis_[i]) != artificial_vars_.end() &&
            tableau_[i].back() != Fraction(0)) {
            has_nonzero_artificial = true;
            break;
        }
    }

    // Фаза I (только если есть ненулевые искусственные переменные)
    if (has_nonzero_artificial) {
        while (true) {
            update_m_row();
            if (has_m_row_ && std::all_of(tableau_[m_row_index_].begin(), tableau_[m_row_index_].end() - 1,
                                          [](const Fraction& x) { return x == Fraction(0); })) {
                std::cout << "M-строка нулевая, удаляем ее.\n";
                std::cout << "Удаляем M-строку\n";
                tableau_.erase(tableau_.begin() + m_row_index_);
                has_m_row_ = false;
                z_row_index_ = tableau_.size() - 1;
                print_tableau();
                break;
            }
            if (is_infeasible_due_to_m_row()) {
                print_tableau();
                std::cout << "Система ограничений несовместна.\n";
                return;
            }
            auto pivot_pos = get_pivot();
            if (pivot_pos.first == std::numeric_limits<size_t>::max()) {
                print_tableau();
                std::cout << "Решение не ограничено или несовместно в Фазе I.\n";
                return;
            }
            print_tableau(&pivot_pos);
            pivot(pivot_pos.first, pivot_pos.second);
            ++iteration_;
        }
    }

    // Проверка искусственных переменных
    for (size_t i = 0; i < basis_.size(); ++i) {
        if (std::find(artificial_vars_.begin(), artificial_vars_.end(), basis_[i]) != artificial_vars_.end() &&
            tableau_[i].back() != Fraction(0)) {
            print_tableau();
            std::cout << "Нет допустимого решения (искусственные переменные остались в базисе с ненулевыми значениями)\n";
            return;
        }
    }

    // Удаление искусственных переменных и восстановление Z-строки
    remove_artificial_vars();
    restore_original_z_row();

    // Фаза II
    while (!is_optimal()) {
        auto pivot_pos = get_pivot();
        if (pivot_pos.first == std::numeric_limits<size_t>::max()) {
            // Проверяем, не неограничено ли решение
            const auto& z_row = tableau_[z_row_index_];
            bool unbounded = false;
            for (size_t j = 0; j < z_row.size() - 1; ++j) {
                if (z_row[j] < Fraction(0)) {
                    bool has_positive = false;
                    for (size_t i = 0; i < tableau_.size() - 1; ++i) {
                        if (tableau_[i][j] > Fraction(0)) {
                            has_positive = true;
                            break;
                        }
                    }
                    if (!has_positive) {
                        unbounded = true;
                        break;
                    }
                }
            }
            if (unbounded) {
                std::cout << "Решение не ограничено.\n";
                return;
            }
            break;
        }
        print_tableau(&pivot_pos);
        pivot_phase2(pivot_pos.first, pivot_pos.second);
        ++iteration_;
    }

    print_tableau();
    print_solution();
    find_alternative_solutions();
}