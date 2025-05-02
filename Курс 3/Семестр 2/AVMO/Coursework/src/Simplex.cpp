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
    : original_obj_func_(obj_func), // Сохраняем исходные коэффициенты
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
    print_tableau();
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

    size_t num_rows = num_constraints + 1 + (artificial_vars_.empty() ? 0 : 1);
    tableau_.resize(num_rows, std::vector<Fraction>(num_vars + 1, Fraction(0)));

    for (size_t i = 0; i < num_constraints; ++i) {
        for (size_t j = 0; j < num_vars; ++j) {
            tableau_[i][j] = constraints_[i][j];
        }
        tableau_[i][num_vars] = rhs_[i];
    }

    z_row_index_ = num_rows - 1;
    if (!artificial_vars_.empty()) {
        z_row_index_--;
        m_row_index_ = num_rows - 1;
    }
    for (size_t j = 0; j < num_vars; ++j) {
        tableau_[z_row_index_][j] = -obj_func_[j];
    }
    tableau_[z_row_index_][num_vars] = Fraction(0);

    if (!artificial_vars_.empty()) {
        for (size_t j = 0; j < num_vars; ++j) {
            Fraction sum(0);
            for (size_t i = 0; i < num_constraints; ++i) {
                if (std::find(artificial_vars_.begin(), artificial_vars_.end(), basis_[i]) != artificial_vars_.end()) {
                    sum = sum + constraints_[i][j];
                }
            }
            if (std::find(artificial_vars_.begin(), artificial_vars_.end(), j) != artificial_vars_.end()) {
                sum = Fraction(0);
            }
            tableau_[m_row_index_][j] = -sum;
        }
        Fraction sum_b(0);
        for (size_t i = 0; i < num_constraints; ++i) {
            if (std::find(artificial_vars_.begin(), artificial_vars_.end(), basis_[i]) != artificial_vars_.end()) {
                sum_b = sum_b + rhs_[i];
            }
        }
        tableau_[m_row_index_][num_vars] = -sum_b;
    } else {
        has_m_row_ = false;
    }
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
    std::cout << (goal_ == "max" ? "Максимизировать" : "Минимизировать") << "\n";
    std::cout << "Z = ";
    bool first_term = true;
    for (size_t i = 0; i < original_obj_func_.size(); ++i) {
        if (original_obj_func_[i] != Fraction(0)) {
            print_term(original_obj_func_[i], i, first_term);
        }
    }
    std::cout << "\n\nПри ограничениях:\n";
    for (size_t i = 0; i < constraints_.size(); ++i) {
        print_equation(constraints_[i], signs_[i], rhs_[i]);
    }
    std::cout << "\n";
}

void SimplexBigM::print_canonical_form() const {
    std::cout << "Каноническая форма задачи:\n";
    std::cout << "Максимизировать\nZ = ";
    bool first_term = true;
    for (size_t i = 0; i < obj_func_.size(); ++i) {
        if (obj_func_[i] != Fraction(0)) {
            print_term(obj_func_[i], i, first_term);
        }
    }
    std::cout << "\n\nПри ограничениях:\n";
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
    std::cout << std::setw(8) << "b" << "\n";

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
    // Если есть M-строка, проверяем, есть ли в ней отрицательные коэффициенты
    if (has_m_row_) {
        for (size_t j = 0; j < tableau_[m_row_index_].size() - 1; ++j) {
            if (tableau_[m_row_index_][j] < Fraction(0)) {
                return false; // M-строка еще не оптимизирована
            }
        }
    }
    // Проверяем Z-строку: все коэффициенты должны быть неотрицательными
    size_t target_row = has_m_row_ ? z_row_index_ : tableau_.size() - 1;
    for (size_t j = 0; j < tableau_[target_row].size() - 1; ++j) {
        if (tableau_[target_row][j] < Fraction(0)) {
            return false; // Z-строка еще не оптимизирована
        }
    }
    return true;
}

std::pair<size_t, size_t> SimplexBigM::get_pivot() const {
    // Сначала проверяем M-строку, если она есть
    if (has_m_row_) {
        const auto& m_row = tableau_[m_row_index_];
        Fraction min_val = Fraction(0);
        size_t col = std::numeric_limits<size_t>::max();
        // Находим самый отрицательный коэффициент в M-строке
        for (size_t j = 0; j < m_row.size() - 1; ++j) {
            if (m_row[j] < min_val) {
                min_val = m_row[j];
                col = j;
            }
        }
        if (col != std::numeric_limits<size_t>::max()) {
            // Находим ведущую строку: минимальное положительное отношение b/a
            std::vector<std::pair<size_t, Fraction>> ratios;
            for (size_t i = 0; i < tableau_.size() - (has_m_row_ ? 2 : 1); ++i) {
                if (tableau_[i][col] > Fraction(0)) {
                    ratios.emplace_back(i, tableau_[i].back() / tableau_[i][col]);
                }
            }
            if (ratios.empty()) {
                // Все коэффициенты в столбце неположительные или нулевые
                bool all_non_positive = true;
                for (size_t i = 0; i < tableau_.size() - (has_m_row_ ? 2 : 1); ++i) {
                    if (tableau_[i][col] > Fraction(0)) {
                        all_non_positive = false;
                        break;
                    }
                }
                if (all_non_positive) {
                    std::cout << "Решение не ограничено: симплекс-метод не может продолжаться.\n";
                } else {
                    std::cout << "Нет допустимого решения: невозможно выбрать ведущую строку.\n";
                }
                return {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
            }
            // Выбираем строку с минимальным отношением
            auto min_it = std::min_element(ratios.begin(), ratios.end(),
                                          [](const auto& a, const auto& b) { return a.second < b.second; });
            return {min_it->first, col};
        }
    }

    // Проверяем Z-строку
    const auto& z_row = has_m_row_ ? tableau_[z_row_index_] : tableau_.back();
    Fraction min_val = Fraction(0);
    size_t col = std::numeric_limits<size_t>::max();
    for (size_t j = 0; j < z_row.size() - 1; ++j) {
        if (z_row[j] < min_val) {
            min_val = z_row[j];
            col = j;
        }
    }
    if (col != std::numeric_limits<size_t>::max()) {
        std::vector<std::pair<size_t, Fraction>> ratios;
        size_t limit = has_m_row_ ? tableau_.size() - 2 : tableau_.size() - 1;
        for (size_t i = 0; i < limit; ++i) {
            if (tableau_[i][col] > Fraction(0)) {
                ratios.emplace_back(i, tableau_[i].back() / tableau_[i][col]);
            }
        }
        if (ratios.empty()) {
            bool all_non_positive = true;
            for (size_t i = 0; i < limit; ++i) {
                if (tableau_[i][col] > Fraction(0)) {
                    all_non_positive = false;
                    break;
                }
            }
            if (all_non_positive) {
                std::cout << "Решение не ограничено.\n";
            } else {
                std::cout << "Нет допустимого решения: невозможно выбрать ведущую строку.\n";
            }
            return {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
        }
        auto min_it = std::min_element(ratios.begin(), ratios.end(),
                                      [](const auto& a, const auto& b) { return a.second < b.second; });
        return {min_it->first, col};
    }

    return {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
}

void SimplexBigM::pivot(size_t row, size_t col) {
    // Нормализуем ведущую строку
    Fraction pivot_val = tableau_[row][col];
    for (auto& val : tableau_[row]) {
        val = val / pivot_val;
    }
    // Обновляем остальные строки
    for (size_t i = 0; i < tableau_.size(); ++i) {
        if (i != row) {
            Fraction factor = tableau_[i][col];
            for (size_t j = 0; j < tableau_[i].size(); ++j) {
                tableau_[i][j] = tableau_[i][j] - factor * tableau_[row][j];
            }
        }
    }
    // Обновляем базис
    basis_[row] = col;
    // Обновляем M-строку
    update_m_row();
}

void SimplexBigM::pivot_phase2(size_t row, size_t col) {
    // Нормализуем ведущую строку
    Fraction pivot_val = tableau_[row][col];
    for (auto& val : tableau_[row]) {
        val = val / pivot_val;
    }
    // Обновляем остальные строки
    for (size_t i = 0; i < tableau_.size(); ++i) {
        if (i != row) {
            Fraction factor = tableau_[i][col];
            for (size_t j = 0; j < tableau_[i].size(); ++j) {
                tableau_[i][j] = tableau_[i][j] - factor * tableau_[row][j];
            }
        }
    }
    // Обновляем базис
    basis_[row] = col;
    // Пересчитываем Z-строку
    restore_original_z_row();
}

void SimplexBigM::update_m_row() {
    if (!has_m_row_) return;
    // Инициализируем M-строку нулями
    std::vector<Fraction> m_row(tableau_[0].size(), Fraction(0));
    // Вычитаем строки, соответствующие искусственным переменным в базисе
    for (size_t i = 0; i < basis_.size(); ++i) {
        if (std::find(artificial_vars_.begin(), artificial_vars_.end(), basis_[i]) != artificial_vars_.end()) {
            for (size_t j = 0; j < m_row.size(); ++j) {
                m_row[j] = m_row[j] - tableau_[i][j];
            }
        }
    }
    // Обнуляем коэффициенты при искусственных переменных
    for (size_t var : artificial_vars_) {
        if (var < m_row.size()) {
            m_row[var] = Fraction(0);
        }
    }
    tableau_[m_row_index_] = m_row;
}

void SimplexBigM::restore_original_z_row() {
    // Инициализируем Z-строку нулями
    std::vector<Fraction> z_row(tableau_[0].size(), Fraction(0));
    // Заполняем коэффициенты исходной целевой функции
    size_t num_vars = std::min(obj_func_.size(), z_row.size() - 1);
    for (size_t j = 0; j < num_vars; ++j) {
        z_row[j] = -obj_func_[j]; // Для max: -c_j, для min: c_j
    }
    // Учитываем текущий базис
    for (size_t i = 0; i < basis_.size(); ++i) {
        size_t var_idx = basis_[i];
        if (var_idx < obj_func_.size()) {
            Fraction coef = -obj_func_[var_idx]; // Для max: -c_j, для min: c_j
            for (size_t j = 0; j < z_row.size(); ++j) {
                z_row[j] = z_row[j] - coef * tableau_[i][j];
            }
        }
    }
    // Устанавливаем новую Z-строку
    size_t z_row_idx = has_m_row_ ? z_row_index_ : tableau_.size() - 1;
    tableau_[z_row_idx] = z_row;
}

void SimplexBigM::remove_artificial_vars() {
    // Собираем столбцы искусственных переменных, которые не в базисе
    std::vector<size_t> cols_to_remove;
    for (size_t var : artificial_vars_) {
        if (std::find(basis_.begin(), basis_.end(), var) == basis_.end()) {
            cols_to_remove.push_back(var);
        }
    }
    // Сортируем в обратном порядке для безопасного удаления
    std::sort(cols_to_remove.begin(), cols_to_remove.end(), std::greater<size_t>());
    // Удаляем столбцы
    for (size_t col : cols_to_remove) {
        for (auto& row : tableau_) {
            row.erase(row.begin() + col);
        }
        // Обновляем индексы базисных переменных и искусственных переменных
        for (auto& b : basis_) {
            if (b > col) --b;
        }
        for (auto& av : artificial_vars_) {
            if (av > col) --av;
        }
    }
    // Проверяем, можно ли удалить M-строку
    if (has_m_row_ && std::all_of(tableau_[m_row_index_].begin(), tableau_[m_row_index_].end() - 1,
                                  [](const Fraction& x) { return x == Fraction(0); })) {
        std::cout << "Удаляем M-строку\n";
        tableau_.erase(tableau_.begin() + m_row_index_);
        has_m_row_ = false;
        m_row_index_ = 0;
        z_row_index_ = tableau_.size() - 1;
    }
    // Обновляем artificial_vars_, оставляя только те, что в базисе
    artificial_vars_.erase(
        std::remove_if(artificial_vars_.begin(), artificial_vars_.end(),
                       [this](size_t var) { return std::find(basis_.begin(), basis_.end(), var) == basis_.end(); }),
        artificial_vars_.end());
}

bool SimplexBigM::is_infeasible_due_to_m_row() const {
    if (!has_m_row_) return false;
    const auto& m_row = tableau_[m_row_index_];
    // Проверяем, все ли коэффициенты в M-строке (кроме последнего) неотрицательны
    bool all_non_negative = std::all_of(m_row.begin(), m_row.end() - 1,
                                        [](const Fraction& x) { return x >= Fraction(0); });
    if (!all_non_negative) return false;
    // Проверяем, есть ли искусственные переменные в базисе с ненулевыми значениями
    for (size_t i = 0; i < basis_.size(); ++i) {
        if (std::find(artificial_vars_.begin(), artificial_vars_.end(), basis_[i]) != artificial_vars_.end() &&
            tableau_[i].back() != Fraction(0)) {
            return true;
        }
    }
    return false;
}

std::vector<Fraction> SimplexBigM::get_current_solution() const {
    // Инициализируем решение нулями для всех переменных
    std::vector<Fraction> solution(tableau_[0].size() - 1, Fraction(0));
    // Заполняем значения базисных переменных
    for (size_t i = 0; i < basis_.size(); ++i) {
        size_t var_idx = basis_[i];
        if (var_idx < solution.size()) {
            solution[var_idx] = tableau_[i].back();
        }
    }
    // Возвращаем только исходные переменные
    return std::vector<Fraction>(solution.begin(), solution.begin() + original_vars_);
}

std::string SimplexBigM::format_solution(const std::vector<Fraction>& solution) const {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < solution.size(); ++i) {
        oss << solution[i].to_string();
        if (i < solution.size() - 1) {
            oss << ";";
        }
    }
    oss << ")";
    return oss.str();
}

void SimplexBigM::print_solution() const {
    std::cout << "\nОптимальное решение:\n";
    std::vector<Fraction> solution = get_current_solution();
    size_t num_vars = tableau_[0].size() - 1;
    // Выводим все переменные, включая слак-переменные
    std::vector<Fraction> full_solution(num_vars, Fraction(0));
    for (size_t i = 0; i < basis_.size(); ++i) {
        if (basis_[i] < num_vars) {
            full_solution[basis_[i]] = tableau_[i].back();
        }
    }
    for (size_t i = 0; i < num_vars; ++i) {
        std::cout << "x" << (i + 1) << " = " << full_solution[i].to_string() << "\n";
    }
    // Вычисляем Z (в терминах исходной функции)
    Fraction z_value = Fraction(0);
    for (size_t i = 0; i < original_obj_func_.size() && i < full_solution.size(); ++i) {
        z_value = z_value + original_obj_func_[i] * full_solution[i];
    }
    std::cout << "Z = " << z_value.to_string() << "\n\n";
    // Формат Z_min/max
    std::cout << "Z_" << goal_ << " = Z" << format_solution(full_solution) << " = " << z_value.to_string() << "\n";
}

void SimplexBigM::solve() {
    // Фаза I: Минимизация искусственных переменных
    std::cout << "Начало Фазы I\n";
    while (true) {
        update_m_row();
        if (has_m_row_ && std::all_of(tableau_[m_row_index_].begin(), tableau_[m_row_index_].end() - 1,
                                      [](const Fraction& x) { return x == Fraction(0); })) {
            print_tableau();
            std::cout << "M-строка нулевая, Фаза I завершена.\n";
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
            std::cout << "Нет допустимого решения или решение не ограничено.\n";
            return;
        }
        print_tableau(&pivot_pos);
        pivot(pivot_pos.first, pivot_pos.second);
        ++iteration_;
    }

    // Проверяем, остались ли искусственные переменные в базисе с ненулевыми значениями
    for (size_t i = 0; i < basis_.size(); ++i) {
        if (std::find(artificial_vars_.begin(), artificial_vars_.end(), basis_[i]) != artificial_vars_.end() &&
            tableau_[i].back() != Fraction(0)) {
            print_tableau();
            std::cout << "Нет допустимого решения (искусственные переменные остались в базисе с ненулевыми значениями)\n";
            return;
        }
    }

    // Удаляем искусственные переменные и M-строку
    remove_artificial_vars();
    restore_original_z_row();
    std::cout << "Начало Фазы II\n";
    print_tableau();

    // Фаза II: Оптимизация целевой функции
    while (!is_optimal()) {
        auto pivot_pos = get_pivot();
        if (pivot_pos.first == std::numeric_limits<size_t>::max()) {
            print_tableau();
            std::cout << "Не удалось продолжить оптимизацию.\n";
            return;
        }
        print_tableau(&pivot_pos);
        pivot_phase2(pivot_pos.first, pivot_pos.second);
        ++iteration_;
    }

    // Выводим финальное решение
    print_tableau();
    print_solution();
}