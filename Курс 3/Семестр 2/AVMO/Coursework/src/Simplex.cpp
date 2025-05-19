#include "Simplex.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <sstream>

Simplex::Simplex(const std::vector<Fraction>& objective_coeffs,
                 const std::vector<std::vector<Fraction>>& constraint_coeffs,
                 const std::vector<std::string>& constraint_signs,
                 const std::vector<Fraction>& constraint_rhs,
                 const std::string& optimization_goal)
    : original_objective_coeffs_(objective_coeffs),
      objective_coeffs_(objective_coeffs),
      constraint_coeffs_(constraint_coeffs),
      constraint_signs_(constraint_signs),
      constraint_rhs_(constraint_rhs),
      optimization_goal_(optimization_goal),
      num_original_vars_(objective_coeffs.size()),
      has_big_m_row_(true),
      objective_row_index_(0),
      big_m_row_index_(0),
      current_iteration_(1),
      big_m_value_(1000) {
    print_original_problem_form();
    transform_to_canonical_form();
    print_canonical_problem_form();
    add_artificial_variables();
    print_canonical_form_with_artificial_variables();
    create_initial_simplex_tableau();
}

void Simplex::transform_to_canonical_form() {
    std::vector<Fraction> new_objective_coeffs = objective_coeffs_;
    if (optimization_goal_ == "min") {
        for (auto& coef : new_objective_coeffs) {
            coef = -coef;
        }
    }

    std::vector<std::vector<Fraction>> new_constraint_coeffs;
    size_t slack_vars = 0;

    for (size_t i = 0; i < constraint_coeffs_.size(); ++i) {
        std::vector<Fraction> row = constraint_coeffs_[i];
        Fraction b = constraint_rhs_[i];
        std::string sign = constraint_signs_[i];

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
            new_objective_coeffs.push_back(Fraction(0));
            ++slack_vars;
        } else if (sign == ">=") {
            row.push_back(Fraction(-1));
            new_objective_coeffs.push_back(Fraction(0));
            ++slack_vars;
        }

        new_constraint_coeffs.push_back(row);
        constraint_rhs_[i] = b;
    }

    size_t max_len = new_objective_coeffs.size();
    for (const auto& row : new_constraint_coeffs) {
        max_len = std::max(max_len, row.size());
    }
    for (auto& row : new_constraint_coeffs) {
        row.resize(max_len, Fraction(0));
    }
    new_objective_coeffs.resize(max_len, Fraction(0));

    objective_coeffs_ = new_objective_coeffs;
    constraint_coeffs_ = new_constraint_coeffs;
    constraint_signs_ = std::vector<std::string>(constraint_coeffs_.size(), "=");
}

void Simplex::add_artificial_variables() {
    size_t num_vars = constraint_coeffs_[0].size();
    size_t num_constraints = constraint_coeffs_.size();
    basis_indices_.resize(num_constraints, 0);

    for (size_t i = 0; i < num_constraints; ++i) {
        bool has_basis = false;
        size_t basis_col = 0;
        for (size_t j = 0; j < num_vars; ++j) {
            bool is_unit = constraint_coeffs_[i][j] == Fraction(1);
            bool is_unique = true;
            for (size_t k = 0; k < num_constraints; ++k) {
                if (k != i && constraint_coeffs_[k][j] != Fraction(0)) {
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
            for (auto& row : constraint_coeffs_) {
                row.push_back(Fraction(0));
            }
            constraint_coeffs_[i].back() = Fraction(1);
            objective_coeffs_.push_back(Fraction(0));
            artificial_var_indices_.push_back(num_vars);
            basis_indices_[i] = num_vars;
            ++num_vars;
        } else {
            basis_indices_[i] = basis_col;
        }
    }
}

void Simplex::create_initial_simplex_tableau() {
    size_t num_constraints = constraint_coeffs_.size();
    size_t num_vars = constraint_coeffs_[0].size();

    size_t num_rows = num_constraints + 2;
    simplex_tableau_.resize(num_rows, std::vector<Fraction>(num_vars + 1, Fraction(0)));

    for (size_t i = 0; i < num_constraints; ++i) {
        for (size_t j = 0; j < num_vars; ++j) {
            simplex_tableau_[i][j] = constraint_coeffs_[i][j];
        }
        simplex_tableau_[i][num_vars] = constraint_rhs_[i];
    }

    objective_row_index_ = num_constraints;
    big_m_row_index_ = num_constraints + 1;
    has_big_m_row_ = true;

    for (size_t j = 0; j < num_vars; ++j) {
        simplex_tableau_[objective_row_index_][j] = objective_coeffs_[j];
    }
    simplex_tableau_[objective_row_index_][num_vars] = Fraction(0);

    std::vector<Fraction> big_m_row(num_vars + 1, Fraction(0));
    for (size_t i = 0; i < basis_indices_.size(); ++i) {
        if (std::find(artificial_var_indices_.begin(), artificial_var_indices_.end(), basis_indices_[i]) != artificial_var_indices_.end()) {
            for (size_t j = 0; j < num_vars + 1; ++j) {
                big_m_row[j] = big_m_row[j] - simplex_tableau_[i][j];
            }
        }
    }
    simplex_tableau_[big_m_row_index_] = big_m_row;
}

void Simplex::print_equation_term(const Fraction& coef, size_t index, bool& first_term) const {
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

void Simplex::print_constraint_equation(const std::vector<Fraction>& coefs, const std::string& sign, const Fraction& rhs) const {
    bool first_term = true;
    bool has_non_zero = false;
    for (size_t j = 0; j < coefs.size(); ++j) {
        if (coefs[j] != Fraction(0)) {
            print_equation_term(coefs[j], j, first_term);
            has_non_zero = true;
        }
    }
    if (!has_non_zero) std::cout << "0";
    std::cout << " " << sign << " " << rhs.to_string() << "\n";
}

void Simplex::print_original_problem_form() const {
    std::cout << "Исходная форма задачи:\n";
    std::cout << "Z = ";
    bool first_term = true;
    for (size_t i = 0; i < original_objective_coeffs_.size(); ++i) {
        if (original_objective_coeffs_[i] != Fraction(0)) {
            print_equation_term(original_objective_coeffs_[i], i, first_term);
        }
    }
    std::cout << " -> " << optimization_goal_ << "\n\n";
    std::cout << "При ограничениях:\n";
    for (size_t i = 0; i < constraint_coeffs_.size(); ++i) {
        print_constraint_equation(constraint_coeffs_[i], constraint_signs_[i], constraint_rhs_[i]);
    }
    std::cout << "\n";
}

void Simplex::print_canonical_problem_form() const {
    std::cout << "Каноническая форма задачи:\n";
    std::cout << "Z = ";
    bool first_term = true;
    for (size_t i = 0; i < objective_coeffs_.size(); ++i) {
        if (objective_coeffs_[i] != Fraction(0)) {
            print_equation_term(objective_coeffs_[i], i, first_term);
        }
    }
    std::cout << " -> max\n\n";
    std::cout << "При ограничениях:\n";
    for (size_t i = 0; i < constraint_coeffs_.size(); ++i) {
        print_constraint_equation(constraint_coeffs_[i], "=", constraint_rhs_[i]);
    }
    std::cout << "\n";
}

void Simplex::print_canonical_form_with_artificial_variables() const {
    std::cout << "Каноническая форма с искусственными переменными:\n";
    std::cout << "При ограничениях:\n";
    for (size_t i = 0; i < constraint_coeffs_.size(); ++i) {
        print_constraint_equation(constraint_coeffs_[i], "=", constraint_rhs_[i]);
    }
    std::cout << "\n";
}

void Simplex::print_simplex_tableau(const std::pair<size_t, size_t>* pivot) const {
    std::cout << "Симплекс-таблица (итерация " << current_iteration_ << "):\n";
    size_t num_vars = simplex_tableau_[0].size() - 1;

    std::cout << "     ";
    for (size_t j = 0; j < num_vars; ++j) {
        std::cout << std::setw(8) << ("x" + std::to_string(j + 1));
    }
    std::cout << std::setw(8) << "1" << "\n";

    for (size_t i = 0; i < simplex_tableau_.size(); ++i) {
        if (i == objective_row_index_) {
            std::cout << "Z    ";
        } else if (i == big_m_row_index_ && has_big_m_row_) {
            std::cout << "M    ";
        } else {
            std::cout << "x" << (basis_indices_[i] + 1) << "   ";
        }
        for (size_t j = 0; j < simplex_tableau_[0].size(); ++j) {
            std::cout << std::setw(8) << simplex_tableau_[i][j].to_string();
        }
        std::cout << "\n";
    }
    if (pivot && pivot->first != std::numeric_limits<size_t>::max()) {
        std::cout << "\nВедущий столбец: x" << (pivot->second + 1)
                  << ", ведущая строка: " << (pivot->first + 1) << "\n";
    }
    std::cout << "\n";
}

bool Simplex::is_solution_optimal() const {
    if (has_big_m_row_) {
        for (size_t j = 0; j < simplex_tableau_[big_m_row_index_].size() - 1; ++j) {
            if (simplex_tableau_[big_m_row_index_][j] < Fraction(0)) {
                return false;
            }
        }
        if (std::all_of(simplex_tableau_[big_m_row_index_].begin(), simplex_tableau_[big_m_row_index_].end() - 1,
                        [](const Fraction& x) { return x == Fraction(0); })) {
            return false;
        }
    }
    for (size_t j = 0; j < simplex_tableau_[objective_row_index_].size() - 1; ++j) {
        if (simplex_tableau_[objective_row_index_][j] < Fraction(0)) {
            return false;
        }
    }
    return true;
}

std::pair<size_t, size_t> Simplex::find_pivot_element() const {
    if (has_big_m_row_) {
        const auto& big_m_row = simplex_tableau_[big_m_row_index_];
        Fraction min_val = Fraction(0);
        size_t col = std::numeric_limits<size_t>::max();
        for (size_t j = 0; j < big_m_row.size() - 1; ++j) {
            if (big_m_row[j] < min_val) {
                min_val = big_m_row[j];
                col = j;
            }
        }
        if (col == std::numeric_limits<size_t>::max()) {
            return {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
        }
        std::vector<std::pair<size_t, Fraction>> ratios;
        for (size_t i = 0; i < simplex_tableau_.size() - 2; ++i) {
            if (simplex_tableau_[i][col] > Fraction(0)) {
                ratios.emplace_back(i, simplex_tableau_[i].back() / simplex_tableau_[i][col]);
            } else if (simplex_tableau_[i][col] == Fraction(0) && simplex_tableau_[i].back() == Fraction(0)) {
                ratios.emplace_back(i, Fraction(0));
            }
        }
        if (ratios.empty()) {
            return {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
        }
        auto min_it = std::min_element(ratios.begin(), ratios.end(),
                                      [](const auto& a, const auto& b) { return a.second < b.second; });
        return {min_it->first, col};
    }

    const auto& objective_row = simplex_tableau_[objective_row_index_];
    Fraction min_val = Fraction(0);
    size_t col = std::numeric_limits<size_t>::max();
    for (size_t j = 0; j < objective_row.size() - 1; ++j) {
        if (objective_row[j] < min_val) {
            min_val = objective_row[j];
            col = j;
        }
    }
    if (col == std::numeric_limits<size_t>::max()) {
        return {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
    }

    std::vector<std::pair<size_t, Fraction>> ratios;
    for (size_t i = 0; i < simplex_tableau_.size() - (has_big_m_row_ ? 2 : 1); ++i) {
        if (simplex_tableau_[i][col] > Fraction(0)) {
            ratios.emplace_back(i, simplex_tableau_[i].back() / simplex_tableau_[i][col]);
        } else if (simplex_tableau_[i][col] == Fraction(0) && simplex_tableau_[i].back() == Fraction(0)) {
            ratios.emplace_back(i, Fraction(0));
        }
    }
    if (ratios.empty()) {
        return {std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
    }
    auto min_it = std::min_element(ratios.begin(), ratios.end(),
                                   [](const auto& a, const auto& b) { return a.second < b.second; });
    return {min_it->first, col};
}

void Simplex::execute_simplex_pivot(size_t row, size_t col) {
    Fraction pivot_val = simplex_tableau_[row][col];
    for (auto& val : simplex_tableau_[row]) {
        val = val / pivot_val;
    }
    for (size_t i = 0; i < simplex_tableau_.size(); ++i) {
        if (i != row) {
            Fraction factor = simplex_tableau_[i][col];
            for (size_t j = 0; j < simplex_tableau_[i].size(); ++j) {
                simplex_tableau_[i][j] = simplex_tableau_[i][j] - factor * simplex_tableau_[row][j];
            }
        }
    }
    basis_indices_[row] = col;
    update_big_m_row();
}

void Simplex::execute_phase_two_pivot(size_t row, size_t col) {
    Fraction pivot_val = simplex_tableau_[row][col];
    for (auto& val : simplex_tableau_[row]) {
        val = val / pivot_val;
    }
    for (size_t i = 0; i < simplex_tableau_.size(); ++i) {
        if (i != row) {
            Fraction factor = simplex_tableau_[i][col];
            for (size_t j = 0; j < simplex_tableau_[i].size(); ++j) {
                simplex_tableau_[i][j] = simplex_tableau_[i][j] - factor * simplex_tableau_[row][j];
            }
        }
    }
    basis_indices_[row] = col;
    restore_objective_row();
}

void Simplex::update_big_m_row() {
    if (!has_big_m_row_) return;
    std::vector<Fraction> big_m_row(simplex_tableau_[0].size(), Fraction(0));
    for (size_t i = 0; i < basis_indices_.size(); ++i) {
        if (std::find(artificial_var_indices_.begin(), artificial_var_indices_.end(), basis_indices_[i]) != artificial_var_indices_.end()) {
            for (size_t j = 0; j < big_m_row.size(); ++j) {
                big_m_row[j] = big_m_row[j] - simplex_tableau_[i][j];
            }
        }
    }
    for (size_t var : artificial_var_indices_) {
        if (var < big_m_row.size()) {
            big_m_row[var] = Fraction(0);
        }
    }
    simplex_tableau_[big_m_row_index_] = big_m_row;
}

void Simplex::restore_objective_row() {
    std::vector<Fraction> objective_row(simplex_tableau_[0].size(), Fraction(0));
    size_t num_vars = std::min(objective_coeffs_.size(), objective_row.size() - 1);

    for (size_t j = 0; j < num_vars; ++j) {
        objective_row[j] = -objective_coeffs_[j];
    }

    for (size_t i = 0; i < basis_indices_.size(); ++i) {
        size_t var_idx = basis_indices_[i];
        if (var_idx < objective_coeffs_.size()) {
            Fraction coef = objective_coeffs_[var_idx];
            for (size_t j = 0; j < objective_row.size(); ++j) {
                objective_row[j] = objective_row[j] + coef * simplex_tableau_[i][j];
            }
        }
    }

    simplex_tableau_[objective_row_index_] = objective_row;
}

void Simplex::remove_artificial_variables() {
    std::vector<size_t> cols_to_remove;
    for (size_t var : artificial_var_indices_) {
        if (std::find(basis_indices_.begin(), basis_indices_.end(), var) == basis_indices_.end()) {
            cols_to_remove.push_back(var);
        }
    }
    std::sort(cols_to_remove.begin(), cols_to_remove.end(), std::greater<size_t>());
    for (size_t col : cols_to_remove) {
        for (auto& row : simplex_tableau_) {
            row.erase(row.begin() + col);
        }
        for (auto& b : basis_indices_) {
            if (b > col) --b;
        }
        for (auto& av : artificial_var_indices_) {
            if (av > col) --av;
        }
    }
    if (has_big_m_row_ && std::all_of(simplex_tableau_[big_m_row_index_].begin(), simplex_tableau_[big_m_row_index_].end() - 1,
                                  [](const Fraction& x) { return x == Fraction(0); })) {
        std::cout << "Удаляем строку большого M\n\n";
        simplex_tableau_.erase(simplex_tableau_.begin() + big_m_row_index_);
        has_big_m_row_ = false;
        big_m_row_index_ = 0;
        objective_row_index_ = simplex_tableau_.size() - 1;
    }
    artificial_var_indices_.erase(
        std::remove_if(artificial_var_indices_.begin(), artificial_var_indices_.end(),
                       [this](size_t var) { return std::find(basis_indices_.begin(), basis_indices_.end(), var) == basis_indices_.end(); }),
        artificial_var_indices_.end());
}

bool Simplex::is_infeasible_due_to_big_m_row() const {
    if (!has_big_m_row_) return false;
    const auto& big_m_row = simplex_tableau_[big_m_row_index_];
    bool all_non_negative = std::all_of(big_m_row.begin(), big_m_row.end() - 1,
                                        [](const Fraction& x) { return x >= Fraction(0); });
    if (!all_non_negative) return false;
    for (size_t i = 0; i < basis_indices_.size(); ++i) {
        if (std::find(artificial_var_indices_.begin(), artificial_var_indices_.end(), basis_indices_[i]) != artificial_var_indices_.end() &&
            simplex_tableau_[i].back() != Fraction(0)) {
            return true;
        }
    }
    return false;
}

std::vector<Fraction> Simplex::extract_current_solution() const {
    std::vector<Fraction> solution(simplex_tableau_[0].size() - 1, Fraction(0));
    for (size_t i = 0; i < basis_indices_.size(); ++i) {
        size_t var_idx = basis_indices_[i];
        if (var_idx < solution.size()) {
            solution[var_idx] = simplex_tableau_[i].back();
        }
    }
    return std::vector<Fraction>(solution.begin(), solution.begin() + num_original_vars_);
}

std::string Simplex::format_solution_as_string(const std::vector<Fraction>& solution) const {
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

void Simplex::print_current_solution() const {
    std::cout << "Оптимальное решение:\n";
    std::vector<Fraction> solution = extract_current_solution();
    size_t num_vars = simplex_tableau_[0].size() - 1;
    std::vector<Fraction> full_solution(num_vars, Fraction(0));
    for (size_t i = 0; i < basis_indices_.size(); ++i) {
        if (basis_indices_[i] < num_vars) {
            full_solution[basis_indices_[i]] = simplex_tableau_[i].back();
        }
    }
    for (size_t i = 0; i < num_vars; ++i) {
        std::cout << "x" << (i + 1) << " = " << full_solution[i].to_string() << "\n";
    }
    Fraction z_value = Fraction(0);
    for (size_t i = 0; i < original_objective_coeffs_.size() && i < full_solution.size(); ++i) {
        z_value = z_value + original_objective_coeffs_[i] * full_solution[i];
    }
    std::cout << "Z = " << z_value.to_string() << "\n\n";
    std::cout << "Z_" << optimization_goal_ << " = Z" << format_solution_as_string(full_solution) << " = " << z_value.to_string() << "\n";
}

void Simplex::find_alternative_optimal_solutions() {
    //std::cout << "\nПроверка альтернативных оптимальных решений:\n";
    size_t z_row_idx = has_big_m_row_ ? objective_row_index_ : simplex_tableau_.size() - 1;

    std::vector<size_t> non_basis_zero;
    for (size_t j = 0; j < simplex_tableau_[z_row_idx].size() - 1; ++j) {
        if (std::find(basis_indices_.begin(), basis_indices_.end(), j) == basis_indices_.end() && simplex_tableau_[z_row_idx][j] == Fraction(0)) {
            non_basis_zero.push_back(j);
        }
    }

    if (non_basis_zero.empty()) {
        //std::cout << "Альтернативные оптимальные решения отсутствуют.\n";
        return;
    }

    auto original_tableau = simplex_tableau_;
    auto original_basis = basis_indices_;
    std::vector<std::vector<Fraction>> solutions;
    solutions.push_back(extract_current_solution());

    for (size_t col : non_basis_zero) {
        std::vector<std::pair<size_t, Fraction>> ratios;
        for (size_t i = 0; i < simplex_tableau_.size() - 1; ++i) {
            if (simplex_tableau_[i][col] > Fraction(0)) {
                ratios.emplace_back(i, simplex_tableau_[i].back() / simplex_tableau_[i][col]);
            }
        }
        if (!ratios.empty()) {
            auto min_it = std::min_element(ratios.begin(), ratios.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; });
            size_t row = min_it->first;

            execute_phase_two_pivot(row, col);
            std::cout << "Альтернативное оптимальное решение:\n";
            print_current_solution();
            solutions.push_back(extract_current_solution());
            simplex_tableau_ = original_tableau;
            basis_indices_ = original_basis;
        }
    }

    if (solutions.size() >= 2) {
        std::cout << "\nСуществует бесконечно много оптимальных решений.\n";
        std::cout << "Общий вид:\n λ * X₁ + (1-λ) * X₂, где 0 ≤ λ ≤ 1\n";
        std::cout << "X₁ = " << format_solution_as_string(solutions[0]) << "\n";
        std::cout << "X₂ = " << format_solution_as_string(solutions[1]) << "\n\n";

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

        Fraction z_value = Fraction(0);
        for (size_t i = 0; i < original_objective_coeffs_.size() && i < solutions[0].size(); ++i) {
            z_value = z_value + original_objective_coeffs_[i] * solutions[0][i];
        }
        std::cout << "Z = " << z_value.to_string() << "\n";
    }
}

bool Simplex::solve_linear_program() {
    print_simplex_tableau();

    bool has_nonzero_artificial = false;
    for (size_t i = 0; i < basis_indices_.size(); ++i) {
        if (std::find(artificial_var_indices_.begin(), artificial_var_indices_.end(), basis_indices_[i]) != artificial_var_indices_.end() &&
            simplex_tableau_[i].back() != Fraction(0)) {
            has_nonzero_artificial = true;
            break;
        }
    }

    if (has_nonzero_artificial) {
        while (true) {
            update_big_m_row();
            if (has_big_m_row_ &&
                std::all_of(simplex_tableau_[big_m_row_index_].begin(), simplex_tableau_[big_m_row_index_].end() - 1,
                            [](const Fraction& x) { return x == Fraction(0); })) {
                std::cout << "M-строка нулевая, удаляем её.\n";
                remove_artificial_variables();
                restore_objective_row();
                print_simplex_tableau();
                break;
            }
            if (is_infeasible_due_to_big_m_row()) {
                print_simplex_tableau();
                std::cout << "Система ограничений несовместна.\n";
                return false;
            }
            auto pivot_pos = find_pivot_element();
            if (pivot_pos.first == std::numeric_limits<size_t>::max()) {
                print_simplex_tableau();
                std::cout << "Решение не ограничено или несовместно в фазе I.\n";
                return false;
            }
            print_simplex_tableau(&pivot_pos);
            execute_simplex_pivot(pivot_pos.first, pivot_pos.second);
            ++current_iteration_;
        }
    } else {
        if (has_big_m_row_ &&
            std::all_of(simplex_tableau_[big_m_row_index_].begin(), simplex_tableau_[big_m_row_index_].end() - 1,
                        [](const Fraction& x) { return x == Fraction(0); })) {
            std::cout << "M-строка нулевая, удаляем её.\n";
            remove_artificial_variables();
            restore_objective_row();
            print_simplex_tableau();
        }
    }

    for (size_t i = 0; i < basis_indices_.size(); ++i) {
        if (std::find(artificial_var_indices_.begin(), artificial_var_indices_.end(), basis_indices_[i]) != artificial_var_indices_.end() &&
            simplex_tableau_[i].back() != Fraction(0)) {
            print_simplex_tableau();
            std::cout << "Нет допустимого решения (искусственные переменные остались в базисе с ненулевыми значениями).\n";
            return false;
        }
    }

    while (!is_solution_optimal()) {
        auto pivot_pos = find_pivot_element();
        if (pivot_pos.first == std::numeric_limits<size_t>::max()) {
            const auto& z_row = simplex_tableau_[objective_row_index_];
            bool unbounded = false;
            size_t col = std::numeric_limits<size_t>::max();
            for (size_t j = 0; j < z_row.size() - 1; ++j) {
                if (z_row[j] < Fraction(0)) {
                    bool all_non_positive = true;
                    for (size_t i = 0; i < simplex_tableau_.size() - 1; ++i) {
                        if (simplex_tableau_[i][j] > Fraction(0)) {
                            all_non_positive = false;
                            break;
                        }
                    }
                    if (all_non_positive) {
                        unbounded = true;
                        col = j;
                        break;
                    }
                }
            }
            if (unbounded) {
                print_simplex_tableau();
                std::cout << "Решение не ограничено.\n";
                return false;
            }
            break;
        }
        print_simplex_tableau(&pivot_pos);
        execute_phase_two_pivot(pivot_pos.first, pivot_pos.second);
        ++current_iteration_;
    }

    print_simplex_tableau();
    print_current_solution();
    find_alternative_optimal_solutions();
    return true;
}
