#include "Simplex.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <iostream>

std::tuple<std::vector<Fraction>, std::vector<std::vector<Fraction>>,
           std::vector<std::string>, std::vector<Fraction>, std::string>
read_input(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Не удалось открыть файл: " + filename);

    std::vector<Fraction> obj_func;
    std::vector<std::vector<Fraction>> constraints;
    std::vector<std::string> signs;
    std::vector<Fraction> rhs;
    std::string goal;

    std::string line;
    if (!std::getline(file, line))
        throw std::runtime_error("Файл пуст");

    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token)
        tokens.push_back(token);

    if (tokens.empty())
        throw std::runtime_error("Отсутствует целевая функция");

    goal = tokens.back();
    if (goal != "max" && goal != "min")
        throw std::runtime_error("Недопустимая цель: " + goal);

    for (size_t i = 0; i < tokens.size() - 1; ++i)
        obj_func.emplace_back(tokens[i]);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<Fraction> coeffs;
        std::string sign;

        while (iss >> token && token != "<=" && token != ">=" && token != "=")
            coeffs.emplace_back(token);

        if (token != "<=" && token != ">=" && token != "=")
            throw std::runtime_error("Недопустимый знак в строке: " + line);

        sign = token;
        if (!(iss >> token))
            throw std::runtime_error("Отсутствует правая часть в строке: " + line);

        Fraction rhs_val(token);
        if (coeffs.size() != obj_func.size())
            throw std::runtime_error("Неправильное количество коэффициентов в строке: " + line);

        constraints.push_back(coeffs);
        signs.push_back(sign);
        rhs.push_back(rhs_val);
    }

    if (constraints.empty())
        throw std::runtime_error("Файл не содержит ограничений");

    return {obj_func, constraints, signs, rhs, goal};
}

int main() {
    try {
        auto [obj_func, constraints, signs, rhs, goal] = read_input("input.txt");
        Simplex solver(obj_func, constraints, signs, rhs, goal);
        if (!solver.solve_linear_program()) {
            std::cerr << "Задача не имеет допустимого решения или решение не ограничено.\n";
            return 2;
        }
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
