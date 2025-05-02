#include "Simplex.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

std::tuple<std::vector<Fraction>, std::vector<std::vector<Fraction>>,
           std::vector<std::string>, std::vector<Fraction>, std::string>
read_input(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл: " + filename);
    }

    std::vector<Fraction> obj_func;
    std::vector<std::vector<Fraction>> constraints;
    std::vector<std::string> signs;
    std::vector<Fraction> rhs;
    std::string goal;

    std::string line;
    if (!std::getline(file, line)) {
        file.close();
        throw std::runtime_error("Файл пуст");
    }
    goal = line;
    if (goal != "max" && goal != "min") {
        file.close();
        throw std::runtime_error("Недопустимая цель: должна быть 'max' или 'min'");
    }

    if (!std::getline(file, line)) {
        file.close();
        throw std::runtime_error("Отсутствует целевая функция");
    }
    std::istringstream iss(line);
    std::string token;
    while (iss >> token) {
        try {
            obj_func.emplace_back(token);
        } catch (const std::exception& e) {
            file.close();
            throw std::runtime_error("Ошибка парсинга коэффициента: " + token);
        }
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<Fraction> coeffs;
        std::string sign, rhs_str;

        while (iss >> token && token != "<=" && token != ">=" && token != "=") {
            try {
                coeffs.emplace_back(token);
            } catch (const std::exception& e) {
                file.close();
                throw std::runtime_error("Ошибка парсинга коэффициента: " + token);
            }
        }
        if (token != "<=" && token != ">=" && token != "=") {
            file.close();
            throw std::runtime_error("Недопустимый знак: " + line);
        }
        sign = token;
        if (!(iss >> rhs_str)) {
            file.close();
            throw std::runtime_error("Отсутствует правая часть: " + line);
        }
        Fraction rhs_val;
        try {
            rhs_val = Fraction(rhs_str);
        } catch (const std::exception& e) {
            file.close();
            throw std::runtime_error("Ошибка парсинга правой части: " + rhs_str);
        }

        if (coeffs.size() != obj_func.size()) {
            file.close();
            throw std::runtime_error("Количество коэффициентов не соответствует");
        }

        constraints.push_back(coeffs);
        signs.push_back(sign);
        rhs.push_back(rhs_val);
    }

    file.close();
    if (constraints.empty()) {
        throw std::runtime_error("Файл не содержит ограничений");
    }

    return {obj_func, constraints, signs, rhs, goal};
}

int main() {
    try {
        auto [obj_func, constraints, signs, rhs, goal] = read_input("input.txt");
        SimplexBigM solver(obj_func, constraints, signs, rhs, goal);
        solver.solve();
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}