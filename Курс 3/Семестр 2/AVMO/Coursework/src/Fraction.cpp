#include "Fraction.hpp"
#include <sstream>
#include <cmath>

int Fraction::gcd(int a, int b) {
    return b == 0 ? std::abs(a) : gcd(b, a % b);
}

void Fraction::simplify() {
    if (denominator_ == 0) {
        throw std::invalid_argument("Знаменатель не может быть равен нулю");
    }
    if (numerator_ == 0) {
        denominator_ = 1;
        return;
    }
    int g = gcd(numerator_, denominator_);
    numerator_ = numerator_ / g;
    denominator_ = denominator_ / g;
    if (denominator_ < 0) {
        numerator_ = -numerator_;
        denominator_ = -denominator_;
    }
}

Fraction::Fraction(int numerator, int denominator)
    : numerator_(numerator), denominator_(denominator) {
    simplify();
}

Fraction::Fraction(const std::string& str) {
    std::string num_str, den_str;
    size_t slash_pos = str.find('/');
    if (slash_pos == std::string::npos) {
        num_str = str;
        denominator_ = 1;
    } else {
        num_str = str.substr(0, slash_pos);
        den_str = str.substr(slash_pos + 1);
        if (den_str.empty()) {
            throw std::invalid_argument("Некорректный формат дроби: " + str);
        }
        denominator_ = std::stoi(den_str);
    }
    numerator_ = std::stoi(num_str);
    simplify();
}

Fraction Fraction::operator+(const Fraction& other) const {
    int new_num = numerator_ * other.denominator_ + other.numerator_ * denominator_;
    int new_den = denominator_ * other.denominator_;
    return Fraction(new_num, new_den);
}

Fraction Fraction::operator-(const Fraction& other) const {
    int new_num = numerator_ * other.denominator_ - other.numerator_ * denominator_;
    int new_den = denominator_ * other.denominator_;
    return Fraction(new_num, new_den);
}

Fraction Fraction::operator*(const Fraction& other) const {
    int new_num = numerator_ * other.numerator_;
    int new_den = denominator_ * other.denominator_;
    return Fraction(new_num, new_den);
}

Fraction Fraction::operator/(const Fraction& other) const {
    if (other.numerator_ == 0) {
        throw std::invalid_argument("Деление на ноль");
    }
    int new_num = numerator_ * other.denominator_;
    int new_den = denominator_ * other.numerator_;
    return Fraction(new_num, new_den);
}

Fraction Fraction::operator-() const {
    return Fraction(-numerator_, denominator_);
}

bool Fraction::operator==(const Fraction& other) const {
    return numerator_ == other.numerator_ && denominator_ == other.denominator_;
}

bool Fraction::operator<(const Fraction& other) const {
    return numerator_ * other.denominator_ < other.numerator_ * denominator_;
}

bool Fraction::operator<=(const Fraction& other) const {
    return *this < other || *this == other;
}

bool Fraction::operator>(const Fraction& other) const {
    return !(*this <= other);
}

bool Fraction::operator>=(const Fraction& other) const {
    return !(*this < other);
}

bool Fraction::operator!=(const Fraction& other) const {
    return !(*this == other);
}

Fraction Fraction::abs() const {
    return Fraction(std::abs(numerator_), denominator_);
}

std::string Fraction::to_string() const {
    if (denominator_ == 1) {
        return std::to_string(numerator_);
    }
    return std::to_string(numerator_) + "/" + std::to_string(denominator_);
}

std::ostream& operator<<(std::ostream& os, const Fraction& fraction) {
    os << fraction.to_string();
    return os;
}