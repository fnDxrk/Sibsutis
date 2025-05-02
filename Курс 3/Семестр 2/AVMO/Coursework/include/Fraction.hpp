#ifndef FRACTION_HPP
#define FRACTION_HPP

#include <string>
#include <stdexcept>

class Fraction {
private:
    int numerator_;
    int denominator_;
    void simplify();
    static int gcd(int a, int b);

public:
    Fraction(int numerator = 0, int denominator = 1);
    Fraction(const std::string& str);
    Fraction(const Fraction& other) = default;
    Fraction& operator=(const Fraction& other) = default;
    ~Fraction() = default;

    Fraction operator+(const Fraction& other) const;
    Fraction operator-(const Fraction& other) const;
    Fraction operator*(const Fraction& other) const;
    Fraction operator/(const Fraction& other) const;
    Fraction operator-() const;
    bool operator==(const Fraction& other) const;
    bool operator<(const Fraction& other) const;
    bool operator<=(const Fraction& other) const;
    bool operator>(const Fraction& other) const;
    bool operator>=(const Fraction& other) const;
    bool operator!=(const Fraction& other) const; // Новый оператор
    Fraction abs() const;
    std::string to_string() const;

    friend std::ostream& operator<<(std::ostream& os, const Fraction& fraction);
};

#endif // FRACTION_HPP