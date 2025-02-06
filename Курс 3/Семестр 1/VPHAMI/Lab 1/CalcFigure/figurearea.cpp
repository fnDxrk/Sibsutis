#include "figurearea.h"

// Конструктор
FigureArea::FigureArea() {}

// Площадь квадрата
double FigureArea::squareArea(double side) {
    return side * side;
}

// Площадь прямоугольника
double FigureArea::rectangleArea(double width, double height) {
    return width * height;
}

// Площадь параллелограма
double FigureArea::parallelogramArea(double base, double height) {
    return base * height;
}

// Площадь ромба
double FigureArea::rhombusArea(double diagonal_1, double diagonal_2) {
    return (diagonal_1 * diagonal_2) / 2;
}

// Площадь треугольника
double FigureArea::triangleArea(double base, double height) {
    return (base * height) / 2;
}

// Площадь трапеции
double FigureArea::trapezoidArea(double base_1, double base_2, double height) {
    return ((base_1 + base_2) / 2) * height;
}

// Площадь круга
double FigureArea::circleArea(double radius) {
    return M_PI * radius * radius;
}

// Площадь сектора
double FigureArea::sektorArea(double radius, double angle) {
    return (M_PI * radius * radius * angle) / 360;
}

