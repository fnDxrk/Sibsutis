#ifndef FIGUREAREA_H
#define FIGUREAREA_H

#include <cmath>

class FigureArea
{
private:
    // Функции для вычисления площади фигуры
    double squareArea(double side);
    double rectangleArea(double width, double height);
    double parallelogramArea(double base, double height);
    double rhombusArea(double diagonal_1, double diagonal_2);
    double triangleArea(double base, double height);
    double trapezoidArea(double base_1, double base_2, double height);
    double circleArea(double radius);
    double sektorArea(double radius, double angle);

public:
    // Конструктор
    FigureArea();
};

#endif // FIGUREAREA_H
