#ifndef FIGUREAREA_H
#define FIGUREAREA_H

#include <cmath>

class FigureArea
{
public:
    // Конструктор
    FigureArea();

    // Функции для вычисления площади фигуры
    static double squareArea(double side);
    static double rectangleArea(double width, double height);
    static double parallelogramArea(double base, double height);
    static double rhombusArea(double diagonal_1, double diagonal_2);
    static double triangleArea(double base, double height);
    static double trapezoidArea(double base_1, double base_2, double height);
    static double circleArea(double radius);
    static double sektorArea(double radius, double angle);
};

#endif // FIGUREAREA_H
