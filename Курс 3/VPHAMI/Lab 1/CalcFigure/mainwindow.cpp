#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "figurearea.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    images = {
        ":/image/images/Square.png",
        ":/image/images/Rectangle.png",
        ":/image/images/Parallelogram.png",
        ":/image/images/Rhombus.png",
        ":/image/images/Triangle.png",
        ":/image/images/Trapezoid.png",
        ":/image/images/Circle.png",
        ":/image/images/Sektor.png"
    };

    on_comboBox_activated(ui->comboBox->currentIndex());
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_menu_3_triggered()
{
    QApplication::quit();
}

// Функция установки изображения по индексу выбранной фигуры
void MainWindow::on_comboBox_activated(int index)
{
    if (index >= 0 ) {
        ui->labelImage->setStyleSheet("image : url(" + images[index] + ")");
    }
}

void MainWindow::on_pushButton_clicked()
{

}

