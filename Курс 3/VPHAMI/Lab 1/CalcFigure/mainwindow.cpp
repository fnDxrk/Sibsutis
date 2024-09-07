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

void MainWindow::argumentUI(const QString &text1, bool show2 = false, const QString &text2 = "", bool show3 = false, const QString &text3 = "") {
    ui->text_1->setText(text1);

    if (show2) {
        ui->text_2->setText(text2);
        ui->text_2->show();
        ui->line_2->show();
    } else {
        ui->text_2->hide();
        ui->line_2->hide();
    }

    if (show3) {
        ui->text_3->setText(text3);
        ui->text_3->show();
        ui->line_3->show();
    } else {
        ui->text_3->hide();
        ui->line_3->hide();
    }
}

void MainWindow::updateUI(int index)
{
    switch (index) {
    case 0:
        argumentUI("Длина");
        break;
    case 1:
        argumentUI("Длина", true, "Высота");
        break;
    case 2:
        argumentUI("Основание", true, "Высота");
        break;
    case 3:
        argumentUI("Диагональ 1", true, "Диагональ 2");
        break;
    case 4:
        argumentUI("Основание", true, "Высота");
        break;
    case 5:
        argumentUI("Основание 1", true, "Основание 2", true, "Высота");
        break;
    case 6:
        argumentUI("Радиус");
        break;
    case 7:
        argumentUI("Радиус", true, "Угол");
        break;
    }
}

void MainWindow::setImage(int index)
{
    if (index >= 0 ) {
        ui->labelImage->setStyleSheet("image : url(" + images[index] + ")");
    }
}

void MainWindow::on_comboBox_activated(int index)
{
    updateUI(index);
    setImage(index);
}

void MainWindow::on_pushButton_clicked()
{

}

