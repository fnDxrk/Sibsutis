#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "figurearea.h"

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    void setImages(const QStringList &newImages);

private slots:
    void on_menu_3_triggered();

    void on_comboBox_activated(int index);

    void on_pushButton_clicked();

private:
    Ui::MainWindow *ui;


    // Массив изображений фигур
    QStringList images;
    // Функция для опредения видимости интерфейса
    void argumentUI(const QString &text1, bool show2, const QString &text2, bool show3, const QString &text3);
    // Функция для обновления интерфейса
    void updateUI(int index);
    // Функция для выбора изображения при изменении индекса
    void setImage(int index);
};
#endif // MAINWINDOW_H
