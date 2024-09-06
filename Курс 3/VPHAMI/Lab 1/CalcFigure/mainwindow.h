#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

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

private:
    Ui::MainWindow *ui;
    // Массив изображений фигур
    QStringList images;
};
#endif // MAINWINDOW_H
