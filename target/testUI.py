import sys
from PyQt5.QtWidgets import QMainWindow,QApplication
from single_object_target import Ui_single_object_target


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = Ui_single_object_target()
    ui.setupUi(window)
    window.show()
    sys.exit(app.exec_())
