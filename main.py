"""
coding: utf-8
Author: 12003990410 陈思宇
Time:   2023/5/20
"""

from Pysot_Window import Pysot_Window
import sys, os
from PyQt5.QtWidgets import QApplication


if __name__ == '__main__':
    app = QApplication(sys.argv)
    pysot_window = Pysot_Window()
    pysot_window.show()
    sys.exit(app.exec_())