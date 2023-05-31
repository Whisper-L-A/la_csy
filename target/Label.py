
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt,QRect
from PyQt5.QtGui import QPainter,QPen

class Label(QLabel):
    # 分别表示矩形框的左上角和右下角坐标
    x0=0
    y0=0
    x1=0
    y1=0
    open_mouse_flag=False  # 鼠标按下标志
    select_roi_flag=False  # 选择矩形框标志
    draw_roi_flag=False  # 绘制矩形框标志
    clear_flag=False  # 清除标志
    rect = QRect()  # 表示矩形框的位置和大小

    #按下鼠标 记录鼠标按下的坐标
    def mousePressEvent(self, event):
        if self.open_mouse_flag is True:
            self.select_roi_flag=True
            self.x0 = event.x()
            self.y0 = event.y()

    #释放鼠标
    def mouseReleaseEvent(self, event):
        self.select_roi_flag = False

    #移动鼠标 记录鼠标移动的坐标，并根据draw_roi_flag判断是否需要更新界面
    def mouseMoveEvent(self, event):
        if self.select_roi_flag is True:
            self.x1=event.x()
            self.y1=event.y()
            if self.draw_roi_flag is True:
                self.update()

    #绘制事件
    def paintEvent(self,event):
        super().paintEvent(event)  # 调用父类的paintevent方法
        painter = QPainter(self)  # 创建一个Qpainter对象
        painter.setPen(QPen(Qt.red, 5, Qt.SolidLine))  #设置画笔颜色和宽度
        if self.clear_flag is True:
            self.x0=0
            self.y0=0
            self.x1=0
            self.y1=0
        # 根据坐标值创建Qrect对象，画出矩形框
        self.rect = QRect(self.x0, self.y0, abs(self.x1 - self.x0), abs(self.y1 - self.y0))
        painter.drawRect(self.rect)
        self.update()





