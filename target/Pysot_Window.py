
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import torch
import numpy as np
from design2.pysot.core.config import cfg
from design2.pysot.models.model_builder import ModelBuilder
from design2.pysot.tracker.tracker_builder import build_tracker

from PyQt5.QtWidgets import QMainWindow, QFileDialog
from single_object_target import Ui_single_object_target
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
import cv2


class Pysot_Window(QMainWindow, Ui_single_object_target):

    def __init__(self):
        super(Pysot_Window, self).__init__()
        self.setupUi(self)
        self.init_ui()
        self.init_slots()
        self.load_yaml_pth()
        self.init_track()
        self.tb_show_message.append('程序初始化完毕!')

    def init_ui(self):
        self.width = self.label_show.width()
        self.height = self.label_show.height()
        self.keyboard_flag = False  # 是否打开键盘标志
        self.camera_flag = False    # 是否打开摄像头标志
        self.video_flag = False     # 是否打开视频文件标志
        self.track_flag = False     # 是否确认跟踪
        self.target_rect = []
        self.video_path = ''
        self.FPS = 60

        # 按键的初始化
        # 前期只需要摄像头按钮和视频的按钮
        self.btn_camera.setEnabled(True)
        self.btn_video.setEnabled(True)
        self.btn_select.setEnabled(False)
        self.btn_end.setEnabled(False)
        self.btn_begin.setEnabled(False)

    # 槽函数初始化
    def init_slots(self):
        self.btn_camera.clicked.connect(self.press_camera)
        self.btn_video.clicked.connect(self.press_video)
        self.btn_select.clicked.connect(self.press_select)
        self.btn_end.clicked.connect(self.press_end)
        self.btn_begin.clicked.connect(self.press_begin)

    # 导入模型文件
    def load_yaml_pth(self):
        self.config_path = '../models/siamrpn_alex_dwxcorr_otb/config.yaml'
        self.snapshot_path = '../models/siamrpn_alex_dwxcorr_otb/model.pth'
    # 初始化跟踪模型
    def init_track(self):
        cfg.merge_from_file(self.config_path)  # 从训练的模型中加载配置信息并将其合并到当前配置中
        cfg.CUDA = torch.cuda.is_available() and cfg.CUDA  # 检查当前系统是否支持CUDA，并将结果存储在变量中
        device = torch.device('cuda' if cfg.CUDA else 'cpu')  # 根据当前系统是否支持CUDA来选择使用CPU还是GPU进行训练和跟踪
        self.tb_show_message.append('模型对象创建...')
        # 从指定pysot的快照文件中加载模型参数并将其存储在变量中
        self.checkpoint = torch.load(self.snapshot_path, map_location=lambda storage, loc: storage.cpu())
        self.model = ModelBuilder()  # 创建ModelBuilder对象用于构建跟踪模型

        self.model.load_state_dict(self.checkpoint)        #  将从快照文件中加载的模型参数加载到跟踪模型
        self.model.eval().to(device)  # 将跟踪模型切换到评估模式，并将其移动到指定的设备
        self.tb_show_message.append('跟踪模型加载完成!')
        # 创建跟踪器
        self.tracker = build_tracker(self.model)

    # 按下打开摄像头的按钮
    def press_camera(self):
        self.tb_show_message.append('准备打开摄像头...')

        self.camera = cv2.VideoCapture(0)
        if self.camera.isOpened:
            self.tb_show_message.append('摄像头打开成功')
        # 摄像头标志位转变
        self.camera_flag = True

        # 只剩下选择跟踪和结束跟踪
        self.btn_camera.setEnabled(False)
        self.btn_video.setEnabled(False)
        self.btn_select.setEnabled(True)
        self.btn_end.setEnabled(True)
        self.btn_begin.setEnabled(False)
        self.label_show.clear_flag = False

        self.camera_timer = QTimer(self)
        self.tb_show_message.append('摄像头线程已创建...')
        self.camera_timer.timeout.connect(self.camera_show)
        self.tb_show_message.append('摄像头拍摄中...')
        self.camera_timer.start(self.FPS)

    # 摄像头视频展示在label上
    def camera_show(self):
        if self.camera_flag is True:
            if self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret is True:
                    self.frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
                    self.label_show.setPixmap(QPixmap.fromImage(img))
            else:
                self.camera_timer.stop()
                self.camera.release()
                self.tb_show_message.append('exceptions!')
        else:
            self.camera_timer.stop()
            self.camera.release()
            self.tb_show_message.append('exceptions!')

    # 按下打开视频的按钮
    def press_video(self):
        video_path = QFileDialog.getOpenFileName(self,
                                                 '选择你要打开的视频文件',
                                                 os.path.dirname(__file__),
                                                 '*.mp4 *.avi')
        if video_path[0] == '':
            return
        self.video_path = video_path[0]
        self.tb_show_message.append('视频文件路径: ' + self.video_path)

        self.label_show.clear_flag = False
        # 视频标志位转变
        self.video_flag = True

        # 剩下选择跟踪和结束跟踪
        self.btn_camera.setEnabled(False)
        self.btn_video.setEnabled(False)
        self.btn_select.setEnabled(True)
        self.btn_end.setEnabled(True)
        self.btn_begin.setEnabled(False)

        self.capture_video = cv2.VideoCapture(self.video_path)
        self.video_timer = QTimer(self)
        self.tb_show_message.append('视频线程已创建...')
        self.video_timer.timeout.connect(self.video_show)
        self.tb_show_message.append('视频播放中...')
        self.video_timer.start(self.FPS)

    # 视频文件展示在label上面
    def video_show(self):
        if self.video_flag is True:
            if self.capture_video.isOpened() is True:
                ret, frame = self.capture_video.read()
                if ret is True:
                    self.frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
                    img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
                    self.label_show.setPixmap(QPixmap.fromImage(img))
            else:
                self.video_timer.stop()
                self.capture_video.release()
                self.tb_show_message.append('exceptions!')
        else:
            self.video_timer.stop()
            self.capture_video.release()
            self.tb_show_message.append('exceptions!')

    # 按下选择跟踪目标按钮
    def press_select(self):
        # 打开键盘的标志位转变
        self.keyboard_flag = True

        # 只剩下目标选择和目标选择结束
        self.btn_camera.setEnabled(False)
        self.btn_video.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.btn_end.setEnabled(True)
        self.btn_begin.setEnabled(True)

        # 如果打开的是摄像头
        if self.camera_flag is True:
            self.first_frame = self.frame
            self.camera_timer.stop()

        # 如果打开的是视频
        if self.video_flag is True:
            self.first_frame = self.frame
            self.video_timer.stop()

        self.tb_show_message.append('请选择您的跟踪目标: 按下b键选择，e键确认')

    # 按下开始跟踪按钮
    def press_begin(self):
        if self.keyboard_flag is False:
            # 按钮只剩下结束的
            self.btn_camera.setEnabled(False)
            self.btn_video.setEnabled(False)
            self.btn_select.setEnabled(False)
            self.btn_end.setEnabled(True)
            self.btn_begin.setEnabled(False)

            self.target_rect = self.transformrect(self.label_show.rect)
            self.tb_show_message.append('目标框格式转码中...')
            init_rect = tuple(self.target_rect)
            self.tracker.init(self.first_frame, init_rect)
            self.tb_show_message.append('目标框初始化完毕!')
            self.clear_label()
            self.tb_show_message.append('跟踪界面清除...')

            # 创建跟踪线程
            self.tracker_timer = QTimer(self)
            self.tb_show_message.append('跟踪线程已创建!')
            self.tracker_timer.timeout.connect(self.track_process)
            self.tracker_timer.start(self.FPS)
            self.open_select_roi = False
            self.tb_show_message.append('目标已经选定！')

    def track_process(self):
        # 改变线程的跟踪标记位
        self.track_flag = True
        # 如果是视频跟踪
        if self.camera_flag is True:
            if self.camera.isOpened() is True:
                ret, frame = self.camera.read()
                if ret is True:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
                    outputs = self.tracker.track(frame)
                    if 'polygon' in outputs:
                        polygon = np.array(outputs['polygon']).astype(np.int32)
                        cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                      True, (0, 255, 0), 3)
                        mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                        mask = mask.astype(np.uint8)
                        mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                        frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                    else:
                        bbox = list(map(int, outputs['bbox']))
                        cv2.rectangle(frame, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                      (0, 255, 0), 3)
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
                    self.label_show.setPixmap(QPixmap.fromImage(img))
        # 如果是摄像头跟踪
        elif self.video_flag is True:
            if self.capture_video.isOpened() is True:
                ret, frame = self.capture_video.read()
                if ret is True:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
                    outputs = self.tracker.track(frame)
                    if 'polygon' in outputs:
                        polygon = np.array(outputs['polygon']).astype(np.int32)
                        cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                      True, (0, 255, 0), 3)
                        mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                        mask = mask.astype(np.uint8)
                        mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                        frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
                    else:
                        bbox = list(map(int, outputs['bbox']))
                        cv2.rectangle(frame, (bbox[0], bbox[1]),
                                      (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                                      (0, 255, 0), 3)
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
                    self.label_show.setPixmap(QPixmap.fromImage(img))
            else:
                self.tracker_timer.stop()
                self.camera.release()
                self.tb_show_message.append('exceptions！')



    def press_end(self):
        # 摄像头的
        if self.camera_flag is True:
            self.camera_timer.stop()
            self.camera.release()
            # 清理label
            self.clear_label()
            # 重新打开选择的两个按钮
            self.keyboard_flag = False
            self.btn_camera.setEnabled(True)
            self.btn_video.setEnabled(True)
            self.btn_select.setEnabled(False)
            self.btn_begin.setEnabled(False)
            self.tb_show_message.append('释放完毕！')
            if self.track_flag is True:
                self.tracker_timer.stop()
                self.camera.release()
                self.track_flag = False
            self.camera_flag = False
        # 视频的
        if self.video_flag is True:
            self.video_timer.stop()
            self.capture_video.release()
            # 清理label
            self.clear_label()
            # 重新打开选择的两个按钮
            self.keyboard_flag = False
            self.btn_camera.setEnabled(True)
            self.btn_video.setEnabled(True)
            self.btn_select.setEnabled(False)
            self.btn_begin.setEnabled(False)
            self.tb_show_message.append('释放完毕！')
            if self.track_flag is True:
                self.tracker_timer.stop()
                self.capture_video.release()
                self.track_flag = False
            self.video_flag = False
        self.tb_show_message.append('跟踪线程已结束!清理完毕！')

    def clear_label(self):
        self.label_show.clear_flag = True
        self.label_show.clear()

    def transformrect(self, rectangle):
        ts_rect = [rectangle.x(), rectangle.y(),
                  rectangle.width(), rectangle.height()]
        return ts_rect

    def keyPressEvent(self, QKeyEvent):
        if self.keyboard_flag == True:
            if QKeyEvent.key() == Qt.Key_B:
                self.label_show.setCursor(Qt.CrossCursor)
                self.label_show.open_mouse_flag = True
                self.label_show.draw_roi_flag = True
            if QKeyEvent.key() == Qt.Key_E:
                self.label_show.unsetCursor()
                self.label_show.draw_roi_flag = False
                self.label_show.open_mouse_flag = False
                self.keyboard_flag = False