# -*- coding: utf-8 -*-
# ui_ai.py  — 现代化重写，替代从 aiui.ui 生成的版本

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QRadioButton, QButtonGroup, QGroupBox, QLineEdit,
    QPushButton, QCheckBox, QSizePolicy, QTextEdit, QDialog,
    QFrame, QScrollArea
)
from PySide6.QtCore import Qt, QSize, QCoreApplication
from PySide6.QtGui import QFont, QPixmap


# ──────────────────────────────────────────────
#  自适应 4:3 视频显示区
# ──────────────────────────────────────────────
class VideoLabel(QLabel):
    """
    类功能简述：自适应 4:3 比例视频显示标签类
    类功能详述：继承自 QLabel，重写 resizeEvent 使控件高度始终锁定为宽度的 3/4，
               保证视频画面以正确的宽高比显示。最小尺寸为 480×360。
               依赖：PySide6.QtWidgets.QLabel
    @author Logic Ye
    @date 2026-03-28
    @version 1.0
    """

    def __init__(self, parent=None):
        """初始化视频标签：设置最小尺寸、居中对齐、深色背景样式及水平伸缩策略。"""
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #1A1A2E;
                color: #6C757D;
                border: 2px solid #2D3748;
                border-radius: 10px;
                font-size: 15px;
            }
        """)
        # 水平方向随布局伸缩，垂直方向由 resizeEvent 根据宽度锁定
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def sizeHint(self):
        return QSize(640, 480)

    def resizeEvent(self, event):
        """每次宽度变化时，把高度强制锁定为 width * 3/4"""
        super().resizeEvent(event)
        w = self.width()
        target_h = max(int(w * 3 / 4), 360)
        if self.height() != target_h:
            self.setFixedHeight(target_h)


# ──────────────────────────────────────────────
#  UI 主类
# ──────────────────────────────────────────────
class Ui_AIModel:
    """
    类功能简述：机器视觉测试平台 UI 构建类
    类功能详述：手工编写的 UI 类，负责构建整个对话框界面，
               包括标题栏、模型选择栏、左侧视频区和右侧信息区。
               通过 setupUi() 方法被 AIWindow（QDialog）调用。
               依赖：PySide6.QtWidgets、PySide6.QtCore、PySide6.QtGui
    @author Logic Ye
    @date 2026-03-28
    @version 1.0
    """

    # 模型结构图路径（可在 assets 目录放置对应图片）
    MODEL_IMAGES = {
        0: "assets/images/model_depth.png",
        1: "assets/images/model_detect.png",
        2: "assets/images/model_seg.png",
        3: "assets/images/model_pose.png",
        4: "assets/images/model_bg.png",
    }

    def setupUi(self, AIModel):
        """
        构建并初始化整个 UI 布局：设置窗口标题和尺寸，创建标题栏、模型选择栏，
        添加左右分割面板，并应用全局样式表。
        @param AIModel  父 QDialog 实例，UI 控件将挂载在其上
        """
        AIModel.setWindowTitle("机器视觉测试")
        AIModel.resize(1380, 860)
        AIModel.setMinimumSize(1100, 700)

        # QDialog 直接用自身作为根，不需要 centralwidget
        self.centralwidget = AIModel
        root = QVBoxLayout(AIModel)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 整体容器 ──
        container = QWidget()
        container.setObjectName("container")
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(20, 16, 20, 16)
        main_layout.setSpacing(12)
        root.addWidget(container)

        # ══════════════ 标题栏 ══════════════
        title_bar = QHBoxLayout()
        self.lbl_title = QLabel("机器视觉测试平台")
        self.lbl_title.setFont(QFont("Microsoft YaHei", 22, QFont.Weight.Bold))
        self.lbl_title.setStyleSheet("color: #1A1A2E;")
        title_bar.addWidget(self.lbl_title)
        title_bar.addStretch()
        main_layout.addLayout(title_bar)

        # ══════════════ 模型选择栏 ══════════════
        self.model_groupbox = QGroupBox()
        self.model_groupbox.setObjectName("model_groupbox")
        model_row = QHBoxLayout(self.model_groupbox)
        model_row.setSpacing(6)
        model_row.setContentsMargins(20, 8, 20, 8)

        self.buttonGroup = QButtonGroup(AIModel)
        self.buttonGroup.setObjectName("buttonGroup")

        model_defs = [
            ("rad_depth",  "深度检测"),
            ("rad_detect", "目标检测"),
            ("rad_seg",    "目标分割"),
            ("rad_pose",   "姿势检测"),
            ("rad_bg",     "背景替换"),
        ]

        for idx, (name, text) in enumerate(model_defs):
            rb = QRadioButton(text)
            rb.setObjectName(name)
            rb.setStyleSheet("""
                QRadioButton {
                    font-size: 14px;
                    padding: 6px 14px;
                    border-radius: 16px;
                }
                QRadioButton::indicator { width: 0; height: 0; }
                QRadioButton:checked {
                    background-color: #4361EE;
                    color: white;
                }
                QRadioButton:!checked {
                    background-color: #EEF2FF;
                    color: #4361EE;
                }
                QRadioButton:hover:!checked { background-color: #DAE2FF; }
            """)
            self.buttonGroup.addButton(rb, idx)
            setattr(self, name, rb)
            model_row.addWidget(rb)
            if idx == 0:
                rb.setChecked(True)

        model_row.addStretch()
        main_layout.addWidget(self.model_groupbox)

        # ══════════════ 主分割区 ══════════════
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(6)
        self.splitter.setStyleSheet("QSplitter::handle { background-color: #E2E8F0; }")
        main_layout.addWidget(self.splitter, stretch=1)

        self.splitter.addWidget(self._build_left(AIModel))
        self.splitter.addWidget(self._build_right())
        self.splitter.setSizes([860, 440])

        self._apply_style(AIModel)

    # ──────────────────────────────────────────────
    #  左侧面板
    # ──────────────────────────────────────────────
    def _build_left(self, AIModel):
        """
        构建左侧面板：包含视频输入源选择区（摄像头/视频文件单选）、
        视频文件路径行（默认隐藏）、视频帧显示区（VideoLabel）及开始/停止控制按钮。
        @param AIModel  父 QDialog，用于挂载独立的视频源 ButtonGroup
        @return 配置完成的左侧 QWidget 面板
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(0, 0, 6, 0)

        # ── 输入源选择 ──
        src_box = QGroupBox("视频输入源")
        src_box.setObjectName("src_box")
        src_row = QHBoxLayout(src_box)
        src_row.setSpacing(20)
        src_row.setContentsMargins(16, 6, 16, 6)

        self.rad_cam   = QRadioButton("摄像头")
        self.rad_video = QRadioButton("视频文件")
        self.rad_cam.setObjectName("rad_cam")
        self.rad_video.setObjectName("rad_video")
        self.rad_cam.setChecked(True)

        # 用独立 ButtonGroup，避免与模型选择冲突
        self.source_group = QButtonGroup(AIModel)
        self.source_group.setObjectName("source_group")
        self.source_group.addButton(self.rad_cam,   0)
        self.source_group.addButton(self.rad_video, 1)

        src_row.addWidget(self.rad_cam)
        src_row.addWidget(self.rad_video)
        src_row.addStretch()

        # ── 视频文件路径行（默认隐藏） ──
        self.file_row = QWidget()
        file_h = QHBoxLayout(self.file_row)
        file_h.setContentsMargins(0, 0, 0, 0)
        file_h.setSpacing(8)

        self.video_input = QLineEdit()
        self.video_input.setObjectName("video_input")
        self.video_input.setPlaceholderText("视频文件路径...")
        self.video_input.setMinimumHeight(38)
        self.video_input.setReadOnly(True)

        self.btn_select = QPushButton("选取视频")
        self.btn_select.setObjectName("btn_select")
        self.btn_select.setMinimumHeight(38)
        self.btn_select.setMinimumWidth(100)

        file_h.addWidget(self.video_input, stretch=1)
        file_h.addWidget(self.btn_select)
        self.file_row.setVisible(False)

        layout.addWidget(src_box)
        layout.addWidget(self.file_row)

        # ── 视频显示区 ──
        self.lbl_video = VideoLabel()
        self.lbl_video.setObjectName("lbl_video")
        self.lbl_video.setText("视频 / 图像显示区域")
        layout.addWidget(self.lbl_video, stretch=1)

        # ── 控制按钮行 ──
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)

        self.btn_start = QPushButton("▶  开始")
        self.btn_stop  = QPushButton("■  停止")
        self.btn_start.setObjectName("btn_start")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_start.setMinimumHeight(40)
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.setEnabled(False)

        ctrl_row.addWidget(self.btn_start)
        ctrl_row.addWidget(self.btn_stop)
        layout.addLayout(ctrl_row)

        return panel

    # ──────────────────────────────────────────────
    #  右侧面板
    # ──────────────────────────────────────────────
    def _build_right(self):
        """
        构建右侧面板：包含视频信息组（目标数、帧率、时长、置信度、当前模型）、
        模型结构图预览区（支持图片加载）以及推理日志文本框。
        @return 配置完成的右侧 QWidget 面板
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)
        layout.setContentsMargins(6, 0, 0, 0)

        # ── 视频信息 ──
        self.video_group = QGroupBox("视频信息")
        self.video_group.setObjectName("video_group")
        vg = QVBoxLayout(self.video_group)
        vg.setSpacing(8)
        vg.setContentsMargins(14, 10, 14, 10)

        info_defs = [
            ("lbl_objects", "检测目标数：—"),
            ("lbl_fps",     "推理帧率：—"),
            ("lbl_time",    "已运行时长：00:00"),
            ("lbl_conf",    "平均置信度：—"),
            ("lbl_model",   "当前模型：深度检测"),
        ]
        for obj_name, text in info_defs:
            lbl = QLabel(text)
            lbl.setObjectName(obj_name)
            lbl.setStyleSheet("font-size: 13px; color: #2D3748;")
            setattr(self, obj_name, lbl)
            vg.addWidget(lbl)

        layout.addWidget(self.video_group)

        # ── 模型结构图 ──
        self.model_info_group = QGroupBox("模型结构")
        self.model_info_group.setObjectName("model_info_group")
        mi = QVBoxLayout(self.model_info_group)
        mi.setContentsMargins(8, 8, 8, 8)

        self.model_preview = QLabel()
        self.model_preview.setObjectName("model_preview")
        self.model_preview.setMinimumSize(300, 220)
        self.model_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_preview.setStyleSheet("""
            QLabel {
                background-color: #F8FAFC;
                border: 1px dashed #CBD5E0;
                border-radius: 8px;
                color: #A0AEC0;
                font-size: 13px;
            }
        """)
        self.model_preview.setText("模型结构图")
        self.model_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        mi.addWidget(self.model_preview)
        layout.addWidget(self.model_info_group, stretch=1)

        # ── 推理信息 ──
        self.edt_info = QTextEdit()
        self.edt_info.setObjectName("edt_info")
        self.edt_info.setReadOnly(True)
        self.edt_info.setMaximumHeight(130)
        self.edt_info.setPlaceholderText("推理日志...")
        self.edt_info.setStyleSheet("""
            QTextEdit {
                background-color: #0F172A;
                color: #94A3B8;
                font-family: Consolas, monospace;
                font-size: 12px;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        layout.addWidget(self.edt_info)

        return panel

    # ──────────────────────────────────────────────
    #  全局样式
    # ──────────────────────────────────────────────
    def _apply_style(self, root):
        """为根窗口设置全局 QSS 样式表，统一定义 QGroupBox、QLineEdit、QPushButton 等控件外观。"""
        root.setStyleSheet("""
            QWidget#container, QDialog, QMainWindow {
                background-color: #F0F4F8;
                font-family: "Microsoft YaHei", Arial;
            }
            QGroupBox {
                background-color: white;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                margin-top: 16px;
                padding: 10px 4px 8px 4px;
                font-size: 13px;
                font-weight: bold;
                color: #2D3748;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
            }
            QGroupBox#model_groupbox {
                margin-top: 0px;
                padding: 6px 4px;
            }
            QGroupBox#model_groupbox::title { color: transparent; }
            QLineEdit {
                padding: 8px 12px;
                border: 1px solid #CBD5E0;
                border-radius: 6px;
                background-color: white;
                font-size: 13px;
            }
            QLineEdit:focus { border-color: #4361EE; }
            QPushButton {
                background-color: #4361EE;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 14px;
                font-weight: 500;
                padding: 0 16px;
            }
            QPushButton:hover { background-color: #3451D1; }
            QPushButton:pressed { background-color: #2C44B8; }
            QPushButton:disabled { background-color: #A0AEC0; }
            QPushButton#btn_stop {
                background-color: #E53E3E;
            }
            QPushButton#btn_stop:hover { background-color: #C53030; }
            QPushButton#btn_stop:disabled { background-color: #A0AEC0; }
            QRadioButton { font-size: 14px; }
            QSplitter::handle { background-color: #E2E8F0; }
        """)

