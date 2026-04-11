#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主窗口 - 语音指令采集工具
负责连接核心逻辑和 UI 组件
"""

import os
from datetime import datetime
from typing import Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QModelIndex, QTimer, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from config import (
    BUTTON_RECORDING_TEXT,
    BUTTON_RECORD_TEXT,
    DEFAULT_BASE_DIR,
    DEFAULT_LABELS,
    EXPORT_FILE_FILTER,
    EXPORT_FILENAME_PREFIX,
    MAX_RECORDING_DURATION,
    MIN_RECORDING_DURATION,
    PATH_LABEL_STYLE,
    RANDOM_SEED,
    RECORD_BUTTON_STYLE,
    STATUS_IDLE_STYLE,
    STATUS_IDLE_TEXT,
    STATUS_RECORDING_STYLE,
    STATUS_RECORDING_TEXT,
    SPLITTER_LEFT_SIZE,
    SPLITTER_RIGHT_SIZE,
    TREE_INDENTATION,
    VOLUME_DECAY_RATE,
    VOLUME_UPDATE_INTERVAL_MS,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
    WINDOW_TITLE,
    MAIN_WINDOW_STYLE,
)
from core import AudioPlayer, AudioRecorderThread, DatasetManager
from ui.dialogs.label_manager import LabelManageDialog
from widgets.audio_file_model import AudioFileSystemModel


class VoiceCommandCollector(QMainWindow):
    """语音指令采集工具主窗口"""

    def __init__(self) -> None:
        super().__init__()
        self.recorder_thread: Optional[AudioRecorderThread] = None
        self.audio_player: Optional[AudioPlayer] = None
        self.current_volume: float = 0.0

        # 数据集管理器
        self.base_dir = DEFAULT_BASE_DIR
        self.dataset_manager = DatasetManager(self.base_dir)

        # 标签
        self.labels = self.dataset_manager.get_all_labels()
        self.dataset_manager.ensure_label_dirs(self.labels)

        self.init_ui()
        self.init_audio()

    # ======================== UI 初始化 ========================

    def init_ui(self) -> None:
        """初始化用户界面"""
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.setStyleSheet(MAIN_WINDOW_STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 左侧：文件树
        left_widget = self._create_file_tree_panel()
        # 右侧：控制和显示
        right_widget = self._create_control_panel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([SPLITTER_LEFT_SIZE, SPLITTER_RIGHT_SIZE])
        main_layout.addWidget(splitter)

        # 音量刷新定时器
        self.volume_timer = QTimer()
        self.volume_timer.timeout.connect(self.update_volume_display)
        self.volume_timer.start(VOLUME_UPDATE_INTERVAL_MS)

    def _create_file_tree_panel(self) -> QWidget:
        """创建左侧文件树面板"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        tree_group = QGroupBox("数据集文件结构")
        tree_layout = QVBoxLayout(tree_group)

        self.tree_view = QTreeView()
        self.file_model = AudioFileSystemModel()
        self.file_model.setRootPath(self.base_dir)
        self.tree_view.setModel(self.file_model)
        self.tree_view.setRootIndex(self.file_model.index(self.base_dir))
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setAnimated(True)
        self.tree_view.setIndentation(TREE_INDENTATION)
        self.tree_view.doubleClicked.connect(self.on_tree_double_click)
        tree_layout.addWidget(self.tree_view)
        left_layout.addWidget(tree_group)
        return left_widget

    def _create_control_panel(self) -> QWidget:
        """创建右侧控制面板"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)

        # 波形图
        waveform_group = self._create_waveform_group()
        # 配置区域
        config_group = self._create_config_group()
        # 状态
        status_layout = self._create_status_layout()
        # 按钮
        button_layout = self._create_button_layout()
        # 底部路径
        bottom_layout = self._create_bottom_layout()

        right_layout.addWidget(waveform_group)
        right_layout.addWidget(config_group)
        right_layout.addLayout(status_layout)
        right_layout.addLayout(button_layout)
        right_layout.addLayout(bottom_layout)
        right_layout.addStretch()
        return right_widget

    def _create_waveform_group(self) -> QGroupBox:
        """创建波形图组件"""
        waveform_group = QGroupBox("实时音频波形")
        waveform_layout = QVBoxLayout(waveform_group)

        self.waveform_widget = pg.PlotWidget()
        self.waveform_widget.setBackground("#2b2b2b")
        self.waveform_widget.setLabel("left", "振幅")
        self.waveform_widget.setLabel("bottom", "采样点")
        self.waveform_widget.showGrid(x=True, y=True, alpha=0.3)
        self.waveform_widget.setYRange(-32768, 32768)
        self.waveform_curve = self.waveform_widget.plot(
            pen=pg.mkPen(color="#5a8aba", width=2)
        )
        waveform_layout.addWidget(self.waveform_widget)

        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("音量:"))
        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(0, 100)
        volume_layout.addWidget(self.volume_bar)
        waveform_layout.addLayout(volume_layout)
        return waveform_group

    def _create_config_group(self) -> QGroupBox:
        """创建配置区域"""
        config_group = QGroupBox("采集配置")
        config_layout = QVBoxLayout(config_group)

        # 时间输入
        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("采集时间:"))
        self.time_input = QLineEdit("1")
        self.time_input.setFixedWidth(60)
        time_row.addWidget(self.time_input)
        time_row.addWidget(QLabel("秒"))
        time_row.addStretch()
        config_layout.addLayout(time_row)

        # 快捷按钮
        shortcut_row = QHBoxLayout()
        shortcut_row.addWidget(QLabel("快捷:"))
        btn_1s = QPushButton("1秒")
        btn_2s = QPushButton("2秒")
        btn_5s = QPushButton("5秒")
        btn_1s.clicked.connect(lambda: self.time_input.setText("1"))
        btn_2s.clicked.connect(lambda: self.time_input.setText("2"))
        btn_5s.clicked.connect(lambda: self.time_input.setText("5"))
        shortcut_row.addWidget(btn_1s)
        shortcut_row.addWidget(btn_2s)
        shortcut_row.addWidget(btn_5s)
        shortcut_row.addStretch()
        config_layout.addLayout(shortcut_row)

        # 标签选择 + 管理按钮
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("指令标签:"))
        self.label_combo = QComboBox()
        self.refresh_labels()
        label_row.addWidget(self.label_combo)

        self.manage_btn = QPushButton("标签管理")
        self.manage_btn.setFixedWidth(80)
        self.manage_btn.setToolTip("新增或删除标签")
        self.manage_btn.clicked.connect(self.manage_labels)
        label_row.addWidget(self.manage_btn)
        label_row.addStretch()
        config_layout.addLayout(label_row)

        return config_group

    def _create_status_layout(self) -> QHBoxLayout:
        """创建状态指示器布局"""
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("状态:"))
        self.status_light = QLabel(STATUS_IDLE_TEXT)
        self.status_light.setStyleSheet(STATUS_IDLE_STYLE)
        status_layout.addWidget(self.status_light)
        status_layout.addStretch()
        return status_layout

    def _create_button_layout(self) -> QHBoxLayout:
        """创建操作按钮布局"""
        button_layout = QHBoxLayout()
        self.record_btn = QPushButton(BUTTON_RECORD_TEXT)
        self.record_btn.clicked.connect(self.start_recording)
        self.record_btn.setStyleSheet(RECORD_BUTTON_STYLE)

        self.update_btn = QPushButton("📊 更新数据集")
        self.update_btn.clicked.connect(self.update_dataset)

        self.export_btn = QPushButton("💾 导出数据集")
        self.export_btn.clicked.connect(self.export_dataset)

        button_layout.addWidget(self.record_btn)
        button_layout.addWidget(self.update_btn)
        button_layout.addWidget(self.export_btn)
        return button_layout

    def _create_bottom_layout(self) -> QHBoxLayout:
        """创建底部路径显示布局"""
        bottom_layout = QHBoxLayout()
        self.path_label = QLabel(f"数据集路径: {self.base_dir}")
        self.path_label.setStyleSheet(PATH_LABEL_STYLE)
        self.path_label.setWordWrap(True)

        self.choose_dir_btn = QPushButton("📁 选择目录")
        self.choose_dir_btn.setFixedWidth(100)
        self.choose_dir_btn.clicked.connect(self.change_dataset_dir)

        bottom_layout.addWidget(self.path_label)
        bottom_layout.addWidget(self.choose_dir_btn)
        bottom_layout.addStretch()
        return bottom_layout

    def init_audio(self) -> None:
        """初始化音频目录"""
        self.dataset_manager.ensure_label_dirs(self.labels)

    # ======================== 标签管理 ========================

    def refresh_labels(self) -> None:
        """刷新标签下拉框"""
        self.label_combo.clear()
        self.label_combo.addItems(self.labels)

    def manage_labels(self) -> None:
        """打开标签管理对话框"""
        dlg = LabelManageDialog(self.labels, self.base_dir, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_labels = dlg.get_labels()
            if set(new_labels) != set(self.labels):
                self.labels = new_labels
                self.refresh_labels()
                self.file_model.setRootPath(self.base_dir)
                self.tree_view.setRootIndex(self.file_model.index(self.base_dir))
                QMessageBox.information(self, "成功", "标签已更新，文件树已刷新")

    # ======================== 数据集目录 ========================

    def change_dataset_dir(self) -> None:
        """切换数据集目录"""
        new_dir = QFileDialog.getExistingDirectory(
            self, "选择数据集根目录", self.base_dir
        )
        if not new_dir:
            return
        self.base_dir = new_dir
        self.dataset_manager.set_base_dir(self.base_dir)

        self.existing_labels = self.dataset_manager.load_existing_labels()
        all_labels = list(set(DEFAULT_LABELS + self.existing_labels))
        self.labels = all_labels
        for label in self.labels:
            os.makedirs(os.path.join(self.base_dir, label), exist_ok=True)

        self.refresh_labels()
        self.file_model.setRootPath(self.base_dir)
        self.tree_view.setRootIndex(self.file_model.index(self.base_dir))
        self.path_label.setText(f"数据集路径: {self.base_dir}")
        QMessageBox.information(self, "成功", f"数据集目录已切换至:\n{self.base_dir}")

    # ======================== 文件树操作 ========================

    def on_tree_double_click(self, index: QModelIndex) -> None:
        """双击文件树项"""
        file_path = self.file_model.filePath(index)
        if os.path.isfile(file_path) and file_path.endswith(".wav"):
            self.play_audio_file(file_path)
        elif os.path.isfile(file_path) and file_path.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                QMessageBox.information(self, "文件内容", content[:500])
            except Exception as e:
                QMessageBox.warning(self, "错误", f"无法读取文件: {e}")

    def play_audio_file(self, filepath: str) -> None:
        """播放音频文件"""
        if not os.path.exists(filepath):
            QMessageBox.warning(self, "警告", "音频文件不存在")
            return
        if self.audio_player and self.audio_player.isRunning():
            self.audio_player.terminate()
        self.audio_player = AudioPlayer(filepath)
        self.audio_player.start()

    # ======================== 录音控制 ========================

    def start_recording(self) -> None:
        """开始录音"""
        if self.recorder_thread and self.recorder_thread.isRunning():
            QMessageBox.warning(self, "警告", "正在录制中，请稍后再试")
            return
        try:
            duration = float(self.time_input.text())
            if duration < MIN_RECORDING_DURATION or duration > MAX_RECORDING_DURATION:
                QMessageBox.warning(
                    self,
                    "警告",
                    f"采集时间必须在{int(MIN_RECORDING_DURATION)}-{int(MAX_RECORDING_DURATION)}秒之间",
                )
                return
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的采集时间")
            return

        label = self.label_combo.currentText()
        if not label:
            QMessageBox.warning(self, "警告", "请选择或添加标签")
            return

        self.recorder_thread = AudioRecorderThread(
            duration, label, self.base_dir
        )
        self.recorder_thread.waveform_data.connect(self.update_waveform)
        self.recorder_thread.volume_level.connect(self.update_volume)
        self.recorder_thread.recording_finished.connect(self.on_recording_finished)
        self.recorder_thread.recording_error.connect(self.on_recording_error)

        self.status_light.setText(STATUS_RECORDING_TEXT)
        self.status_light.setStyleSheet(STATUS_RECORDING_STYLE)
        self.record_btn.setEnabled(False)
        self.record_btn.setText(BUTTON_RECORDING_TEXT)
        self.recorder_thread.start()

    def update_waveform(self, data: np.ndarray) -> None:
        """更新波形显示"""
        try:
            if data is None or len(data) == 0:
                return
            self.waveform_curve.setData(data)
        except Exception as e:
            print(f"波形更新错误: {e}")

    def update_volume(self, level: float) -> None:
        """更新音量值"""
        try:
            if np.isnan(level) or np.isinf(level):
                self.current_volume = 0.0
            else:
                self.current_volume = float(level)
        except Exception:
            self.current_volume = 0.0

    def update_volume_display(self) -> None:
        """定时器调用：刷新音量条"""
        try:
            self.volume_bar.setValue(int(self.current_volume))
            if not (self.recorder_thread and self.recorder_thread.isRunning()):
                self.current_volume = max(0.0, self.current_volume - VOLUME_DECAY_RATE)
        except Exception:
            pass

    def on_recording_finished(self, save_path: str) -> None:
        """录音完成回调"""
        self.status_light.setText(STATUS_IDLE_TEXT)
        self.status_light.setStyleSheet(STATUS_IDLE_STYLE)
        self.record_btn.setEnabled(True)
        self.record_btn.setText(BUTTON_RECORD_TEXT)
        self.file_model.setRootPath(self.base_dir)
        self.tree_view.setRootIndex(self.file_model.index(self.base_dir))
        QMessageBox.information(self, "成功", f"音频已保存到:\n{save_path}")

    def on_recording_error(self, error_msg: str) -> None:
        """录音错误回调"""
        self.status_light.setText(STATUS_IDLE_TEXT)
        self.status_light.setStyleSheet(STATUS_IDLE_STYLE)
        self.record_btn.setEnabled(True)
        self.record_btn.setText(BUTTON_RECORD_TEXT)
        QMessageBox.critical(self, "错误", error_msg)

    # ======================== 数据集操作 ========================

    def update_dataset(self) -> None:
        """更新数据集（划分训练/验证集）"""
        train_count, val_count = self.dataset_manager.update_dataset()
        if train_count == 0 and val_count == 0:
            QMessageBox.warning(self, "警告", "没有找到任何音频文件")
            return
        # 刷新文件树以显示新生成的txt文件
        self.file_model.setRootPath(self.base_dir)
        self.tree_view.setRootIndex(self.file_model.index(self.base_dir))
        QMessageBox.information(
            self, "成功", f"数据集已更新！\n训练集: {train_count}个\n验证集: {val_count}个"
        )

    def export_dataset(self) -> None:
        """导出数据集为 ZIP"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{EXPORT_FILENAME_PREFIX}{timestamp}.zip"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出数据集", default_name, EXPORT_FILE_FILTER
        )
        if not file_path:
            return
        if self.dataset_manager.export_dataset(file_path):
            QMessageBox.information(self, "成功", f"数据集已导出到:\n{file_path}")
        else:
            QMessageBox.critical(self, "错误", "导出失败")

    # ======================== 关闭事件 ========================

    def closeEvent(self, event) -> None:
        """窗口关闭时清理资源"""
        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.stop_recording()
            self.recorder_thread.wait(1000)
        if self.audio_player and self.audio_player.isRunning():
            self.audio_player.terminate()
        event.accept()
