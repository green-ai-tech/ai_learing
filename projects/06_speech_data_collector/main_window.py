"""
语音指令采集工具 - 主窗口
"""
import os
from datetime import datetime
from typing import Optional

import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt, QTimer

from services.recorder_service import AudioRecorderThread
from services.player_service import AudioPlayer
from services.dataset_service import DatasetManager
from widgets.waveform_widget import WaveformWidget
from widgets.file_list_panel import FileListPanel
from widgets.control_panel import ControlPanel


class VoiceCommandCollector(QMainWindow):
    """语音指令采集工具主窗口"""

    def __init__(self):
        super().__init__()
        self.recorder_thread: Optional[AudioRecorderThread] = None
        self.audio_player: Optional[AudioPlayer] = None
        # 数据集路径
        self.base_dir = "/Users/logicye/Code/my_Datasets/方向数据集"

        # 创建数据集目录
        os.makedirs(self.base_dir, exist_ok=True)

        self._init_ui()
        self._init_audio()
        self._refresh_file_lists()

    def _init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("语音指令采集工具")
        self.setMinimumSize(1200, 700)

        # 设置全局样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                border: 2px solid #3c3c3c;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 10px;
                color: #e0e0e0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 12px;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QLineEdit, QComboBox {
                background-color: #3c3c3c;
                color: #e0e0e0;
                border: 1px solid #5a5a5a;
                border-radius: 4px;
                padding: 5px;
                font-size: 12px;
            }
            QListWidget {
                background-color: #3c3c3c;
                color: #e0e0e0;
                border: 1px solid #5a5a5a;
                border-radius: 4px;
                font-size: 11px;
            }
            QListWidget::item:hover {
                background-color: #4a6a8a;
            }
            QListWidget::item:selected {
                background-color: #5a8aba;
            }
            QProgressBar {
                border: 1px solid #5a5a5a;
                border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4caf50, stop:0.5 #ffc107, stop:1 #f44336);
                border-radius: 2px;
            }
            QPushButton[shortcut="true"] {
                background-color: #5a6a7a;
                padding: 5px 10px;
                font-size: 11px;
            }
            QPushButton[shortcut="true"]:hover {
                background-color: #6a7a8a;
            }
        """)

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # ========== 左侧区域：音频文件列表 ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        self.up_panel = FileListPanel("向上", self.base_dir)
        self.down_panel = FileListPanel("向下", self.base_dir)

        left_layout.addWidget(self.up_panel)
        left_layout.addWidget(self.down_panel)

        # ========== 右侧区域：控制和显示 ==========
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)

        # 波形显示区域
        self.waveform_widget = WaveformWidget()
        right_layout.addWidget(self.waveform_widget)

        # 控制面板
        self.control_panel = ControlPanel()
        self.control_panel.set_dataset_path(self.base_dir)
        right_layout.addWidget(self.control_panel)

        # 连接控制面板按钮信号
        self.control_panel.record_btn.clicked.connect(self._start_recording)
        self.control_panel.update_btn.clicked.connect(self._update_dataset)
        self.control_panel.export_btn.clicked.connect(self._export_dataset)

        # 使用分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 800])

        main_layout.addWidget(splitter)

        # 启动定时器更新音量显示
        self.volume_timer = QTimer()
        self.volume_timer.timeout.connect(self._update_volume_display)
        self.volume_timer.start(100)

    def _init_audio(self):
        """初始化音频相关"""
        # 创建数据集目录结构
        for label in ['向上', '向下']:
            os.makedirs(os.path.join(self.base_dir, label), exist_ok=True)

    def _start_recording(self):
        """开始录制音频"""
        if self.recorder_thread and self.recorder_thread.isRunning():
            QMessageBox.warning(self, "警告", "正在录制中，请稍后再试")
            return

        # 获取参数
        try:
            duration = float(self.control_panel.get_duration())
            if duration < 1 or duration > 5:
                QMessageBox.warning(self, "警告", "采集时间必须在1-5秒之间")
                return
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的采集时间")
            return

        label = self.control_panel.get_label()

        # 创建录制线程
        self.recorder_thread = AudioRecorderThread(duration, label, self.base_dir)
        self.recorder_thread.waveform_data.connect(self.waveform_widget.update_waveform)
        self.recorder_thread.volume_level.connect(self.waveform_widget.update_volume)
        self.recorder_thread.recording_finished.connect(self._on_recording_finished)
        self.recorder_thread.recording_error.connect(self._on_recording_error)

        # 更新UI状态
        self.control_panel.set_recording_status(True)

        # 开始录制
        self.recorder_thread.start()

    def _on_recording_finished(self, save_path: str):
        """录制完成回调"""
        self.control_panel.set_recording_status(False)

        # 刷新文件列表
        self._refresh_file_lists()

        # 显示成功消息
        QMessageBox.information(self, "成功", f"音频已保存到:\n{save_path}")

    def _on_recording_error(self, error_msg: str):
        """录制错误回调"""
        self.control_panel.set_recording_status(False)
        QMessageBox.critical(self, "错误", error_msg)

    def _refresh_file_lists(self):
        """刷新文件列表"""
        up_files = []
        down_files = []
        up_mod_time = None
        down_mod_time = None

        # 读取向上指令文件
        up_dir = os.path.join(self.base_dir, "向上")
        if os.path.exists(up_dir):
            for file in sorted(os.listdir(up_dir), reverse=True):
                if file.endswith('.wav'):
                    up_files.append(file)
                    file_path = os.path.join(up_dir, file)
                    mtime = os.path.getmtime(file_path)
                    if up_mod_time is None or mtime > up_mod_time:
                        up_mod_time = mtime

        # 读取向下指令文件
        down_dir = os.path.join(self.base_dir, "向下")
        if os.path.exists(down_dir):
            for file in sorted(os.listdir(down_dir), reverse=True):
                if file.endswith('.wav'):
                    down_files.append(file)
                    file_path = os.path.join(down_dir, file)
                    mtime = os.path.getmtime(file_path)
                    if down_mod_time is None or mtime > down_mod_time:
                        down_mod_time = mtime

        # 更新面板
        self.up_panel.refresh(up_files, datetime.fromtimestamp(up_mod_time) if up_mod_time else None)
        self.down_panel.refresh(down_files, datetime.fromtimestamp(down_mod_time) if down_mod_time else None)

    def _update_volume_display(self):
        """定时更新音量显示"""
        try:
            volume = self.waveform_widget.get_volume()
            volume_int = int(volume)
            self.waveform_widget.volume_bar.setValue(volume_int)

            # 如果不在录制中，逐渐降低音量显示
            if not (self.recorder_thread and self.recorder_thread.isRunning()):
                self.waveform_widget.decay_volume()
        except Exception:
            pass

    def _update_dataset(self):
        """更新数据集"""
        success, message = DatasetManager.update_dataset(self.base_dir)
        if success:
            QMessageBox.information(self, "成功", message)
        else:
            QMessageBox.warning(self, "警告", message)

    def _export_dataset(self):
        """导出数据集"""
        # 选择保存路径
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"方向数据集_{timestamp}.zip"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出数据集", default_name, "ZIP文件 (*.zip)"
        )

        if not file_path:
            return

        success, message = DatasetManager.export_dataset(self.base_dir, file_path)
        if success:
            QMessageBox.information(self, "成功", message)
        else:
            QMessageBox.critical(self, "错误", message)

    def closeEvent(self, event):
        """关闭事件处理"""
        # 停止录制线程
        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.stop_recording()
            self.recorder_thread.wait(1000)

        # 停止播放线程
        if self.audio_player and self.audio_player.isRunning():
            self.audio_player.terminate()

        event.accept()
