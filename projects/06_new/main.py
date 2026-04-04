#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音指令采集工具 - 支持自定义标签 + 文件树显示 + 可切换数据集目录
基于 Python + PySide6 实现
"""

import sys
import os
import wave
import shutil
import zipfile
import numpy as np
from datetime import datetime
from typing import List, Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QLineEdit, QPushButton, QTreeView, QFileSystemModel,
    QGroupBox, QSplitter, QProgressBar, QMessageBox, QFileDialog, QDialog,
    QDialogButtonBox, QListWidget, QListWidgetItem, QVBoxLayout as QVBoxLayoutDlg,
    QPushButton as QPushButtonDlg, QHBoxLayout as QHBoxLayoutDlg, QInputDialog
)
from PySide6.QtCore import (
    Qt, QThread, Signal, QTimer, QModelIndex, QDir
)
from PySide6.QtGui import QFont

import pyqtgraph as pg
import pyaudio
import torchaudio


# ==================== 音频录制线程 ====================
class AudioRecorderThread(QThread):
    waveform_data = Signal(np.ndarray)
    recording_finished = Signal(str)
    recording_error = Signal(str)
    volume_level = Signal(float)

    def __init__(self, duration: float, label: str, save_dir: str):
        super().__init__()
        self.duration = duration
        self.label = label
        self.save_dir = save_dir
        self.is_recording = True
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024

    def run(self):
        try:
            p = pyaudio.PyAudio()
            if p.get_device_count() == 0:
                self.recording_error.emit("未找到音频设备")
                p.terminate()
                return

            stream = p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )

            frames = []
            frames_per_second = int(self.rate / self.chunk)
            total_frames = int(self.duration * frames_per_second)

            for i in range(total_frames):
                if not self.is_recording:
                    break
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)

                try:
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    if len(audio_data) > 0:
                        rms = np.sqrt(np.mean(audio_data**2))
                        if np.isnan(rms) or np.isinf(rms):
                            volume_level = 0
                        else:
                            volume_level = min(100, int((rms / 32768) * 100))
                    else:
                        volume_level = 0
                except Exception:
                    volume_level = 0
                self.volume_level.emit(volume_level)

                try:
                    if len(audio_data) > 500:
                        display_data = audio_data[::len(audio_data)//500]
                    else:
                        display_data = audio_data
                    self.waveform_data.emit(display_data)
                except Exception:
                    self.waveform_data.emit(np.zeros(100))

                self.msleep(int(self.duration * 1000 / total_frames))

            stream.stop_stream()
            stream.close()
            p.terminate()

            if frames and self.is_recording:
                save_path = self.save_audio(frames)
                self.recording_finished.emit(save_path)
            else:
                self.recording_error.emit("录制被取消")
        except Exception as e:
            self.recording_error.emit(f"录制错误: {str(e)}")

    def save_audio(self, frames: List[bytes]) -> str:
        label_dir = os.path.join(self.save_dir, self.label)
        os.makedirs(label_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{self.label}_{timestamp}.wav"
        filepath = os.path.join(label_dir, filename)
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
        return filepath

    def stop_recording(self):
        self.is_recording = False


# ==================== 音频播放线程 ====================
class AudioPlayer(QThread):
    play_finished = Signal()

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            wf = wave.open(self.filepath, 'rb')
            p = pyaudio.PyAudio()
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
        except Exception as e:
            print(f"播放错误: {e}")
        finally:
            self.play_finished.emit()


# ==================== 自定义标签管理对话框 ====================
class LabelManageDialog(QDialog):
    def __init__(self, current_labels: List[str], base_dir: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("自定义标签管理")
        self.setMinimumWidth(300)
        self.base_dir = base_dir
        self.all_labels = current_labels.copy()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayoutDlg(self)

        self.list_widget = QListWidget()
        for label in self.all_labels:
            self.list_widget.addItem(label)
        layout.addWidget(self.list_widget)

        btn_layout = QHBoxLayoutDlg()
        self.add_btn = QPushButtonDlg("新增标签")
        self.del_btn = QPushButtonDlg("删除选中")
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.del_btn)
        layout.addLayout(btn_layout)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.add_btn.clicked.connect(self.add_label)
        self.del_btn.clicked.connect(self.delete_label)

    def add_label(self):
        new_label, ok = QInputDialog.getText(self, "新增标签", "请输入新标签名称:")
        if ok and new_label.strip():
            new_label = new_label.strip()
            if new_label not in self.all_labels:
                self.all_labels.append(new_label)
                self.list_widget.addItem(new_label)
                label_dir = os.path.join(self.base_dir, new_label)
                os.makedirs(label_dir, exist_ok=True)
                QMessageBox.information(self, "成功", f"标签 '{new_label}' 已添加，文件夹已创建")
            else:
                QMessageBox.warning(self, "警告", "标签已存在")

    def delete_label(self):
        current = self.list_widget.currentItem()
        if not current:
            QMessageBox.warning(self, "警告", "请先选中要删除的标签")
            return
        label = current.text()
        reply = QMessageBox.question(self, "确认删除",
                                     f"确定要删除标签 '{label}' 吗？\n对应的文件夹也会被删除（如果有音频文件将丢失）",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            self.all_labels.remove(label)
            self.list_widget.takeItem(self.list_widget.row(current))
            label_dir = os.path.join(self.base_dir, label)
            if os.path.exists(label_dir):
                shutil.rmtree(label_dir)
            QMessageBox.information(self, "成功", f"标签 '{label}' 已删除")

    def get_labels(self):
        return self.all_labels


# ==================== 文件系统树形视图（显示 wav 和 txt 文件） ====================
class AudioFileSystemModel(QFileSystemModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFilter(QDir.Filter.AllDirs | QDir.Filter.NoDotAndDotDot | QDir.Filter.Files)
        self.setNameFilters(["*.wav", "*.txt"])   # 同时显示 wav 和 txt
        self.setNameFilterDisables(False)

    def columnCount(self, parent=QModelIndex()):
        return 1


# ==================== 主窗口 ====================
class VoiceCommandCollector(QMainWindow):
    def __init__(self):
        super().__init__()
        self.recorder_thread: Optional[AudioRecorderThread] = None
        self.audio_player: Optional[AudioPlayer] = None
        self.current_volume = 0.0
        # 数据集根目录
        self.base_dir = "//Users/logicye/Code/my_Datasets/系统控制"
        os.makedirs(self.base_dir, exist_ok=True)

        self.default_labels = ["动鼠标", "截屏"]
        self.existing_labels = self.load_existing_labels()
        all_labels = list(set(self.default_labels + self.existing_labels))
        self.labels = all_labels
        for label in self.labels:
            os.makedirs(os.path.join(self.base_dir, label), exist_ok=True)

        self.init_ui()
        self.init_audio()

    def load_existing_labels(self) -> List[str]:
        labels = []
        if os.path.exists(self.base_dir):
            for item in os.listdir(self.base_dir):
                item_path = os.path.join(self.base_dir, item)
                if os.path.isdir(item_path) and not item.startswith("."):
                    labels.append(item)
        return labels

    def refresh_labels(self):
        self.label_combo.clear()
        self.label_combo.addItems(self.labels)

    def change_dataset_dir(self):
        new_dir = QFileDialog.getExistingDirectory(self, "选择数据集根目录", self.base_dir)
        if not new_dir:
            return
        self.base_dir = new_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.existing_labels = self.load_existing_labels()
        all_labels = list(set(self.default_labels + self.existing_labels))
        self.labels = all_labels
        for label in self.labels:
            os.makedirs(os.path.join(self.base_dir, label), exist_ok=True)
        self.refresh_labels()
        self.file_model.setRootPath(self.base_dir)
        self.tree_view.setRootIndex(self.file_model.index(self.base_dir))
        self.path_label.setText(f"数据集路径: {self.base_dir}")
        QMessageBox.information(self, "成功", f"数据集目录已切换至:\n{self.base_dir}")

    def init_ui(self):
        self.setWindowTitle("语音指令采集工具 - 自定义标签版")
        self.setMinimumSize(1200, 700)

        # 全局样式表（确保按钮文字可见）
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QGroupBox {
                font-size: 14px; font-weight: bold;
                border: 2px solid #3c3c3c; border-radius: 8px;
                margin-top: 10px; padding-top: 10px;
                color: #e0e0e0;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLabel { color: #e0e0e0; font-size: 12px; }
            QPushButton {
                background-color: #4a4a4a; color: white;
                border: none; padding: 8px 16px; border-radius: 5px;
                font-size: 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #5a5a5a; }
            QPushButton:pressed { background-color: #3a3a3a; }
            QLineEdit, QComboBox {
                background-color: #3c3c3c; color: #e0e0e0;
                border: 1px solid #5a5a5a; border-radius: 4px;
                padding: 5px; font-size: 12px;
            }
            QTreeView {
                background-color: #3c3c3c; color: #e0e0e0;
                border: 1px solid #5a5a5a; border-radius: 4px;
                font-size: 11px;
            }
            QTreeView::item:hover { background-color: #4a6a8a; }
            QTreeView::item:selected { background-color: #5a8aba; }
            QProgressBar {
                border: 1px solid #5a5a5a; border-radius: 3px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #4caf50, stop:0.5 #ffc107, stop:1 #f44336);
                border-radius: 2px;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # 左侧：文件树
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
        self.tree_view.setIndentation(20)
        self.tree_view.doubleClicked.connect(self.on_tree_double_click)
        tree_layout.addWidget(self.tree_view)
        left_layout.addWidget(tree_group)

        # 右侧：控制和显示
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)

        # 波形图
        waveform_group = QGroupBox("实时音频波形")
        waveform_layout = QVBoxLayout(waveform_group)
        self.waveform_widget = pg.PlotWidget()
        self.waveform_widget.setBackground('#2b2b2b')
        self.waveform_widget.setLabel('left', '振幅')
        self.waveform_widget.setLabel('bottom', '采样点')
        self.waveform_widget.showGrid(x=True, y=True, alpha=0.3)
        self.waveform_widget.setYRange(-32768, 32768)
        self.waveform_curve = self.waveform_widget.plot(pen=pg.mkPen(color='#5a8aba', width=2))
        waveform_layout.addWidget(self.waveform_widget)

        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("音量:"))
        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(0, 100)
        volume_layout.addWidget(self.volume_bar)
        waveform_layout.addLayout(volume_layout)

        # 配置区域
        config_group = QGroupBox("采集配置")
        config_layout = QVBoxLayout(config_group)

        time_row = QHBoxLayout()
        time_row.addWidget(QLabel("采集时间:"))
        self.time_input = QLineEdit("1")
        self.time_input.setFixedWidth(60)
        time_row.addWidget(self.time_input)
        time_row.addWidget(QLabel("秒"))
        time_row.addStretch()
        config_layout.addLayout(time_row)

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

        # 标签选择 + 管理按钮（改用文字按钮“标签管理”）
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

        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("状态:"))
        self.status_light = QLabel("● 休闲中")
        self.status_light.setStyleSheet("color: #4caf50; font-size: 14px; font-weight: bold;")
        status_layout.addWidget(self.status_light)
        status_layout.addStretch()

        button_layout = QHBoxLayout()
        self.record_btn = QPushButton("🎤 语音采集")
        self.record_btn.clicked.connect(self.start_recording)
        self.record_btn.setStyleSheet("background-color: #2196f3; font-size: 14px; padding: 10px;")
        self.update_btn = QPushButton("📊 更新数据集")
        self.update_btn.clicked.connect(self.update_dataset)
        self.export_btn = QPushButton("💾 导出数据集")
        self.export_btn.clicked.connect(self.export_dataset)
        button_layout.addWidget(self.record_btn)
        button_layout.addWidget(self.update_btn)
        button_layout.addWidget(self.export_btn)

        # 底部：路径显示 + 选择目录按钮
        bottom_layout = QHBoxLayout()
        self.path_label = QLabel(f"数据集路径: {self.base_dir}")
        self.path_label.setStyleSheet("color: #888; font-size: 10px; padding: 5px;")
        self.path_label.setWordWrap(True)
        self.choose_dir_btn = QPushButton("📁 选择目录")
        self.choose_dir_btn.setFixedWidth(100)
        self.choose_dir_btn.clicked.connect(self.change_dataset_dir)
        bottom_layout.addWidget(self.path_label)
        bottom_layout.addWidget(self.choose_dir_btn)
        bottom_layout.addStretch()

        right_layout.addWidget(waveform_group)
        right_layout.addWidget(config_group)
        right_layout.addLayout(status_layout)
        right_layout.addLayout(button_layout)
        right_layout.addLayout(bottom_layout)
        right_layout.addStretch()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 800])
        main_layout.addWidget(splitter)

        self.volume_timer = QTimer()
        self.volume_timer.timeout.connect(self.update_volume_display)
        self.volume_timer.start(100)

    def init_audio(self):
        for label in self.labels:
            os.makedirs(os.path.join(self.base_dir, label), exist_ok=True)

    def manage_labels(self):
        dlg = LabelManageDialog(self.labels, self.base_dir, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_labels = dlg.get_labels()
            if set(new_labels) != set(self.labels):
                self.labels = new_labels
                self.refresh_labels()
                self.file_model.setRootPath(self.base_dir)
                self.tree_view.setRootIndex(self.file_model.index(self.base_dir))
                QMessageBox.information(self, "成功", "标签已更新，文件树已刷新")

    def on_tree_double_click(self, index: QModelIndex):
        file_path = self.file_model.filePath(index)
        if os.path.isfile(file_path) and file_path.endswith('.wav'):
            self.play_audio_file(file_path)
        elif os.path.isfile(file_path) and file_path.endswith('.txt'):
            # 可选的：双击 txt 文件时显示内容
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                QMessageBox.information(self, "文件内容", content[:500])
            except Exception as e:
                QMessageBox.warning(self, "错误", f"无法读取文件: {e}")

    def play_audio_file(self, filepath: str):
        if not os.path.exists(filepath):
            QMessageBox.warning(self, "警告", "音频文件不存在")
            return
        if self.audio_player and self.audio_player.isRunning():
            self.audio_player.terminate()
        self.audio_player = AudioPlayer(filepath)
        self.audio_player.start()

    def start_recording(self):
        if self.recorder_thread and self.recorder_thread.isRunning():
            QMessageBox.warning(self, "警告", "正在录制中，请稍后再试")
            return
        try:
            duration = float(self.time_input.text())
            if duration < 1 or duration > 5:
                QMessageBox.warning(self, "警告", "采集时间必须在1-5秒之间")
                return
        except ValueError:
            QMessageBox.warning(self, "警告", "请输入有效的采集时间")
            return
        label = self.label_combo.currentText()
        if not label:
            QMessageBox.warning(self, "警告", "请选择或添加标签")
            return
        self.recorder_thread = AudioRecorderThread(duration, label, self.base_dir)
        self.recorder_thread.waveform_data.connect(self.update_waveform)
        self.recorder_thread.volume_level.connect(self.update_volume)
        self.recorder_thread.recording_finished.connect(self.on_recording_finished)
        self.recorder_thread.recording_error.connect(self.on_recording_error)
        self.status_light.setText("● 录音中")
        self.status_light.setStyleSheet("color: #f44336; font-size: 14px; font-weight: bold;")
        self.record_btn.setEnabled(False)
        self.record_btn.setText("🔴 录音中...")
        self.recorder_thread.start()

    def update_waveform(self, data: np.ndarray):
        try:
            if data is None or len(data) == 0:
                return
            self.waveform_curve.setData(data)
        except Exception as e:
            print(f"波形更新错误: {e}")

    def update_volume(self, level: float):
        try:
            if np.isnan(level) or np.isinf(level):
                self.current_volume = 0.0
            else:
                self.current_volume = float(level)
        except Exception:
            self.current_volume = 0.0

    def update_volume_display(self):
        try:
            self.volume_bar.setValue(int(self.current_volume))
            if not (self.recorder_thread and self.recorder_thread.isRunning()):
                self.current_volume = max(0.0, self.current_volume - 5.0)
        except Exception:
            pass

    def on_recording_finished(self, save_path: str):
        self.status_light.setText("● 休闲中")
        self.status_light.setStyleSheet("color: #4caf50; font-size: 14px; font-weight: bold;")
        self.record_btn.setEnabled(True)
        self.record_btn.setText("🎤 语音采集")
        self.file_model.setRootPath(self.base_dir)
        self.tree_view.setRootIndex(self.file_model.index(self.base_dir))
        QMessageBox.information(self, "成功", f"音频已保存到:\n{save_path}")

    def on_recording_error(self, error_msg: str):
        self.status_light.setText("● 休闲中")
        self.status_light.setStyleSheet("color: #4caf50; font-size: 14px; font-weight: bold;")
        self.record_btn.setEnabled(True)
        self.record_btn.setText("🎤 语音采集")
        QMessageBox.critical(self, "错误", error_msg)

    def update_dataset(self):
        audio_files = []
        for label in self.labels:
            label_dir = os.path.join(self.base_dir, label)
            if os.path.exists(label_dir):
                for file in os.listdir(label_dir):
                    if file.endswith('.wav'):
                        rel_path = os.path.join(label, file)
                        audio_files.append(rel_path)
        if not audio_files:
            QMessageBox.warning(self, "警告", "没有找到任何音频文件")
            return
        np.random.seed(42)
        indices = np.random.permutation(len(audio_files))
        split_idx = int(len(audio_files) * 0.8)
        train_files = [audio_files[i] for i in indices[:split_idx]]
        val_files = [audio_files[i] for i in indices[split_idx:]]
        val_path = os.path.join(self.base_dir, 'validation_list.txt')
        with open(val_path, 'w', encoding='utf-8') as f:
            for file in val_files:
                f.write(file + '\n')
        test_path = os.path.join(self.base_dir, 'testing_list.txt')
        with open(test_path, 'w', encoding='utf-8') as f:
            for file in train_files:
                f.write(file + '\n')
        # 刷新文件树以显示新生成的txt文件
        self.file_model.setRootPath(self.base_dir)
        self.tree_view.setRootIndex(self.file_model.index(self.base_dir))
        QMessageBox.information(self, "成功", f"数据集已更新！\n训练集: {len(train_files)}个\n验证集: {len(val_files)}个")

    def export_dataset(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"dataset_{timestamp}.zip"
        file_path, _ = QFileDialog.getSaveFileName(self, "导出数据集", default_name, "ZIP文件 (*.zip)")
        if not file_path:
            return
        try:
            with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.base_dir):
                    for file in files:
                        if file.endswith('.wav') or file.endswith('.txt'):
                            full_path = os.path.join(root, file)
                            arcname = os.path.relpath(full_path, self.base_dir)
                            zipf.write(full_path, arcname)
            QMessageBox.information(self, "成功", f"数据集已导出到:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def closeEvent(self, event):
        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.stop_recording()
            self.recorder_thread.wait(1000)
        if self.audio_player and self.audio_player.isRunning():
            self.audio_player.terminate()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = VoiceCommandCollector()
    window.show()
    sys.exit(app.exec())