#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音控制器 - 增加冷却机制和置信度阈值，鼠标动作幅度3倍，1秒完成
"""

import os
import sys
import json
import time
import math
import subprocess
from datetime import datetime
from collections import deque

import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
import numpy as np
from transformers import PretrainedConfig, PreTrainedModel
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QPushButton, QLabel, QFrame)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot
from PySide6.QtGui import QPainter, QPen, QColor, QFont
import pyaudio
import pyautogui
from PIL import ImageGrab

# 安全设置
pyautogui.FAILSAFE = True

# 配置参数
# CONFIDENCE_THRESHOLD = 0.60   # 置信度阈值，低于此值忽略
# COOLDOWN_SECONDS = 1.0        # 动作执行后的冷却时间（秒）
# ENERGY_THRESHOLD = 0.005      # 音频能量阈值（归一化后），低于此值视为静音

CONFIDENCE_THRESHOLD = 0.45   # 平衡准确率和触发率
ENERGY_THRESHOLD = 0      # 稍微过滤静音即可
COOLDOWN_SECONDS = 1       # 缩短冷却，便于快速连续指令

# ==================== 1. 模型定义 ====================
class SimpleCNNForAudioClassification(PreTrainedModel):
    config_class = PretrainedConfig
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        conv_layers = []
        in_channels = 1
        for out_channels in config.conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=config.kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
        conv_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.conv = nn.Sequential(*conv_layers)
        self.classifier = nn.Linear(config.conv_channels[-1] * 4 * 4, config.num_labels)
        self.post_init()
    def forward(self, input_values, labels=None, return_dict=True):
        x = self.conv(input_values)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        if return_dict:
            return {"loss": loss, "logits": logits}
        return (loss, logits) if loss is not None else (logits,)

# ==================== 2. 音频处理与推理线程 ====================
class AudioProcessor(QThread):
    result_signal = Signal(str)
    waveform_signal = Signal(np.ndarray)
    def __init__(self, model_path, duration=2, sample_rate=16000):
        super().__init__()
        self.model_path = model_path
        self.duration = duration
        self.sample_rate = sample_rate
        self.chunk = 1024
        self.audio_buffer = deque(maxlen=int(sample_rate * duration))
        self.is_running = False
        self.recording = False
        self.load_model()
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.cooldown = False
        self.cooldown_timer = QTimer()
        self.cooldown_timer.setSingleShot(True)
        self.cooldown_timer.timeout.connect(self._clear_cooldown)

    def load_model(self):
        self.model = SimpleCNNForAudioClassification.from_pretrained(self.model_path)
        self.model.eval()
        label_path = os.path.join(self.model_path, "label_mapping.json")
        with open(label_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
        self.label2id = mapping["label2id"]
        print(f"模型加载成功，指令: {list(self.id2label.values())}")

    def start_recording(self):
        if self.is_running:
            return
        self.is_running = True
        self.recording = True
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  frames_per_buffer=self.chunk,
                                  stream_callback=self._audio_callback)
        self.stream.start_stream()
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_audio)
        self.timer.start(self.duration * 1000)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.recording:
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_buffer.extend(audio_data)
            if len(self.audio_buffer) == self.audio_buffer.maxlen:
                self.waveform_signal.emit(np.array(self.audio_buffer))
        return (in_data, pyaudio.paContinue)

    def process_audio(self):
        if not self.recording or len(self.audio_buffer) < self.sample_rate * self.duration:
            return
        audio = np.array(list(self.audio_buffer)[-self.sample_rate * self.duration:])
        energy = np.mean(audio**2)
        if energy < ENERGY_THRESHOLD:
            return
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        mel_spec = T.MelSpectrogram(sample_rate=self.sample_rate, n_mels=64, n_fft=1024, hop_length=512)
        mel = mel_spec(waveform)
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        with torch.no_grad():
            outputs = self.model(input_values=mel.unsqueeze(0), return_dict=True)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)
            max_prob, pred_id = torch.max(probs, dim=1)
            confidence = max_prob.item()
            command = self.id2label[pred_id.item()]
        if confidence >= CONFIDENCE_THRESHOLD:
            if not self.cooldown:
                self.result_signal.emit(command)
                self.cooldown = True
                self.cooldown_timer.start(int(COOLDOWN_SECONDS * 1000))
        else:
            print(f"低置信度忽略: {command} ({confidence:.2f})")

    def _clear_cooldown(self):
        self.cooldown = False

    def stop_recording(self):
        if self.is_running:
            self.recording = False
            self.is_running = False
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.timer.stop()
            self.cooldown_timer.stop()
            self.cooldown = False

    def run(self):
        pass

# ==================== 3. 动作执行线程（修改鼠标动作为幅度3倍、1秒完成）====================
class ActionExecutor(QThread):
    finished = Signal()
    def __init__(self, command):
        super().__init__()
        self.command = command
    def run(self):
        if self.command == "动鼠标":
            self.do_mouse_circle_and_zigzag()
        elif self.command == "截屏":
            self.do_screenshot()
        elif self.command == "打开计算器":
            self.open_calculator()
        self.finished.emit()

    def do_mouse_circle_and_zigzag(self):
        """画圆+上下往返"""
        start_x, start_y = pyautogui.position()
        radius = 300
        amplitude = 450
        total_duration = 0.6          # 总时长0.6秒
        # 画圆部分0.3秒
        circle_duration = total_duration * 0.5
        steps = 30                   # 减少步数，加快移动
        for i in range(steps + 1):
            t = i / steps
            angle = 2 * math.pi * t
            x = start_x + radius * math.cos(angle)
            y = start_y + radius * math.sin(angle)
            pyautogui.moveTo(x, y, duration=circle_duration / steps)
        # 上下往返部分0.3秒
        vertical_duration = total_duration * 0.5
        steps_v = 40                  # 减少步数
        for _ in range(2):
            # 向上
            for i in range(steps_v):
                y = start_y - amplitude * (i / steps_v)
                pyautogui.moveTo(start_x, y, duration=vertical_duration / steps_v / 2)
            # 向下
            for i in range(steps_v):
                y = start_y - amplitude * (1 - i / steps_v)
                pyautogui.moveTo(start_x, y, duration=vertical_duration / steps_v / 2)
       


    def do_screenshot(self):
        save_dir = os.path.join(os.getcwd(), "output_screen_capture")
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        screenshot = ImageGrab.grab()
        screenshot.save(filepath)
        print(f"截图已保存: {filepath}")

    def open_calculator(self):
        system = sys.platform
        try:
            if system == "win32":
                subprocess.Popen("calc.exe")
            elif system == "darwin":
                subprocess.Popen(["open", "-a", "Calculator"])
            else:
                subprocess.Popen(["gnome-calculator"])
        except Exception as e:
            print(f"打开计算器失败: {e}")

# ==================== 4. 波形显示控件 ====================
class WaveformWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.wave_data = np.zeros(1000)
        self.setMinimumHeight(80)
        self.setFrameShape(QFrame.Box)
        self.setStyleSheet("border: 1px solid #555; border-radius: 5px;")
    def update_waveform(self, data):
        if len(data) > 1000:
            step = max(1, len(data) // 1000)
            self.wave_data = data[::step][:1000]
        else:
            self.wave_data = data
        self.update()
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(30, 30, 40))
        pen = QPen(QColor(0, 200, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        w = self.width()
        h = self.height()
        center_y = h // 2
        if len(self.wave_data) == 0:
            painter.end()
            return
        max_val = max(abs(np.max(self.wave_data)), abs(np.min(self.wave_data))) + 1e-6
        norm_data = self.wave_data / max_val
        step_x = w / len(norm_data)
        points = []
        for i, val in enumerate(norm_data):
            x = int(i * step_x)
            y = int(center_y - val * (h // 3))
            points.append((x, y))
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
        painter.end()

# ==================== 5. 主窗口 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("语音控制器")
        self.setFixedSize(400, 300)
        self.setStyleSheet("background-color: #2C2F33;")
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        self.waveform = WaveformWidget()
        layout.addWidget(self.waveform)
        self.result_label = QLabel("等待指令...")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 12))
        self.result_label.setStyleSheet("color: #FFFFFF; background-color: #23272A; padding: 8px; border-radius: 5px;")
        layout.addWidget(self.result_label)
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始语音控制")
        self.stop_btn = QPushButton("停止语音控制")
        self.stop_btn.setEnabled(False)
        for btn in (self.start_btn, self.stop_btn):
            btn.setFixedHeight(40)
            btn.setFont(QFont("Arial", 10))
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #5865F2;
                    color: white;
                    border: none;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #4752C4;
                }
                QPushButton:pressed {
                    background-color: #3C45A5;
                }
                QPushButton:disabled {
                    background-color: #4E5058;
                }
            """)
            btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)
        self.status_label = QLabel("● 未开始")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 9))
        self.status_label.setStyleSheet("color: #99AAB5;")
        layout.addWidget(self.status_label)
        self.start_btn.clicked.connect(self.start_control)
        self.stop_btn.clicked.connect(self.stop_control)
        model_path = "/Users/logicye/Code/my_trained_models/model_speech_control_transformer"
        self.audio_processor = AudioProcessor(model_path)
        self.audio_processor.result_signal.connect(self.on_command)
        self.audio_processor.waveform_signal.connect(self.waveform.update_waveform)
        self.executor = None
    def start_control(self):
        self.audio_processor.start_recording()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("● 正在监听...")
        self.status_label.setStyleSheet("color: #57F287;")
    def stop_control(self):
        self.audio_processor.stop_recording()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("● 已停止")
        self.status_label.setStyleSheet("color: #ED4245;")
    @Slot(str)
    def on_command(self, command):
        self.result_label.setText(f"识别: {command}")
        if self.executor and self.executor.isRunning():
            return
        self.executor = ActionExecutor(command)
        self.executor.finished.connect(lambda: self.result_label.setText("等待指令..."))
        self.executor.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())