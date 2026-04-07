#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音控制器 - 按键Q触发
通过语音指令控制系统操作（移动鼠标、截屏、打开计算器）

工作流程：
1. 用户按住 Q 键说话
2. 系统录制音频并进行实时识别
3. 识别成功后执行对应的系统操作
4. 通过 GUI 简单显示识别结果和状态

作者：Logic Ye
日期：2026-04-04

“notes：写代码，应该有艺术感”
"""

import os
import sys
import json
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
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot
from PySide6.QtGui import QFont
import pyaudio
import pyautogui
from PIL import ImageGrab

# ==================== 安全设置 ====================
# 启用故障安全模式：将鼠标移动到屏幕角落可以中止 pyautogui 操作
pyautogui.FAILSAFE = True

# ==================== 配置参数 ====================
CONFIDENCE_THRESHOLD = 0.35             # 置信度阈值：预测概率超过此值才认为识别成功
ENERGY_THRESHOLD     = 0.001            # 能量检测阈值：用于判断是否有声音输入
COOLDOWN_SECONDS     = 0.5              # 动作执行后冷却时间（秒），防止重复触发
RECORD_DURATION      = 1                # 录音长度（秒），与模型训练相匹配

# ==================== 1. 模型定义 ====================
class SimpleCNNForAudioClassification(PreTrainedModel):
    """
    音频分类模型 - 与训练脚本中完全一致的定义
    
    模型结构：
    - 多层 2D 卷积 + BatchNorm + ReLU + MaxPooling
    - 自适应平均池化层（输出固定 4x4 尺寸）
    - 全连接分类层
    
   
    """
    config_class = PretrainedConfig  # 配置类（从 config.json 加载实际配置）
    
    def __init__(self, config):
        """
        初始化模型
        
        参数:
            config: 配置对象，包含模型超参数
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # 构建卷积层序列
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
        
        # 自适应池化层
        conv_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.conv = nn.Sequential(*conv_layers)
        
        # 分类层
        self.classifier = nn.Linear(config.conv_channels[-1] * 4 * 4, config.num_labels)
        self.post_init()
    
    def forward(self, input_values, labels=None, return_dict=True):
        """
        前向传播
        
        参数:
            input_values: 输入特征（Mel频谱图）
            labels: 标签（可选，用于计算损失）
            return_dict: 是否返回字典格式
        
        返回:
            包含 logits 和可选 loss 的字典或元组
        """
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

# ==================== 2. 录音与推理（按键Q触发）====================
class AudioRecorder(QThread):
    """
    音频录制线程 - 负责录音和实时识别（重头戏！！！）
    
    工作流程：
    1. 按下 Q 键时开始录音
    2. 通过回调函数持续读取音频数据
    3. 松开 Q 键时停止录音并进行识别
    4. 识别结果通过信号发送给主窗口
    
    特性：
    - 能量检测：自动过滤静音片段
    - 冷却机制：防止短时间内重复触发
    - 线程安全：在后台线程中运行，不阻塞 UI
    
    信号：
        result_signal: 识别结果信号 (指令, 置信度)
        status_signal: 状态变化信号 (状态文本)
    """
    result_signal = Signal(str, float)   # (指令, 置信度)
    status_signal = Signal(str)          # 状态变化信号

    def __init__(self, model_path, sample_rate=16000):
        """
        初始化录音器
        
        参数:
            model_path: 模型保存路径
            sample_rate: 音频采样率（Hz）
        """
        super().__init__()
        self.model_path     = model_path
        self.sample_rate    = sample_rate
        self.chunk          = 1024                  # 每次读取的音频采样点数
        self.recording      = False                 # 是否正在录音
        self.audio_frames   = []                    # 存储录音数据
        self.load_model()                           # 加载模型
        self.p              = pyaudio.PyAudio()     # 初始化 PyAudio
        self.stream         = None                  # 音频流
        self.cooldown       = False                 # 是否处于冷却状态
        self.cooldown_timer = QTimer()              # 冷却定时器
        self.cooldown_timer.setSingleShot(True)     # 单次触发
        self.cooldown_timer.timeout.connect(self._clear_cooldown)  # 冷却结束回调
        self.enabled        = True   # 始终启用

    def load_model(self):
        """加载模型和标签映射"""
        self.model = SimpleCNNForAudioClassification.from_pretrained(self.model_path)
        self.model.eval()  # 设置为评估模式
        
        # 加载标签映射
        label_path = os.path.join(self.model_path, "label_mapping.json")
        with open(label_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
        self.label2id = mapping["label2id"]
        print(f"模型加载成功，指令: {list(self.id2label.values())}")

    def start_recording(self):
        """开始录音"""
        if self.recording:
            return  # 如果正在录音，忽略重复请求
        self.recording = True
        self.audio_frames = []  # 清空之前的录音
        # 打开音频输入流
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  frames_per_buffer=self.chunk,
                                  stream_callback=self._audio_callback)
        self.stream.start_stream()
        self.status_signal.emit("录音中")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        音频数据回调函数 - 持续接收麦克风输入
        
        参数:
            in_data: 输入的音频数据（字节格式）
            frame_count: 帧数量
            time_info: 时间信息
            status: 状态标志
        
        返回:
            (in_data, pyaudio.paContinue): 继续录音
        """
        if self.recording:
            # 将字节数据转换为 numpy 数组并归一化
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_frames.append(audio_data)
        return (in_data, pyaudio.paContinue)

    def stop_recording_and_predict(self):
        """
        停止录音并进行识别
        
        工作流程：
        1. 停止录音并关闭音频流
        2. 能量检测：过滤静音片段
        3. 预处理音频（Mel频谱图 + 标准化）
        4. 模型推理
        5. 根据置信度决定是否执行动作
        """
        if not self.recording:
            return  # 如果没在录音，直接返回
        self.recording = False
        
        # 停止并关闭音频流
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if len(self.audio_frames) == 0:
            return  # 没有录音数据，直接返回
        
        self.status_signal.emit("识别中")
        
        # 合并所有音频帧
        full_audio = np.concatenate(self.audio_frames)
        target_len = RECORD_DURATION * self.sample_rate
        
        # 截断或填充到目标长度
        if len(full_audio) < target_len:
            pad = np.zeros(target_len - len(full_audio))
            audio = np.concatenate([full_audio, pad])
        else:
            audio = full_audio[-target_len:]  # 取最后一段
        
        # 能量检测：判断是否有声音输入
        energy = np.mean(audio**2)
        if energy < ENERGY_THRESHOLD:
            self.result_signal.emit("静音", 0.0)
            self.status_signal.emit("监听中")
            return
        
        # 转换为张量
        waveform = torch.from_numpy(audio).float().unsqueeze(0)
        
        # 计算 Mel 频谱图
        mel_spec = T.MelSpectrogram(sample_rate=self.sample_rate,
                                    n_mels=64, n_fft=1024, hop_length=512)
        mel = mel_spec(waveform)
        
        # 标准化
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        
        # 模型推理
        with torch.no_grad():
            outputs = self.model(input_values=mel.unsqueeze(0), return_dict=True)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)  # 转换为概率
            max_prob, pred_id = torch.max(probs, dim=1)
            confidence = max_prob.item()
            command = self.id2label[pred_id.item()]
        
        # 如果处于冷却状态，不执行动作
        if self.cooldown:
            self.result_signal.emit(f"(冷却中) {command}", confidence)
            self.status_signal.emit("监听中")
            return
        
        # 如果置信度达到阈值，执行动作
        if confidence >= CONFIDENCE_THRESHOLD:
            self.result_signal.emit(command, confidence)
            self.cooldown = True
            self.cooldown_timer.start(int(COOLDOWN_SECONDS * 1000))  # 启动冷却定时器
            self.status_signal.emit("执行中")
        else:
            self.result_signal.emit(f"低置信度: {command}", confidence)
            self.status_signal.emit("监听中")

    def _clear_cooldown(self):
        """冷却结束回调 - 重置冷却状态"""
        self.cooldown = False
        self.status_signal.emit("监听中")

    def run(self):
        """线程运行方法（占位，实际工作由其他方法处理）"""
        pass

# ==================== 3. 动作执行线程 ====================
class ActionExecutor(QThread):
    """
    动作执行线程 - 在后台执行系统操作
    
    支持的动作：
    - 动鼠标：鼠标画圆 + 锯齿移动
    - 截屏：保存当前屏幕截图
    - 打开计算器：启动系统计算器应用
    
    使用后台线程的原因：
    - 避免阻塞主 UI 线程
    - 保证界面响应流畅
    """
    finished = Signal()  # 动作完成信号
    
    def __init__(self, command):
        """
        初始化执行器
        
        参数:
            command: 要执行的命令名称
        """
        super().__init__()
        self.command = command
    
    def run(self):
        """线程运行方法 - 根据命令执行对应动作"""
        if self.command == "动鼠标":
            self.do_mouse_circle_and_zigzag()
        elif self.command == "截屏":
            self.do_screenshot()
        elif self.command == "打开计算器":
            self.open_calculator()
        self.finished.emit()  # 发出完成信号

    def do_mouse_circle_and_zigzag(self):
        """
        鼠标动画：先画圆，再做锯齿形移动
        
        效果：
        1. 从当前位置开始，画一个半径300像素的圆（0.5秒）
        2. 然后在垂直方向做锯齿形移动（0.5秒）
        """
        start_x, start_y = pyautogui.position()  # 获取鼠标当前位置
        radius = 300        # 圆的半径（像素）
        amplitude = 450     # 锯齿的振幅（像素）
        total_duration = 1.0  # 总持续时间（秒）
        
        # 第一阶段：画圆
        circle_duration = total_duration * 0.5  # 0.5秒
        steps = 30  # 圆的步数
        for i in range(steps + 1):
            t = i / steps
            angle = 2 * math.pi * t  # 角度（0 到 2π）
            x = start_x + radius * math.cos(angle)
            y = start_y + radius * math.sin(angle)
            pyautogui.moveTo(x, y, duration=circle_duration / steps)
        
        # 第二阶段：锯齿形移动
        vertical_duration = total_duration * 0.5  # 0.5秒
        steps_v = 20  # 每段的步数
        for _ in range(2):  # 往返2次
            # 向上移动
            for i in range(steps_v):
                y = start_y - amplitude * (i / steps_v)
                pyautogui.moveTo(start_x, y, duration=vertical_duration / steps_v / 2)
            # 向下移动
            for i in range(steps_v):
                y = start_y - amplitude * (1 - i / steps_v)
                pyautogui.moveTo(start_x, y, duration=vertical_duration / steps_v / 2)

    def do_screenshot(self):
        """
        截屏功能
        
        保存位置：当前目录下的 output_screen_capture 文件夹
        文件名格式：screenshot_YYYYMMDD_HHMMSS.png
        """
        save_dir = os.path.join(os.getcwd(), "output_screen_capture")
        os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        
        screenshot = ImageGrab.grab()  # 截取屏幕
        screenshot.save(filepath)  # 保存截图
        print(f"截图已保存: {filepath}")

    def open_calculator(self):
        """
        打开系统计算器应用
        
        支持多平台：
        - Windows: calc.exe
        - macOS: Calculator.app
        - Linux: gnome-calculator
        """
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

# ==================== 4. 主窗口 =========================
class MainWindow(QMainWindow):
    """
    主窗口 - 简洁的语音控制器界面
    
    界面布局：
    - 顶部：识别结果标签（持续显示，直到下一次新识别）
    - 中部：状态标签（显示当前状态：监听中/录音中/识别中/执行中）
    - 底部：操作提示
    
    交互方式：
    - 按住 Q 键：开始录音
    - 松开 Q 键：停止录音并识别
    """
    def __init__(self):
        """初始化主窗口"""
        super().__init__()
        self.setWindowTitle("语音控制器")
        self.setFixedSize(300, 180)  # 固定窗口大小
        self.setStyleSheet("background-color: #808080;")  # 灰色背景
        
        # 创建中心部件和布局
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)

        # 识别结果标签（持续显示）
        self.result_label = QLabel("等待指令\n按住Q说话")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 10))
        self.result_label.setStyleSheet("background-color: #C0C0C0; padding: 5px; border: 1px solid black;")
        layout.addWidget(self.result_label)

        # 状态标签
        self.status_label = QLabel("● 监听中")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 8))
        self.status_label.setStyleSheet("color: black; background-color: #A0A0A0; padding: 2px;")
        layout.addWidget(self.status_label)

        # 提示标签
        info_label = QLabel("按住 Q 录音，松开识别")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setFont(QFont("Arial", 7))
        info_label.setStyleSheet("color: #000000; background-color: #A0A0A0;")
        layout.addWidget(info_label)

        # 初始化录音器
        model_path = "/Volumes/AI/models/my_trained/model_speech_control_transformer"
        self.recorder = AudioRecorder(model_path)
        self.recorder.result_signal.connect(self.on_result)
        self.recorder.status_signal.connect(self.on_status)
        self.executor = None  # 动作执行器
        self.recording_active = False  # 是否正在录音

    def keyPressEvent(self, event):
        """
        键盘按下事件
        
        参数:
            event: 键盘事件对象
        """
        # 按下 Q 键且未在录音时，开始录音
        if event.key() == Qt.Key_Q and not self.recording_active:
            self.start_recording()

    def keyReleaseEvent(self, event):
        """
        键盘松开事件
        
        参数:
            event: 键盘事件对象
        """
        # 松开 Q 键且正在录音时，停止录音并识别
        if event.key() == Qt.Key_Q and self.recording_active:
            self.stop_recording_and_predict()

    def start_recording(self):
        """开始录音"""
        self.recording_active = True
        QApplication.beep()  # 发出提示音
        self.recorder.start_recording()

    def stop_recording_and_predict(self):
        """停止录音并识别"""
        if not self.recording_active:
            return
        self.recording_active = False
        self.recorder.stop_recording_and_predict()

    @Slot(str, float)
    def on_result(self, command, confidence):
        """
        处理识别结果
        
        参数:
            command: 识别到的命令名称
            confidence: 置信度（0-1）
        """
        # 显示识别结果，直到下一次新识别覆盖
        if command.startswith("低置信度") or command.startswith("冷却中") or command == "静音":
            # 异常情况：低置信度、冷却中、静音
            self.result_label.setText(f"{command}\n({confidence:.2f})")
        else:
            # 正常识别成功
            self.result_label.setText(f"识别: {command}\n置信度 {confidence:.2f}")
            
            # 检查是否有动作正在执行
            if self.executor and self.executor.isRunning():
                self.result_label.setText(f"动作冲突: {command}")
            else:
                # 创建并启动动作执行器
                self.executor = ActionExecutor(command)
                self.executor.finished.connect(self._on_action_finished)
                self.executor.start()

    @Slot(str)
    def on_status(self, status):
        """
        处理状态变化
        
        参数:
            status: 状态文本
        """
        # 根据不同状态设置不同的显示文本和样式
        if status == "监听中":
            self.status_label.setText("● 监听中")
            self.status_label.setStyleSheet("color: black; background-color: #A0A0A0; padding: 2px;")
        elif status == "录音中":
            self.status_label.setText("● 录音中 (松开Q)")
            self.status_label.setStyleSheet("color: black; background-color: #FFFF80; padding: 2px;")  # 黄色
        elif status == "识别中":
            self.status_label.setText("● 识别中...")
            self.status_label.setStyleSheet("color: black; background-color: #FFA500; padding: 2px;")  # 橙色
        elif status == "执行中":
            self.status_label.setText("● 执行动作中")
            self.status_label.setStyleSheet("color: black; background-color: #80FF80; padding: 2px;")  # 绿色

    def _on_action_finished(self):
        """动作执行完成后的回调"""
        # 动作执行完毕，恢复监听状态
        # 状态会由 recorder 的冷却结束信号更新
        pass

# ==================== 程序入口 ====================
if __name__ == "__main__":
    # 创建 Qt 应用实例
    app = QApplication(sys.argv)
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 启动应用事件循环
    sys.exit(app.exec())