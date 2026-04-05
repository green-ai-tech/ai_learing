#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频录制服务 - 负责录音和实时语音识别

作者：Logic Ye
日期：2026-04-04
"""

import os
import json

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import pyaudio
from PySide6.QtCore import QThread, Signal, QTimer

from config.settings import AppConfig
from models.speech_model import SimpleCNNForAudioClassification


class AudioRecorder(QThread):
    """
    音频录制服务 - 负责录音和实时识别
    
    工作流程：
    1. 调用 start_recording() 开始录音
    2. 通过回调函数持续读取音频数据
    3. 调用 stop_recording_and_predict() 停止录音并识别
    4. 识别结果通过信号发送给监听者
    
    核心特性：
    - 能量检测：自动过滤静音片段
    - 冷却机制：防止短时间内重复触发
    - 线程安全：在后台线程中运行，不阻塞 UI
    
    信号：
        result_signal: 识别结果信号 (command: str, confidence: float)
        status_signal: 状态变化信号 (status: str)
    
    使用示例：
        >>> recorder = AudioRecorder(model_path)
        >>> recorder.result_signal.connect(on_result)
        >>> recorder.status_signal.connect(on_status)
        >>> recorder.start()
        >>> recorder.start_recording()
        >>> # ... 用户说话 ...
        >>> recorder.stop_recording_and_predict()
    """
    result_signal = Signal(str, float)   # (指令, 置信度)
    status_signal = Signal(str)          # 状态变化信号

    def __init__(self, model_path: str, sample_rate: int = AppConfig.SAMPLE_RATE):
        """
        初始化录音器
        
        参数:
            model_path: 模型保存路径
            sample_rate: 音频采样率（Hz）
        """
        super().__init__()
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.chunk = AppConfig.AUDIO_CHUNK_SIZE
        
        # 录音状态
        self.recording = False
        self.audio_frames = []
        
        # PyAudio 实例
        self.p = pyaudio.PyAudio()
        self.stream = None
        
        # 冷却机制
        self.cooldown = False
        self.cooldown_timer = QTimer()
        self.cooldown_timer.setSingleShot(True)
        self.cooldown_timer.timeout.connect(self._clear_cooldown)
        
        # 启用状态
        self.enabled = True
        
        # 加载模型
        self._load_model()
        self._init_mel_transform()

    def _load_model(self):
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

    def _init_mel_transform(self):
        """初始化 Mel 频谱图转换器"""
        self.mel_spec = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=AppConfig.N_MELS,
            n_fft=AppConfig.N_FFT,
            hop_length=AppConfig.HOP_LENGTH
        )

    def start_recording(self):
        """开始录音"""
        if self.recording:
            return  # 如果正在录音，忽略重复请求
        
        self.recording = True
        self.audio_frames = []  # 清空之前的录音
        
        # 打开音频输入流
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            stream_callback=self._audio_callback
        )
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
        """停止录音并进行识别"""
        if not self.recording:
            return
        
        self.recording = False

        # 停止并关闭音频流
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        if len(self.audio_frames) == 0:
            return  # 没有录音数据

        self.status_signal.emit("识别中")

        # 合并所有音频帧
        full_audio = np.concatenate(self.audio_frames)
        target_len = AppConfig.RECORD_DURATION * self.sample_rate

        # 截断或填充到目标长度
        if len(full_audio) < target_len:
            pad = np.zeros(target_len - len(full_audio))
            audio = np.concatenate([full_audio, pad])
        else:
            audio = full_audio[-target_len:]

        # 能量检测：判断是否有声音输入
        energy = np.mean(audio ** 2)
        if energy < AppConfig.ENERGY_THRESHOLD:
            self.result_signal.emit("静音", 0.0)
            self.status_signal.emit("监听中")
            return

        # 执行识别
        self._predict(audio)

    def _predict(self, audio: np.ndarray):
        """
        对音频进行识别
        
        参数:
            audio: 音频数据（numpy 数组）
        """
        # 转换为张量
        waveform = torch.from_numpy(audio).float().unsqueeze(0)

        # 计算 Mel 频谱图
        mel = self.mel_spec(waveform)

        # 标准化
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)

        # 模型推理
        with torch.no_grad():
            outputs = self.model(input_values=mel.unsqueeze(0), return_dict=True)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=1)
            max_prob, pred_id = torch.max(probs, dim=1)
            confidence = max_prob.item()
            command = self.id2label[pred_id.item()]

        # 如果处于冷却状态，不执行动作
        if self.cooldown:
            self.result_signal.emit(f"(冷却中) {command}", confidence)
            self.status_signal.emit("监听中")
            return

        # 如果置信度达到阈值，执行动作
        if confidence >= AppConfig.CONFIDENCE_THRESHOLD:
            self.result_signal.emit(command, confidence)
            self.cooldown = True
            self.cooldown_timer.start(int(AppConfig.COOLDOWN_SECONDS * 1000))
            self.status_signal.emit("执行中")
        else:
            self.result_signal.emit(f"低置信度: {command}", confidence)
            self.status_signal.emit("监听中")

    def _clear_cooldown(self):
        """冷却结束回调 - 重置冷却状态"""
        self.cooldown = False
        self.status_signal.emit("监听中")

    def cleanup(self):
        """清理资源"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

    def run(self):
        """线程运行方法（占位，实际工作由其他方法处理）"""
        pass
