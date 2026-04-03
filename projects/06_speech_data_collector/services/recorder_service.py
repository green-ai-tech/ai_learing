"""
音频录制服务
"""
import os
import wave
from datetime import datetime
from typing import List, Optional

import numpy as np
import pyaudio
from PySide6.QtCore import QThread, Signal


class AudioRecorderThread(QThread):
    """音频录制线程，避免阻塞UI"""

    # 定义信号
    waveform_data = Signal(np.ndarray)  # 波形数据
    recording_finished = Signal(str)    # 录制完成，返回保存路径
    recording_error = Signal(str)       # 录制错误
    volume_level = Signal(float)        # 音量级别

    def __init__(self, duration: float, label: str, save_dir: str):
        super().__init__()
        self.duration = duration
        self.label = label
        self.save_dir = save_dir
        self.is_recording = True
        self.audio_format = pyaudio.paInt16  # 16-bit PCM
        self.channels = 1                    # 单声道
        self.rate = 16000                    # 采样率 16kHz
        self.chunk = 1024                    # 每次读取的帧大小

    def run(self):
        """录制音频的主逻辑"""
        try:
            # 初始化 PyAudio
            p = pyaudio.PyAudio()

            # 检查是否有可用的输入设备
            if p.get_device_count() == 0:
                self.recording_error.emit("未找到音频设备")
                p.terminate()
                return

            # 打开音频流
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

                # 读取音频数据
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)

                # 计算音量级别（0-100）- 修复 NaN 问题
                try:
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    if len(audio_data) > 0:
                        # 计算均方根（RMS）
                        rms = np.sqrt(np.mean(audio_data**2))
                        # 避免 NaN 和无穷大
                        if np.isnan(rms) or np.isinf(rms):
                            volume_level = 0
                        else:
                            # 映射到 0-100 范围（16位音频最大值32768）
                            volume_level = min(100, int((rms / 32768) * 100))
                    else:
                        volume_level = 0
                except Exception:
                    volume_level = 0

                self.volume_level.emit(volume_level)

                # 发送波形数据（用于实时显示）
                try:
                    # 降采样显示
                    if len(audio_data) > 500:
                        display_data = audio_data[::len(audio_data)//500]
                    else:
                        display_data = audio_data
                    self.waveform_data.emit(display_data)
                except Exception:
                    # 如果波形数据有问题，发送空数组
                    self.waveform_data.emit(np.zeros(100))

                # 控制录制速度
                self.msleep(int(self.duration * 1000 / total_frames))

            # 停止和关闭流
            stream.stop_stream()
            stream.close()
            p.terminate()

            # 保存音频文件
            if frames and self.is_recording:
                save_path = self.save_audio(frames)
                self.recording_finished.emit(save_path)
            else:
                self.recording_error.emit("录制被取消")

        except Exception as e:
            self.recording_error.emit(f"录制错误: {str(e)}")

    def save_audio(self, frames: List[bytes]) -> str:
        """保存音频为WAV文件"""
        # 创建标签对应的文件夹
        label_dir = os.path.join(self.save_dir, self.label)
        os.makedirs(label_dir, exist_ok=True)

        # 生成文件名：标签_时间戳.wav
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{self.label}_{timestamp}.wav"
        filepath = os.path.join(label_dir, filename)

        # 保存WAV文件
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))

        return filepath

    def stop_recording(self):
        """停止录制"""
        self.is_recording = False
