#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频录制线程 - 负责录音、波形数据发射、音量检测
"""

import os
import wave
from datetime import datetime
from typing import List, Optional

import numpy as np
import pyaudio
from PySide6.QtCore import QThread, Signal

from config import (
    AUDIO_CHANNELS,
    AUDIO_CHUNK_SIZE,
    AUDIO_FORMAT,
    AUDIO_SAMPLE_RATE,
)


class AudioRecorderThread(QThread):
    """音频录制线程类"""

    waveform_data = Signal(np.ndarray)
    recording_finished = Signal(str)
    recording_error = Signal(str)
    volume_level = Signal(float)

    def __init__(self, duration: float, label: str, save_dir: str) -> None:
        super().__init__()
        self.duration = duration
        self.label = label
        self.save_dir = save_dir
        self.is_recording = True
        self.audio_format = getattr(pyaudio, AUDIO_FORMAT)
        self.channels = AUDIO_CHANNELS
        self.rate = AUDIO_SAMPLE_RATE
        self.chunk = AUDIO_CHUNK_SIZE

    def run(self) -> None:
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
                frames_per_buffer=self.chunk,
            )

            frames: List[bytes] = []
            frames_per_second = int(self.rate / self.chunk)
            total_frames = int(self.duration * frames_per_second)

            for i in range(total_frames):
                if not self.is_recording:
                    break
                data = stream.read(self.chunk, exception_on_overflow=False)
                frames.append(data)

                # Calculate volume level
                try:
                    audio_data = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                    if len(audio_data) > 0:
                        rms = np.sqrt(np.mean(audio_data**2))
                        if np.isnan(rms) or np.isinf(rms):
                            volume = 0
                        else:
                            volume = min(100, int((rms / 32768) * 100))
                    else:
                        volume = 0
                except Exception:
                    volume = 0
                self.volume_level.emit(volume)

                # Emit waveform data
                try:
                    if len(audio_data) > 500:
                        display_data = audio_data[:: len(audio_data) // 500]
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
        """保存音频帧到 WAV 文件"""
        label_dir = os.path.join(self.save_dir, self.label)
        os.makedirs(label_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{self.label}_{timestamp}.wav"
        filepath = os.path.join(label_dir, filename)
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(frames))
        return filepath

    def stop_recording(self) -> None:
        """停止录制"""
        self.is_recording = False
