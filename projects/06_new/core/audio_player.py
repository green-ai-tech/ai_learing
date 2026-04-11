#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频播放线程 - 负责播放 WAV 文件
"""

import os
import wave

import pyaudio
from PySide6.QtCore import QThread, Signal


class AudioPlayer(QThread):
    """音频播放线程类"""

    play_finished = Signal()

    def __init__(self, filepath: str) -> None:
        super().__init__()
        self.filepath = filepath

    def run(self) -> None:
        """播放音频文件"""
        try:
            wf = wave.open(self.filepath, "rb")
            p = pyaudio.PyAudio()
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
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
