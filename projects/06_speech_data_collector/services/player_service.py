"""
音频播放服务
"""
import wave

import pyaudio
from PySide6.QtCore import QThread, Signal


class AudioPlayer(QThread):
    """音频播放线程"""

    play_finished = Signal()

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def run(self):
        """播放音频"""
        try:
            # 打开WAV文件
            wf = wave.open(self.filepath, 'rb')

            # 初始化PyAudio
            p = pyaudio.PyAudio()

            # 打开音频流
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )

            # 播放数据
            data = wf.readframes(1024)
            while data:
                stream.write(data)
                data = wf.readframes(1024)

            # 清理
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()

        except Exception as e:
            print(f"播放错误: {e}")
        finally:
            self.play_finished.emit()
