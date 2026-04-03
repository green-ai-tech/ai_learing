"""
波形显示组件
"""
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar


class WaveformWidget(QWidget):
    """实时音频波形显示组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._current_volume = 0.0

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 创建波形图
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#2b2b2b')
        self.plot_widget.setLabel('left', '振幅')
        self.plot_widget.setLabel('bottom', '采样点')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setYRange(-32768, 32768)
        self.curve = self.plot_widget.plot(pen=pg.mkPen(color='#5a8aba', width=2))
        layout.addWidget(self.plot_widget)

        # 音量显示条
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("音量:"))
        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(0, 100)
        volume_layout.addWidget(self.volume_bar)
        layout.addLayout(volume_layout)

    def update_waveform(self, data: np.ndarray):
        """更新波形显示"""
        try:
            if data is None or len(data) == 0:
                return
            self.curve.setData(data)
        except Exception as e:
            print(f"波形更新错误: {e}")

    def update_volume(self, level: float):
        """更新音量级别"""
        try:
            if np.isnan(level) or np.isinf(level):
                self._current_volume = 0.0
            else:
                self._current_volume = float(level)
        except Exception:
            self._current_volume = 0.0

    def get_volume(self) -> float:
        """获取当前音量"""
        return self._current_volume

    def decay_volume(self, decay_rate: float = 5.0):
        """音量衰减"""
        self._current_volume = max(0.0, self._current_volume - decay_rate)
