"""
文件列表面板组件
"""
import os
from datetime import datetime
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QGroupBox, QMessageBox
)
from PySide6.QtCore import Qt

from services.player_service import AudioPlayer


class FileListPanel(QWidget):
    """音频文件列表面板"""

    def __init__(self, label: str, base_dir: str, parent=None):
        super().__init__(parent)
        self.label = label
        self.base_dir = base_dir
        self.audio_player: Optional[AudioPlayer] = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 文件列表组
        group = QGroupBox(f"{self.label}指令")
        group_layout = QVBoxLayout(group)

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        group_layout.addWidget(self.list_widget)

        # 统计信息标签
        self.stats_label = QLabel("音频数量: 0 | 最近更新: --")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        group_layout.addWidget(self.stats_label)

        layout.addWidget(group)

    def refresh(self, files: list, latest_time: Optional[datetime] = None):
        """刷新文件列表"""
        self.list_widget.clear()
        for filename in files:
            self.list_widget.addItem(filename)

        time_str = latest_time.strftime("%Y-%m-%d %H:%M:%S") if latest_time else "--"
        self.stats_label.setText(f"音频数量: {len(files)} | 最近更新: {time_str}")

    def _on_item_double_clicked(self, item: QListWidgetItem):
        """双击播放音频"""
        if not item:
            return

        filename = item.text()
        filepath = os.path.join(self.base_dir, self.label, filename)

        if not os.path.exists(filepath):
            QMessageBox.warning(self, "警告", "音频文件不存在")
            return

        # 停止当前播放
        if self.audio_player and self.audio_player.isRunning():
            self.audio_player.terminate()

        self.audio_player = AudioPlayer(filepath)
        self.audio_player.start()
