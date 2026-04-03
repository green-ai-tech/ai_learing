"""
采集控制面板组件
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QLineEdit, QPushButton, QGroupBox
)


class IntValidator:
    """整数验证器"""
    def __init__(self, min_val: int, max_val: int):
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, text: str, pos: int) -> bool:
        try:
            if not text:
                return True
            val = int(text)
            if self.min_val <= val <= self.max_val:
                return True
        except ValueError:
            pass
        return False


class ControlPanel(QWidget):
    """采集控制面板"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 配置组
        config_group = QGroupBox("采集配置")
        config_layout = QVBoxLayout(config_group)

        # 采集时间行
        time_row_layout = QHBoxLayout()
        time_row_layout.addWidget(QLabel("采集时间:"))
        self.time_input = QLineEdit("1")
        self.time_input.setFixedWidth(60)
        self.time_validator = IntValidator(1, 5)
        self.time_input.textChanged.connect(self._validate_time_input)
        time_row_layout.addWidget(self.time_input)
        time_row_layout.addWidget(QLabel("秒"))
        time_row_layout.addStretch()

        # 快捷按钮行
        shortcut_layout = QHBoxLayout()
        shortcut_layout.addWidget(QLabel("快捷:"))

        self.btn_1s = QPushButton("1秒")
        self.btn_1s.setProperty("shortcut", "true")
        self.btn_1s.clicked.connect(lambda: self.set_duration(1))

        self.btn_2s = QPushButton("2秒")
        self.btn_2s.setProperty("shortcut", "true")
        self.btn_2s.clicked.connect(lambda: self.set_duration(2))

        self.btn_5s = QPushButton("5秒")
        self.btn_5s.setProperty("shortcut", "true")
        self.btn_5s.clicked.connect(lambda: self.set_duration(5))

        shortcut_layout.addWidget(self.btn_1s)
        shortcut_layout.addWidget(self.btn_2s)
        shortcut_layout.addWidget(self.btn_5s)
        shortcut_layout.addStretch()

        # 指令标签行
        label_layout = QHBoxLayout()
        label_layout.addWidget(QLabel("指令标签:"))
        self.label_combo = QComboBox()
        self.label_combo.addItems(["向上", "向下"])
        self.label_combo.setFixedWidth(100)
        label_layout.addWidget(self.label_combo)
        label_layout.addStretch()

        config_layout.addLayout(time_row_layout)
        config_layout.addLayout(shortcut_layout)
        config_layout.addLayout(label_layout)
        layout.addWidget(config_group)

        # 状态灯
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("状态:"))
        self.status_light = QLabel("● 休闲中")
        self.status_light.setStyleSheet("color: #4caf50; font-size: 14px; font-weight: bold;")
        status_layout.addWidget(self.status_light)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # 功能按钮
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.record_btn = QPushButton("🎤 语音采集")
        self.record_btn.setStyleSheet("background-color: #2196f3; font-size: 14px; padding: 10px;")

        self.update_btn = QPushButton("📊 更新数据集")
        self.export_btn = QPushButton("💾 导出数据集")

        button_layout.addWidget(self.record_btn)
        button_layout.addWidget(self.update_btn)
        button_layout.addWidget(self.export_btn)
        layout.addLayout(button_layout)

        # 路径标签
        self.path_label = QLabel()
        self.path_label.setStyleSheet("color: #888; font-size: 10px; padding: 5px;")
        self.path_label.setWordWrap(True)
        layout.addWidget(self.path_label)

    def set_duration(self, seconds: int):
        """设置录制时长"""
        self.time_input.setText(str(seconds))

    def get_duration(self) -> str:
        """获取录制时长"""
        return self.time_input.text()

    def get_label(self) -> str:
        """获取指令标签"""
        return self.label_combo.currentText()

    def _validate_time_input(self, text: str):
        """验证输入时间"""
        if text and not self.time_validator.validate(text, 0):
            if text:
                try:
                    val = int(text)
                    if val < 1:
                        self.time_input.setText("1")
                    elif val > 5:
                        self.time_input.setText("5")
                except ValueError:
                    self.time_input.setText("1")

    def set_recording_status(self, is_recording: bool):
        """设置录制状态"""
        if is_recording:
            self.status_light.setText("● 录音中")
            self.status_light.setStyleSheet("color: #f44336; font-size: 14px; font-weight: bold;")
            self.record_btn.setEnabled(False)
            self.record_btn.setText("🔴 录音中...")
        else:
            self.status_light.setText("● 休闲中")
            self.status_light.setStyleSheet("color: #4caf50; font-size: 14px; font-weight: bold;")
            self.record_btn.setEnabled(True)
            self.record_btn.setText("🎤 语音采集")

    def set_dataset_path(self, path: str):
        """设置数据集路径显示"""
        self.path_label.setText(f"数据集路径: {path}")
