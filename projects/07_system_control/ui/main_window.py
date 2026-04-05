#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主窗口 - 语音控制器图形界面

作者：Logic Ye
日期：2026-04-04
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from config.settings import AppConfig
from services.audio_recorder import AudioRecorder
from services.action_executor import ActionExecutor


class MainWindow(QMainWindow):
    """
    主窗口 - 简洁的语音控制器界面
    
    界面布局（从上到下）：
    1. 识别结果标签：持续显示，直到下一次新识别覆盖
    2. 状态标签：显示当前状态（监听中/录音中/识别中/执行中）
    3. 操作提示：简单说明使用方法
    
    交互方式：
    - 按住 Q 键：开始录音
    - 松开 Q 键：停止录音并识别
    - 识别成功后自动执行对应动作
    
    设计原则：
    - 职责单一：只负责 UI 展示和用户交互
    - 依赖注入：通过构造函数接收服务实例
    - 信号槽机制：与服务层解耦
    """
    
    def __init__(self, recorder: AudioRecorder):
        """
        初始化主窗口
        
        参数:
            recorder: 音频录音器实例
        """
        super().__init__()
        self.recorder = recorder
        self.executor = None
        self.recording_active = False
        
        # 连接信号
        self.recorder.result_signal.connect(self._on_result)
        self.recorder.status_signal.connect(self._on_status)
        
        # 初始化 UI
        self._setup_ui()

    def _setup_ui(self):
        """设置用户界面"""
        self.setWindowTitle("语音控制器")
        self.setFixedSize(AppConfig.WINDOW_WIDTH, AppConfig.WINDOW_HEIGHT)
        self.setStyleSheet(f"background-color: {AppConfig.BG_COLOR};")

        # 创建中心部件
        central = QWidget()
        self.setCentralWidget(central)
        
        # 垂直布局
        layout = QVBoxLayout(central)
        layout.setSpacing(5)
        layout.setContentsMargins(10, 10, 10, 10)

        # 识别结果标签
        self.result_label = self._create_label(
            text="等待指令\n按住Q说话",
            font_size=AppConfig.RESULT_FONT_SIZE,
            bg_color=AppConfig.RESULT_BG_COLOR,
            border=True
        )
        layout.addWidget(self.result_label)

        # 状态标签
        self.status_label = self._create_label(
            text="● 监听中",
            font_size=AppConfig.STATUS_FONT_SIZE,
            bg_color=AppConfig.STATUS_COLORS["监听中"]
        )
        layout.addWidget(self.status_label)

        # 提示标签
        info_label = self._create_label(
            text="按住 Q 录音，松开识别",
            font_size=AppConfig.INFO_FONT_SIZE,
            bg_color=AppConfig.STATUS_COLORS["监听中"]
        )
        layout.addWidget(info_label)

    def _create_label(self, text: str, font_size: int, 
                      bg_color: str, border: bool = False) -> QLabel:
        """
        创建标签的辅助方法
        
        参数:
            text: 标签文本
            font_size: 字体大小
            bg_color: 背景颜色
            border: 是否显示边框
        
        返回:
            QLabel 实例
        """
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont(AppConfig.FONT_FAMILY, font_size))
        
        if border:
            label.setStyleSheet(
                f"background-color: {bg_color}; "
                f"padding: 5px; border: 1px solid black;"
            )
        else:
            label.setStyleSheet(
                f"color: black; background-color: {bg_color}; padding: 2px;"
            )
        
        return label

    def keyPressEvent(self, event):
        """
        键盘按下事件
        
        参数:
            event: 键盘事件对象
        """
        if event.key() == Qt.Key_Q and not self.recording_active:
            self._start_recording()

    def keyReleaseEvent(self, event):
        """
        键盘松开事件
        
        参数:
            event: 键盘事件对象
        """
        if event.key() == Qt.Key_Q and self.recording_active:
            self._stop_recording_and_predict()

    def _start_recording(self):
        """开始录音"""
        self.recording_active = True
        QApplication.beep()  # 发出提示音
        self.recorder.start_recording()

    def _stop_recording_and_predict(self):
        """停止录音并识别"""
        if not self.recording_active:
            return
        self.recording_active = False
        self.recorder.stop_recording_and_predict()

    @Slot(str, float)
    def _on_result(self, command: str, confidence: float):
        """
        处理识别结果
        
        参数:
            command: 识别到的命令名称
            confidence: 置信度（0-1）
        """
        # 异常情况：低置信度、冷却中、静音
        if (command.startswith("低置信度") or 
            command.startswith("冷却中") or 
            command == "静音"):
            self.result_label.setText(f"{command}\n({confidence:.2f})")
            return

        # 正常识别成功
        self.result_label.setText(f"识别: {command}\n置信度 {confidence:.2f}")

        # 检查是否有动作正在执行
        if self.executor and self.executor.isRunning():
            self.result_label.setText(f"动作冲突: {command}")
            return

        # 创建并启动动作执行器
        self.executor = ActionExecutor(command)
        self.executor.finished.connect(self._on_action_finished)
        self.executor.start()

    @Slot(str)
    def _on_status(self, status: str):
        """
        处理状态变化
        
        参数:
            status: 状态文本
        """
        self.status_label.setText(f"● {status}")
        
        # 获取对应颜色
        color = AppConfig.STATUS_COLORS.get(status, AppConfig.STATUS_COLORS["监听中"])
        self.status_label.setStyleSheet(
            f"color: black; background-color: {color}; padding: 2px;"
        )

    def _on_action_finished(self):
        """动作执行完成后的回调"""
        # 状态会由 recorder 的冷却结束信号更新
        pass

    def closeEvent(self, event):
        """
        窗口关闭事件 - 清理资源
        
        参数:
            event: 关闭事件对象
        """
        self.recorder.cleanup()
        event.accept()
