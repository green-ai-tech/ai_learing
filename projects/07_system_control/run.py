#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音控制器 - 主程序入口

通过语音指令控制系统操作（移动鼠标、截屏、打开计算器等）

工作流程：
1. 用户按住 Q 键说话
2. 系统录制音频并进行实时识别
3. 识别成功后执行对应的系统操作
4. 通过 GUI 显示识别结果和状态

作者：Logic Ye
日期：2026-04-04

设计理念：
- 面向对象：每个组件职责单一清晰
- 分层架构：配置层 -> 模型层 -> 服务层 -> 表现层
- 依赖注入：组件之间通过接口交互
- 代码艺术：结构清晰，注释详细，易于维护

使用示例：
    python run.py
"""

import sys
from PySide6.QtWidgets import QApplication

from config.settings import AppConfig
from services.audio_recorder import AudioRecorder
from ui.main_window import MainWindow


def create_app() -> QApplication:
    """
    创建 Qt 应用实例
    
    返回:
        QApplication 实例
    """
    return QApplication(sys.argv)


def create_recorder() -> AudioRecorder:
    """
    创建音频录音器实例
    
    返回:
        AudioRecorder 实例
    """
    return AudioRecorder(model_path=AppConfig.MODEL_PATH)


def create_window(recorder: AudioRecorder) -> MainWindow:
    """
    创建主窗口实例
    
    参数:
        recorder: 音频录音器实例
    
    返回:
        MainWindow 实例
    """
    return MainWindow(recorder)


def main():
    """主函数 - 应用启动入口"""
    # 创建应用
    app = create_app()
    
    # 创建服务
    recorder = create_recorder()
    recorder.start()  # 启动后台线程
    
    # 创建并显示主窗口
    window = create_window(recorder)
    window.show()
    
    # 启动应用事件循环
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
