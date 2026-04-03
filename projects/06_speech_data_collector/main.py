#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音指令采集工具
基于 Python + PySide6 实现
用于采集向上/向下指令的音频数据
"""
import sys

from PySide6.QtWidgets import QApplication

from main_window import VoiceCommandCollector


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    window = VoiceCommandCollector()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
