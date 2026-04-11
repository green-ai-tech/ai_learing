#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音指令采集工具 - 程序入口
"""

import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import VoiceCommandCollector


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = VoiceCommandCollector()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
