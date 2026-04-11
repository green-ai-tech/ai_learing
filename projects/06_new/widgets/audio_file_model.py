#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频文件系统模型 - 自定义 QFileSystemModel，仅显示 wav 和 txt 文件
"""

from PySide6.QtCore import QDir, QModelIndex
from PySide6.QtWidgets import QFileSystemModel

from config import SUPPORTED_FILE_EXTENSIONS


class AudioFileSystemModel(QFileSystemModel):
    """自定义文件系统模型，过滤显示音频和文本文件"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFilter(
            QDir.Filter.AllDirs
            | QDir.Filter.NoDotAndDotDot
            | QDir.Filter.Files
        )
        self.setNameFilters(SUPPORTED_FILE_EXTENSIONS)
        self.setNameFilterDisables(False)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 1
