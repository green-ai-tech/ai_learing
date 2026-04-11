#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局配置 - 路径、默认标签、样式表等常量
"""

import os


# ==================== 路径配置 ====================
DEFAULT_BASE_DIR = "/Users/logicye/Code/my_Datasets/系统控制"

# ==================== 音频配置 ====================
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = "paInt16"
AUDIO_CHUNK_SIZE = 1024
MIN_RECORDING_DURATION = 1.0
MAX_RECORDING_DURATION = 5.0
DEFAULT_RECORDING_DURATION = 1.0

# ==================== 默认标签 ====================
DEFAULT_LABELS = ["动鼠标", "截屏"]

# ==================== 文件过滤器 ====================
AUDIO_FILE_EXTENSION = ".wav"
TEXT_FILE_EXTENSION = ".txt"
SUPPORTED_FILE_EXTENSIONS = [f"*{AUDIO_FILE_EXTENSION}", f"*{TEXT_FILE_EXTENSION}"]

# ==================== 数据集划分 ====================
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# ==================== UI 配置 ====================
WINDOW_TITLE = "语音指令采集工具 - 自定义标签版"
WINDOW_MIN_WIDTH = 1200
WINDOW_MIN_HEIGHT = 700
SPLITTER_LEFT_SIZE = 400
SPLITTER_RIGHT_SIZE = 800
TREE_INDENTATION = 20
VOLUME_UPDATE_INTERVAL_MS = 100
VOLUME_DECAY_RATE = 5.0
WAVEFORM_DISPLAY_POINTS = 500
WAVEFORM_PEN_COLOR = "#5a8aba"
WAVEFORM_PEN_WIDTH = 2
VOLUME_BAR_MAX = 100

# ==================== 样式表 ====================
MAIN_WINDOW_STYLE = """
    QMainWindow { background-color: #2b2b2b; }
    QGroupBox {
        font-size: 14px; font-weight: bold;
        border: 2px solid #3c3c3c; border-radius: 8px;
        margin-top: 10px; padding-top: 10px;
        color: #e0e0e0;
    }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
    QLabel { color: #e0e0e0; font-size: 12px; }
    QPushButton {
        background-color: #4a4a4a; color: white;
        border: none; padding: 8px 16px; border-radius: 5px;
        font-size: 12px; font-weight: bold;
    }
    QPushButton:hover { background-color: #5a5a5a; }
    QPushButton:pressed { background-color: #3a3a3a; }
    QLineEdit, QComboBox {
        background-color: #3c3c3c; color: #e0e0e0;
        border: 1px solid #5a5a5a; border-radius: 4px;
        padding: 5px; font-size: 12px;
    }
    QTreeView {
        background-color: #3c3c3c; color: #e0e0e0;
        border: 1px solid #5a5a5a; border-radius: 4px;
        font-size: 11px;
    }
    QTreeView::item:hover { background-color: #4a6a8a; }
    QTreeView::item:selected { background-color: #5a8aba; }
    QProgressBar {
        border: 1px solid #5a5a5a; border-radius: 3px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: qlineargradient(x1:0,y1:0,x2:1,y2:0,
            stop:0 #4caf50, stop:0.5 #ffc107, stop:1 #f44336);
        border-radius: 2px;
    }
"""

RECORD_BUTTON_STYLE = "background-color: #2196f3; font-size: 14px; padding: 10px;"
STATUS_IDLE_STYLE = "color: #4caf50; font-size: 14px; font-weight: bold;"
STATUS_RECORDING_STYLE = "color: #f44336; font-size: 14px; font-weight: bold;"
PATH_LABEL_STYLE = "color: #888; font-size: 10px; padding: 5px;"

# ==================== 状态文本 ====================
STATUS_IDLE_TEXT = "● 休闲中"
STATUS_RECORDING_TEXT = "● 录音中"
BUTTON_RECORD_TEXT = "🎤 语音采集"
BUTTON_RECORDING_TEXT = "🔴 录音中..."

# ==================== 导出配置 ====================
EXPORT_FILE_FILTER = "ZIP文件 (*.zip)"
EXPORT_FILENAME_PREFIX = "dataset_"
