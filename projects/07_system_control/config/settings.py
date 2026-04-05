#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用配置 - 集中管理所有配置参数

作者：Logic Ye
日期：2026-04-04
"""

import os


class AppConfig:
    """
    应用配置类 - 集中管理所有常量参数
    
    设计原则：
    - 使用类属性而非全局变量，便于管理和修改
    - 按功能分组，提高可读性
    """
    
    # ==================== 模型配置 ====================
    # 模型保存路径（训练好的语音识别模型）
    MODEL_PATH = "/Users/logicye/Code/my_trained_models/model_speech_control_transformer"
    
    # 音频采样率（Hz）- 与训练时保持一致
    SAMPLE_RATE = 16000
    
    # ==================== 识别配置 ====================
    # 置信度阈值：预测概率超过此值才认为识别成功
    CONFIDENCE_THRESHOLD = 0.35
    
    # 能量检测阈值：用于判断是否有声音输入
    ENERGY_THRESHOLD = 0.001
    
    # 录音长度（秒）- 与模型训练时输入长度匹配
    RECORD_DURATION = 1
    
    # 动作执行后冷却时间（秒）- 防止重复触发
    COOLDOWN_SECONDS = 0.5
    
    # ==================== 音频配置 ====================
    # 音频块大小（每次读取的采样点数）
    AUDIO_CHUNK_SIZE = 1024
    
    # Mel 频谱图参数
    N_MELS = 64          # Mel 滤波器数量
    N_FFT = 1024         # FFT 窗口大小
    HOP_LENGTH = 512     # 帧移
    
    # ==================== UI 配置 ====================
    # 窗口尺寸
    WINDOW_WIDTH = 300
    WINDOW_HEIGHT = 180
    
    # 字体设置
    FONT_FAMILY = "Arial"
    RESULT_FONT_SIZE = 10
    STATUS_FONT_SIZE = 8
    INFO_FONT_SIZE = 7
    
    # 颜色配置
    BG_COLOR = "#808080"              # 背景色（灰色）
    RESULT_BG_COLOR = "#C0C0C0"       # 结果标签背景色
    STATUS_COLORS = {
        "监听中": "#A0A0A0",          # 灰色
        "录音中": "#FFFF80",          # 黄色
        "识别中": "#FFA500",          # 橙色
        "执行中": "#80FF80",          # 绿色
    }
    
    # ==================== 动作配置 ====================
    # 鼠标动画参数
    MOUSE_CIRCLE_RADIUS         = 300         # 圆的半径（像素）
    MOUSE_ZIGZAG_AMPLITUDE      = 450      # 锯齿的振幅（像素）
    MOUSE_ANIMATION_DURATION    = 1.0    # 总持续时间（秒）
    
    # 截屏保存目录
    SCREENSHOT_DIR = "output_screen_capture"
