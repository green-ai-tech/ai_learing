#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动作执行器 - 执行系统操作（鼠标、截屏、计算器）

作者：Logic Ye
日期：2026-04-04
"""

import os
import sys
import math
import subprocess
from datetime import datetime

import pyautogui
from PIL import ImageGrab
from PySide6.QtCore import QThread, Signal

from config.settings import AppConfig


class ActionExecutor(QThread):
    """
    动作执行线程 - 在后台执行系统操作，避免阻塞 UI
    
    支持的动作：
    - 动鼠标：鼠标画圆 + 锯齿形移动
    - 截屏：保存当前屏幕截图
    - 打开计算器：启动系统计算器应用
    
    设计原则：
    - 使用后台线程执行耗时操作
    - 通过信号通知执行完成
    - 每个动作封装为独立方法，便于扩展
    
    信号：
        finished: 动作完成信号
    
    使用示例：
        >>> executor = ActionExecutor("截屏")
        >>> executor.finished.connect(on_done)
        >>> executor.start()
    """
    finished = Signal()  # 动作完成信号

    def __init__(self, command: str):
        """
        初始化执行器
        
        参数:
            command: 要执行的命令名称
        """
        super().__init__()
        self.command = command

    def run(self):
        """线程运行方法 - 根据命令执行对应动作"""
        if self.command == "动鼠标":
            self._do_mouse_animation()
        elif self.command == "截屏":
            self._do_screenshot()
        elif self.command == "打开计算器":
            self._open_calculator()
        
        # 发出完成信号
        self.finished.emit()

    def _do_mouse_animation(self):
        """
        鼠标动画：先画圆，再做锯齿形移动
        
        效果：
        1. 从当前位置开始，画一个半径 300 像素的圆（0.5 秒）
        2. 然后在垂直方向做锯齿形移动（0.5 秒）
        """
        start_x, start_y = pyautogui.position()
        radius = AppConfig.MOUSE_CIRCLE_RADIUS
        amplitude = AppConfig.MOUSE_ZIGZAG_AMPLITUDE
        total_duration = AppConfig.MOUSE_ANIMATION_DURATION
        
        # 第一阶段：画圆
        self._draw_circle(start_x, start_y, radius, total_duration * 0.5)
        
        # 第二阶段：锯齿形移动
        self._draw_zigzag(start_x, start_y, amplitude, total_duration * 0.5)

    def _draw_circle(self, center_x, center_y, radius, duration):
        """
        画圆
        
        参数:
            center_x, center_y: 圆心坐标
            radius: 半径
            duration: 持续时间（秒）
        """
        steps = 30
        for i in range(steps + 1):
            t = i / steps
            angle = 2 * math.pi * t
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            pyautogui.moveTo(x, y, duration=duration / steps)

    def _draw_zigzag(self, center_x, center_y, amplitude, duration):
        """
        锯齿形移动
        
        参数:
            center_x, center_y: 中心坐标
            amplitude: 振幅
            duration: 持续时间（秒）
        """
        steps = 20
        for _ in range(2):  # 往返 2 次
            # 向上移动
            for i in range(steps):
                y = center_y - amplitude * (i / steps)
                pyautogui.moveTo(center_x, y, duration=duration / steps / 2)
            # 向下移动
            for i in range(steps):
                y = center_y - amplitude * (1 - i / steps)
                pyautogui.moveTo(center_x, y, duration=duration / steps / 2)

    def _do_screenshot(self):
        """
        截屏功能
        
        保存位置：当前目录下的 output_screen_capture 文件夹
        文件名格式：screenshot_YYYYMMDD_HHMMSS.png
        """
        save_dir = os.path.join(os.getcwd(), AppConfig.SCREENSHOT_DIR)
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)

        screenshot = ImageGrab.grab()
        screenshot.save(filepath)
        print(f"截图已保存: {filepath}")

    def _open_calculator(self):
        """
        打开系统计算器应用
        
        支持多平台：
        - Windows: calc.exe
        - macOS: Calculator.app
        - Linux: gnome-calculator
        """
        system = sys.platform
        try:
            if system == "win32":
                subprocess.Popen("calc.exe")
            elif system == "darwin":
                subprocess.Popen(["open", "-a", "Calculator"])
            else:
                subprocess.Popen(["gnome-calculator"])
        except Exception as e:
            print(f"打开计算器失败: {e}")
