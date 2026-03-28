"""
机器视觉测试平台 — 程序入口
创建 AIApplication 和 AIWindow 实例，启动 Qt 事件循环并在退出时返回状态码。
@author Logic Ye
@date 2026-03-28
@version 1.0
"""
from app import AIApplication
from win import AIWindow

app = AIApplication()
win = AIWindow()
status  = app.exec()
import sys
sys.exit(status)
