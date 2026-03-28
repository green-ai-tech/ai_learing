from PySide6.QtWidgets import QApplication
import sys
# Qt三大组件：QWidgets（GUI组件）， QGui（底层图形图像），QCore（线程，信号）


class AIApplication(QApplication):
    """
    类功能简述：AI应用程序管理类
    类功能详述：继承自 QApplication，作为整个机器视觉测试平台的应用程序入口。
               负责初始化 Qt 事件循环、管理全局资源（如模型预加载、用户配置等）。
               依赖：PySide6.QtWidgets.QApplication
    @author Logic Ye
    @date 2026-03-28
    @version 1.0
    """

    def __init__(self):
        """初始化应用程序，传入命令行参数并执行全局预处理（模型加载、用户配置等）。"""
        super(AIApplication, self).__init__(sys.argv)
        # 其他用户的处理
        # 模型加载