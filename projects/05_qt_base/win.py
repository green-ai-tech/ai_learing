import time
from PySide6.QtWidgets import QDialog, QFileDialog
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QTimer

from ui_ai import Ui_AIModel
from dev import AIDevice

MODEL_NAMES = ["深度检测", "目标检测", "目标分割", "姿势检测", "背景替换"]


class AIWindow(QDialog):
    """
    类功能简述：AI视觉测试主窗口类
    类功能详述：继承自 QDialog，是整个机器视觉测试平台的主界面控制器。
               负责界面信号绑定、视频源切换、模型切换、启停控制、帧显示及统计信息更新。
               依赖：Ui_AIModel（界面布局）、AIDevice（推理设备线程）、PySide6
    @author Logic Ye
    @date 2026-03-28
    @version 1.0
    """

    def __init__(self):
        """初始化主窗口：构建 UI、创建计时器、绑定信号，并显示窗口。"""
        super().__init__()
        self.ui = Ui_AIModel()
        self.ui.setupUi(self)

        self._t_start = time.time()
        self._running = False

        # 计时器（每秒刷新已运行时长）
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._update_elapsed)

        self._bind_signals()
        self.show()

    # ──────────────────────────────────────────────
    #  信号绑定
    # ──────────────────────────────────────────────
    def _bind_signals(self):
        """将界面中各控件的信号连接到对应的槽函数，完成事件驱动的逻辑关联。"""
        ui = self.ui

        # 模型选择
        ui.rad_depth.toggled.connect(lambda c: c and self._switch_model(0))
        ui.rad_detect.toggled.connect(lambda c: c and self._switch_model(1))
        ui.rad_seg.toggled.connect(lambda c: c and self._switch_model(2))
        ui.rad_pose.toggled.connect(lambda c: c and self._switch_model(3))
        ui.rad_bg.toggled.connect(lambda c: c and self._switch_model(4))

        # 视频源
        ui.rad_cam.toggled.connect(self._on_source_changed)
        ui.rad_video.toggled.connect(self._on_source_changed)

        # 文件选取
        ui.btn_select.clicked.connect(self._select_video_file)

        # 开始 / 停止
        ui.btn_start.clicked.connect(self._start)
        ui.btn_stop.clicked.connect(self._stop)

    # ──────────────────────────────────────────────
    #  开始 / 停止
    # ──────────────────────────────────────────────
    def _start(self):
        """
        启动推理线程。
        创建 AIDevice 线程实例，连接视频帧和统计信号，根据当前界面配置
        设置视频源和模型类别，启动线程并更新按钮状态与日志。
        """
        if self._running:
            return
        self.dev = AIDevice()
        self.dev.signal_video.connect(self._show_video)
        self.dev.signal_stats.connect(self._update_stats)

        # 应用当前视频源设置
        if self.ui.rad_video.isChecked():
            path = self.ui.video_input.text().strip()
            if path:
                self.dev.switch_to_video(path)
        # 同步模型
        self.dev.model_class = self._current_model_id()

        self.dev.start()
        self._running  = True
        self._t_start  = time.time()
        self._timer.start()

        self.ui.btn_start.setEnabled(False)
        self.ui.btn_stop.setEnabled(True)
        self._log(f"启动 [{MODEL_NAMES[self.dev.model_class]}]")

    def _stop(self):
        """停止推理线程，释放设备资源，重置按钮状态并清空视频显示区。"""
        if not self._running:
            return
        self.dev.close_device()
        self._running = False
        self._timer.stop()
        self.ui.btn_start.setEnabled(True)
        self.ui.btn_stop.setEnabled(False)
        self.ui.lbl_video.setText("视频 / 图像显示区域")
        self._log("已停止")

    # ──────────────────────────────────────────────
    #  视频源切换
    # ──────────────────────────────────────────────
    def _on_source_changed(self):
        """
        响应视频源单选按钮切换事件。
        根据当前选中状态显示/隐藏文件路径行；若推理正在运行则同步切换设备视频源。
        """
        is_video = self.ui.rad_video.isChecked()
        self.ui.file_row.setVisible(is_video)

        if not self._running:
            return
        if is_video:
            path = self.ui.video_input.text().strip()
            if path:
                self.dev.switch_to_video(path)
        else:
            self.dev.switch_to_camera()

    def _select_video_file(self):
        """弹出文件选择对话框，将所选视频路径写入输入框，并在运行时即时切换视频源。"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选取视频文件", "",
            "视频文件 (*.mp4 *.avi *.mkv *.mov *.wmv);;所有文件 (*)"
        )
        if not path:
            return
        self.ui.video_input.setText(path)
        if self._running:
            self.dev.switch_to_video(path)

    # ──────────────────────────────────────────────
    #  模型切换
    # ──────────────────────────────────────────────
    def _current_model_id(self):
        """遍历模型单选按钮，返回当前选中模型的索引（0~4），默认返回 0（深度检测）。"""
        for i, rb in enumerate([
            self.ui.rad_depth, self.ui.rad_detect,
            self.ui.rad_seg,   self.ui.rad_pose, self.ui.rad_bg
        ]):
            if rb.isChecked():
                return i
        return 0

    def _switch_model(self, idx: int):
        """
        切换推理模型。
        若推理线程正在运行，同步更新 AIDevice 的模型类别；
        更新界面模型标签并加载对应结构图，并写入操作日志。
        """
        if self._running:
            self.dev.model_class = idx
        self.ui.lbl_model.setText(f"当前模型：{MODEL_NAMES[idx]}")
        self._load_model_image(idx)
        self._log(f"切换模型 → {MODEL_NAMES[idx]}")

    def _load_model_image(self, idx: int):
        """
        加载并显示指定模型的结构图。
        从 Ui_AIModel.MODEL_IMAGES 中取对应路径，按比例缩放填充预览标签；
        若图片不存在则显示占位文字提示。
        """
        path = Ui_AIModel.MODEL_IMAGES.get(idx, "")
        px = QPixmap(path)
        lbl = self.ui.model_preview
        if px.isNull():
            lbl.setText(f"[{MODEL_NAMES[idx]}] 结构图\n(请在 assets/images/ 放置对应图片)")
            lbl.setPixmap(QPixmap())
        else:
            lbl.setPixmap(
                px.scaled(lbl.width(), lbl.height(),
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
            )

    # ──────────────────────────────────────────────
    #  槽：显示视频帧
    # ──────────────────────────────────────────────
    def _show_video(self, h, w, c, data):
        """
        接收推理线程发出的视频帧信号，将 BGR 字节数据转为 QPixmap 并按比例缩放显示。
        @param h    帧高度（像素）
        @param w    帧宽度（像素）
        @param c    通道数（通常为 3）
        @param data BGR 字节数据
        """
        qimage = QImage(data, w, h, w * c, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(qimage)
        lbl = self.ui.lbl_video
        lbl.setPixmap(
            pixmap.scaled(lbl.size(),
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.FastTransformation)
        )

    # ──────────────────────────────────────────────
    #  槽：更新统计信息
    # ──────────────────────────────────────────────
    def _update_stats(self, n_obj: int, fps: float, avg_conf: float):
        """
        接收推理统计信号，刷新界面上的目标数、帧率和平均置信度标签。
        @param n_obj     检测到的目标数量
        @param fps       当前推理帧率
        @param avg_conf  所有目标的平均置信度（无目标时为 0）
        """
        self.ui.lbl_objects.setText(f"检测目标数：{n_obj}")
        self.ui.lbl_fps.setText(f"推理帧率：{fps:.1f} FPS")
        if avg_conf > 0:
            self.ui.lbl_conf.setText(f"平均置信度：{avg_conf:.2f}")
        else:
            self.ui.lbl_conf.setText("平均置信度：—")

    def _update_elapsed(self):
        """由 QTimer 每秒触发，计算并刷新界面上的已运行时长标签（格式 MM:SS）。"""
        elapsed = int(time.time() - self._t_start)
        m, s = divmod(elapsed, 60)
        self.ui.lbl_time.setText(f"已运行时长：{m:02d}:{s:02d}")

    # ──────────────────────────────────────────────
    #  推理日志
    # ──────────────────────────────────────────────
    def _log(self, msg: str):
        """向推理日志文本框追加带时间戳的消息。"""
        ts = time.strftime("%H:%M:%S")
        self.ui.edt_info.append(f"[{ts}] {msg}")

    # ──────────────────────────────────────────────
    #  关闭窗口
    # ──────────────────────────────────────────────
    def closeEvent(self, event):
        """窗口关闭事件处理：若推理线程仍在运行，先优雅地停止设备线程再接受关闭。"""
        if self._running:
            self.dev.close_device()
        event.accept()
