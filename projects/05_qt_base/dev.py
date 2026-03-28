from PySide6.QtCore import QThread, Signal
import os, sys, cv2
os.environ["TRANSFORMERS_OFFLINE"] = "1"   # 禁止 transformers 联网
os.environ["HF_DATASETS_OFFLINE"]  = "1"
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import YolosForObjectDetection, YolosImageProcessor
from ultralytics import YOLO
from ultralytics import settings as ul_settings
import PIL.Image as Image
import torch
import torch.nn as nn
import numpy as np
import time

# 禁止 ultralytics 联网检查更新
ul_settings.update({"sync": False})


MIDAS_DIR = "E:/models/midas-small"
if MIDAS_DIR not in sys.path:
    sys.path.insert(0, MIDAS_DIR)

# ──────────────────────────────────────────────
#  model_class 编号
#   0 = 深度检测  (MiDaS small, E:/models/midas-small)
#   1 = 目标检测  (yolos-tiny,   E:/models/yolos-tiny)
#   2 = 目标分割  (face-parsing, E:/models/face-parsing)
#   3 = 姿势检测  (yolo11n-pose, E:/models/yolo11n-pose.pt)
#   4 = 背景替换  (yolo11n-seg,  E:/models/yolo11n-seg.pt)
# ──────────────────────────────────────────────

YOLOS_TINY_PATH = "E:/models/yolos-tiny"
SEG_PATH        = "E:/models/face-parsing"
POSE_PATH       = "E:/models/yolo11n-pose.pt"
SEG_UL_PATH     = "E:/models/yolo11n-seg.pt"
BG_IMAGE_PATH   = "E:/ai_learning/assets/images/06_replace.png"

# COCO 17 关键点骨架连接
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class AIDevice(QThread):
    """
    类功能简述：AI推理设备线程类
    类功能详述：继承自 QThread，作为后台推理工作线程运行。
               支持五种推理模式（深度估计、目标检测、语义分割、姿势检测、背景替换），
               通过信号将每帧结果和统计数据传递给主窗口。
               支持运行时热切换视频源（摄像头/视频文件）和推理模型（懒加载策略）。
               依赖：PySide6.QtCore.QThread、ultralytics、transformers、MiDaS、OpenCV、PyTorch
    @author Logic Ye
    @date 2026-03-28
    @version 1.0
    """
    signal_video   = Signal(int, int, int, bytes)
    # 推理统计信号：目标数, FPS, 置信度均值
    signal_stats   = Signal(int, float, float)

    def __init__(self):
        """
        初始化推理设备线程。
        打开默认摄像头（索引 0），初始化各模型句柄为 None，读取背景替换图片，
        并设置运行状态标志与统计计时器。
        """
        super().__init__()
        self.isStop = False

        # 默认摄像头
        self._source_type = 0          # 0=摄像头, 1=视频文件
        self._video_path   = ""
        self.camera = cv2.VideoCapture(0)

        self._pending_class = 0
        self._loaded_class  = -1

        # 模型句柄
        self.model_depth    = None
        self.transform_depth = None
        self.model_seg      = None
        self.precosse_seg   = None
        self.model_detect   = None
        self.processor_detect = None
        self.model_pose     = None
        self.model_seg_ul   = None

        self.bg_image = cv2.imread(BG_IMAGE_PATH)

        # 统计
        self._t_start  = time.time()
        self._frame_t  = time.time()

    # ──────────────────────────────────────────────
    #  属性
    # ──────────────────────────────────────────────
    @property
    def model_class(self):
        """返回当前待切换的模型类别索引（0~4），写操作会触发下一帧时的懒加载切换。"""
        return self._pending_class

    @model_class.setter
    def model_class(self, value):
        self._pending_class = value

    # ──────────────────────────────────────────────
    #  视频源切换
    # ──────────────────────────────────────────────
    def switch_to_camera(self):
        """切换视频源为系统默认摄像头（索引 0），释放旧的 VideoCapture 资源。"""
        old = self.camera
        self._source_type = 0
        self.camera = cv2.VideoCapture(0)
        old.release()

    def switch_to_video(self, path: str):
        """切换视频源为指定路径的视频文件，释放旧的 VideoCapture 资源。"""
        old = self.camera
        self._source_type = 1
        self._video_path  = path
        self.camera = cv2.VideoCapture(path)
        old.release()

    # ──────────────────────────────────────────────
    #  模型管理
    # ──────────────────────────────────────────────
    def _unload_models(self):
        """卸载所有已加载的模型，释放 GPU 显存（调用 torch.cuda.empty_cache()）。"""
        for attr in ("model_depth", "model_seg", "model_detect",
                     "model_pose", "model_seg_ul"):
            m = getattr(self, attr, None)
            if m is not None:
                del m
                setattr(self, attr, None)
        self.transform_depth  = None
        self.precosse_seg     = None
        self.processor_detect = None
        torch.cuda.empty_cache()

    def _apply_pending(self):
        """
        检查是否有待切换的模型，若与当前已加载模型不同则先卸载旧模型再加载新模型。
        加载失败时将 pending 重置为已加载状态，避免每帧反复重试。
        """
        target = self._pending_class
        if self._loaded_class == target:
            return
        self._unload_models()
        try:
            self._load_model(target)
        except Exception as e:
            print(f"[AIDevice] 模型加载失败 (class={target}): {e}")
            # 失败时把 pending 也重置，避免每帧无限重试
            self._pending_class = self._loaded_class if self._loaded_class >= 0 else -1
            self._loaded_class = self._pending_class
            return
        self._loaded_class = target

    def _load_model(self, target):
        """
        根据目标模型编号加载对应模型到内存/显存。
        0=MiDaS深度估计（本地加载，CPU/GPU自动选择）
        1=YOLOS目标检测（HuggingFace transformers，GPU）
        2=Segformer语义分割（HuggingFace transformers，GPU）
        3=YOLO姿势检测（ultralytics，GPU）
        4=YOLO实例分割（ultralytics，GPU，用于背景替换）
        """
        if target == 0:
            # MiDaS small — 完全本地加载，不联网
            from midas.midas_net_custom import MidasNet_small
            from midas.transforms import Resize, NormalizeImage, PrepareForNet
            from torchvision.transforms import Compose

            weights_path = f"{MIDAS_DIR}/midas_v21_small_256.pt"
            device = "cuda" if torch.cuda.is_available() else "cpu"

            model = MidasNet_small(
                None, features=64, backbone="efficientnet_lite3",
                exportable=True, non_negative=True, blocks={"expand": True}
            )
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            self.model_depth = model

            self.transform_depth = Compose([
                lambda img: {"image": img / 255.0},
                Resize(256, 256,
                       resize_target=None, keep_aspect_ratio=True,
                       ensure_multiple_of=32, resize_method="upper_bound",
                       image_interpolation_method=cv2.INTER_CUBIC),
                NormalizeImage(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
                lambda sample: torch.from_numpy(sample["image"]).unsqueeze(0),
            ])

        elif target == 1:
            self.model_detect     = YolosForObjectDetection.from_pretrained(
                YOLOS_TINY_PATH, local_files_only=True)
            self.processor_detect = YolosImageProcessor.from_pretrained(
                YOLOS_TINY_PATH, local_files_only=True)
            self.model_detect.to("cuda")

        elif target == 2:
            self.model_seg    = SegformerForSemanticSegmentation.from_pretrained(
                SEG_PATH, local_files_only=True)
            self.precosse_seg = SegformerImageProcessor.from_pretrained(
                SEG_PATH, local_files_only=True)
            self.model_seg.to("cuda")

        elif target == 3:
            self.model_pose = YOLO(POSE_PATH)

        elif target == 4:
            self.model_seg_ul = YOLO(SEG_UL_PATH)

    # ──────────────────────────────────────────────
    #  推理方法
    # ──────────────────────────────────────────────
    def infer_depth(self, image):
        """
        MiDaS small 单目深度估计。
        将 BGR 帧转为 RGB → 经预处理管线输入模型 → 双三次插值还原到原图尺寸 →
        归一化后应用 PLASMA 伪彩色 → 与原图 35%/65% 融合叠加。
        返回：BGR 格式的深度可视化图像。
        """
        # 根据是否有 GPU 选择推理设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # BGR → RGB，因为 MiDaS 训练时使用 RGB 通道顺序
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 经预处理管线（缩放/归一化/转 Tensor）后送入设备
        input_batch = self.transform_depth(rgb).to(device)
        with torch.no_grad():                         # 推理阶段不需要梯度
            prediction = self.model_depth(input_batch)  # 前向传播，输出深度图（低分辨率）
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),              # 添加 channel 维，形状变为 (1,1,H,W)
                size=image.shape[:2],                 # 目标尺寸 = 原图 (H, W)
                mode="bicubic",                       # 双三次插值，深度图放大效果更平滑
                align_corners=False,
            ).squeeze()                               # 去掉多余维度，恢复为 (H, W)
        depth = prediction.cpu().numpy()              # 转为 numpy 数组，方便后续 OpenCV 处理
        # 将深度值线性映射到 [0, 255]，以便可视化
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:                             # 正常情况：存在深度差异
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:                                         # 极端情况：画面全黑/全白，避免除零
            depth_norm = np.zeros_like(depth, dtype=np.uint8)
        # 将灰度深度图映射为 PLASMA 伪彩色（近=黄/白，远=紫/蓝）
        colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)
        # 原图占 35%、伪彩色占 65% 加权融合，保留纹理细节同时突出深度信息
        result = cv2.addWeighted(image, 0.35, colored, 0.65, 0)
        return result

    def infer_detect(self, image, threshold=0.7):
        """
        YOLOS 目标检测推理。
        将 BGR 帧送入 YOLOS-tiny 模型，后处理过滤置信度低于 threshold 的结果，
        在原图上绘制边界框和标签文字。
        返回：(标注后的 BGR 图像, 目标数量, 平均置信度)
        """
        # OpenCV BGR → PIL RGB，YOLOS processor 期望 PIL Image 输入
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # processor 将图像编码为模型所需的 tensor，并送入 GPU
        inputs = self.processor_detect(images=img, return_tensors="pt").to("cuda")
        # 前向推理，outputs 包含 logits 和预测框（归一化坐标）
        outputs = self.model_detect(**inputs)
        # 告知后处理将框坐标还原到原图像素坐标 (H, W)
        target_sizes = [(image.shape[0], image.shape[1])]
        # 过滤低置信度预测，返回 boxes/labels/scores
        results = self.processor_detect.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes)[0]
        result_img = image.copy()             # 在副本上绘制，不破坏原帧
        scores = results["scores"]
        for score, label, box in zip(scores, results["labels"], results["boxes"]):
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]   # 框坐标取整
            # 通过模型配置的 id2label 字典将类别 id 转为可读名称
            label_name = self.model_detect.config.id2label.get(
                label.item(), str(label.item()))
            # 绘制绿色矩形框
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 220, 100), 2)
            # 在框上方显示类别名称和置信度分数
            cv2.putText(result_img, f"{label_name} {score:.2f}",
                        (x1, max(y1 - 5, 0)),        # y1-5 防止文字超出图像顶部
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 100), 1)
        n_obj = len(scores)                           # 检出目标总数
        avg_conf = float(scores.mean()) if n_obj > 0 else 0.0  # 平均置信度，无目标时为 0
        return result_img, n_obj, avg_conf

    def infer_seg(self, image):
        """
        Segformer 语义分割推理（face-parsing 模型）。
        将 BGR 帧送入模型，将 logits 双线性插值还原到原始分辨率，
        返回每像素的类别索引 mask（numpy 二维数组）。
        """
        # image 已是 RGB 顺序（调用方传入的 BGR 帧需注意），直接包装为 PIL Image
        img = Image.fromarray(image)
        # processor 对图像做 resize / normalize，并转为 GPU tensor
        inputs = self.precosse_seg(img, return_tensors="pt").to("cuda")
        # 前向推理，outputs.logits 形状为 (1, num_classes, H', W')，H'/W' 为下采样后的尺寸
        outputs = self.model_seg(**inputs)
        results = outputs.logits
        # 将低分辨率 logits 双线性插值回原图尺寸
        # img.size 返回 (W, H)，[::-1] 转为 (H, W) 以匹配 interpolate 的 size 参数
        results = nn.functional.interpolate(
            results, size=img.size[::-1], mode='bilinear', align_corners=False)
        # 沿类别维度取 argmax，得到每像素预测类别；[0] 去掉 batch 维，转 numpy
        masks = results.argmax(dim=1)[0].cpu().numpy()
        return masks

    def infer_pose(self, image):
        """
        YOLO 姿势检测推理（COCO 17 关键点）。
        检测图像中所有人体关键点，对置信度 > 0.5 的点绘制圆圈，
        并按 SKELETON 连接关系绘制骨骼线段（橙色）。
        返回：标注骨骼的 BGR 图像。
        """
        # verbose=False 关闭 ultralytics 的逐帧控制台输出
        results = self.model_pose(image, verbose=False)
        result_img = image.copy()           # 在副本上绘制，不破坏原帧
        for r in results:
            if r.keypoints is None:         # 当前帧未检测到人体，跳过
                continue
            # kps 形状 (N, 17, 2)：N 个人，每人 17 个关键点的 (x, y) 像素坐标
            kps   = r.keypoints.xy.cpu().numpy()
            confs = r.keypoints.conf        # 形状 (N, 17)：每个关键点的置信度
            if confs is not None:
                confs = confs.cpu().numpy()
            for i in range(len(kps)):       # 遍历每个检测到的人
                pts  = kps[i]               # 当前人的 17 个关键点坐标
                # 若模型未输出置信度，默认所有关键点置信度为 1（全部可见）
                conf = confs[i] if confs is not None else np.ones(17)
                # 绘制关键点圆圈：置信度 > 0.5 且坐标有效（非零）才绘制
                for j, (x, y) in enumerate(pts):
                    if conf[j] > 0.5 and x > 0 and y > 0:
                        cv2.circle(result_img, (int(x), int(y)), 4, (0, 255, 255), -1)
                # 按骨架定义连接相邻关键点：两端点均置信 > 0.5 且坐标有效才画线
                for a, b in SKELETON:
                    xa, ya = pts[a]
                    xb, yb = pts[b]
                    if (conf[a] > 0.5 and conf[b] > 0.5
                            and xa > 0 and ya > 0 and xb > 0 and yb > 0):
                        cv2.line(result_img,
                                 (int(xa), int(ya)), (int(xb), int(yb)),
                                 (0, 165, 255), 2)   # 橙色骨骼线，线宽 2
        return result_img

    def infer_bg_replace(self, image):
        """
        YOLO 实例分割 + 背景替换。
        仅对 class=0（人）进行分割，将所有人体 mask 合并为前景掩码，
        前景区域保留原图，背景区域替换为预加载的背景图片。
        返回：背景替换后的 BGR 图像（uint8）。
        """
        h, w = image.shape[:2]
        # 将背景图片缩放到与当前帧相同的分辨率，确保像素对齐
        bg = cv2.resize(self.bg_image, (w, h))
        # classes=[0] 只检测"人"，减少无关目标的干扰；verbose=False 关闭控制台输出
        results = self.model_seg_ul(image, classes=[0], verbose=False)
        # 初始化全零掩码，像素值 1=前景（人体），0=背景
        person_mask = np.zeros((h, w), dtype=np.uint8)
        for r in results:
            if r.masks is None:             # 当前帧未检测到人，跳过
                continue
            for mask in r.masks.data:       # 遍历每个检测到的人的实例掩码
                m = mask.cpu().numpy()      # 掩码值在 [0,1] 之间（sigmoid 输出）
                # 掩码分辨率可能与原图不同，需 resize 回原图尺寸
                m = cv2.resize(m, (w, h))
                # 阈值 0.5 二值化后与已有掩码取最大值（逻辑 OR），合并多人掩码
                person_mask = np.maximum(person_mask, (m > 0.5).astype(np.uint8))
        # 将单通道掩码扩展为 3 通道，以便与 BGR 图像做像素级选择
        mask3 = np.stack([person_mask] * 3, axis=2)
        # 前景（mask3==1）取原图像素，背景（mask3==0）取替换背景像素
        return np.where(mask3 == 1, image, bg).astype(np.uint8)

    # ──────────────────────────────────────────────
    #  视频文件循环播放辅助
    # ──────────────────────────────────────────────
    def _read_frame(self):
        """
        读取一帧画面。
        视频文件播放到末尾时自动回绕到第 0 帧实现循环播放；摄像头模式直接读取。
        返回：(status, image) 与 cv2.VideoCapture.read() 保持一致。
        """
        status, image = self.camera.read()
        if not status and self._source_type == 1:
            # 视频结束，回头播
            self.camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            status, image = self.camera.read()
        return status, image

    # ──────────────────────────────────────────────
    #  主循环
    # ──────────────────────────────────────────────
    def run(self):
        """
        线程主循环：每次迭代先检查并切换模型，读取一帧，按当前模型执行推理，
        将结果帧通过 signal_video 信号发送，并通过 signal_stats 发送统计数据。
        循环间隔 10ms，避免 CPU 空转。
        """
        self._t_start = time.time()
        while not self.isStop:
            self._apply_pending()
            status, image = self._read_frame()

            if not status:
                print("设备故障，采集终止")
                break

            n_obj, avg_conf = 0, 0.0
            cur = self._loaded_class

            # cur == -1 表示模型加载失败，直接透传原帧；各分支额外检查句柄非 None，防止切换瞬间的空窗期崩溃
            if cur == 0 and self.model_depth is not None:
                image = self.infer_depth(image)
            elif cur == 1 and self.processor_detect is not None:
                image, n_obj, avg_conf = self.infer_detect(image)
            elif cur == 2 and self.model_seg is not None:
                masks = self.infer_seg(image)
                masks = np.where(masks == 3, 255, 0)
                masks = np.stack([masks, masks, masks], axis=2)
                image = masks
            elif cur == 3 and self.model_pose is not None:
                image = self.infer_pose(image)
            elif cur == 4 and self.model_seg_ul is not None:
                image = self.infer_bg_replace(image)

            image = image.astype(np.uint8)
            h, w, c = image.shape
            self.signal_video.emit(h, w, c, image.tobytes())

            # FPS 计算
            now = time.time()
            fps = 1.0 / max(now - self._frame_t, 1e-6)
            self._frame_t = now
            self.signal_stats.emit(n_obj, fps, avg_conf)

            QThread.usleep(10000)

    def close_device(self):
        """






                优雅停止线程：设置停止标志后自旋等待线程退出，再释放 VideoCapture 资源。
        调用方（主线程）应在停止按钮或窗口关闭事件中调用此方法。
        """
        self.isStop = True
        while self.isRunning():
            pass
        self.camera.release()
        print("线程优雅地结束")