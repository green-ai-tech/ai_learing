# 01_机器视觉 (Machine Vision)

## 📚 学习主题
本模块主要学习**机器视觉 (Machine Vision/Computer Vision)** 的基础知识与深度学习应用，涵盖从传统图像处理到现代深度学习模型（如 CNN, LeNet5, Transformers）的演进。

---

## 📁 文件结构

```
01_machine_vision/
├── 01_video_object_detection.ipynb          # 实时视频目标检测 (YOLO)
├── 02_video_background_replacement.ipynb    # 视频背景替换 (OpenCV 抠图)
├── 03_image_processing_and_features.ipynb   # 图像处理与特征提取 (边缘检测、滤波等)
├── 04_lenet5_handwritten_digit_recognition.ipynb # 经典 LeNet-5 手写数字识别
├── 05_pytorch_tensors_and_nn.ipynb          # PyTorch 核心：张量与神经网络
├── 06_model_training_finetuning_01.ipynb    # 模型训练、微调与迁移学习 (基础)
├── 07_model_architecture.ipynb              # 模型架构设计
├── 07_model_training_finetuning_02.ipynb    # 模型训练与微调 (Transformers 训练框架)
├── 08_model_postprocessing_and_architecture.ipynb # 模型后处理与架构优化
├── 08_transfer_learning_fine_tuning_dataset.ipynb.ipynb # 迁移学习数据集处理
├── 09_machine_vision_segmentation_model_testing.ipynb # 图像分割模型测试
├── 10_transformers_summary.ipynb            # Transformers 知识总结
├── 11_transformers_training_framework.ipynb # Transformers 训练框架实战
├── video_yolo.py                            # Python 脚本版 YOLO 推理
└── homework/                                # 课后作业与练习
```

---

## 🎯 核心学习路径

### 第一阶段：传统机器视觉 (OpenCV)
- **内容**: 图像的基本操作、特征提取、视频处理。
- **重点**:
    - **目标检测**: 使用 YOLO 等预训练模型进行实时目标检测 (01_video)。
    - **背景替换**: 掌握 OpenCV 的图像分割与掩膜 (Mask) 技术 (02_video)。
    - **特征提取**: 边缘检测、滤波、关键点检测 (03_image_processing)。

### 第二阶段：深度学习基础 (PyTorch & CNN)
- **内容**: 从全连接网络向卷积神经网络 (CNN) 的过渡。
- **重点**:
    - **LeNet-5**: 深度学习最经典的入门模型，理解卷积层、池化层、全连接层的作用 (04_lenet5)。
    - **张量操作**: 理解 PyTorch 的 Tensor 计算和 `nn.Module` 的构建方式 (05_pytorch)。

### 第三阶段：模型训练与迁移学习
- **内容**: 如何在自己的数据集上训练和微调模型。
- **重点**:
    - **预训练模型**: 理解 Pre-training 和 Transfer Learning 的概念 (06, 08)。
    - **训练框架**: 学习如何编写一个完整的模型训练循环（数据加载、损失计算、优化器）(07, 11)。
    - **图像分割**: 从分类任务拓展到像素级的分割任务 (09_segmentation)。

---

## 🚀 快速开始

### 1. 环境准备
确保已安装 `opencv-python`, `torch`, `torchvision`, `numpy`, `matplotlib` 等库。
```bash
pip install opencv-python torch torchvision numpy matplotlib
```

### 2. 运行建议
- **初学者**: 建议从 **`04_lenet5_handwritten_digit_recognition.ipynb`** 开始，因为代码最简洁，容易理解深度学习全流程。
- **进阶**: 完成基础后，尝试 **`01_video_object_detection.ipynb`** 体验最酷炫的实时检测效果。
- **工程实战**: 参考 **`07`** 和 **`11`** 中的训练框架代码，学习如何组织自己的训练脚本。

---

## 📊 关键概念

| 概念 | 解释 | 对应文件 |
|------|------|----------|
| **目标检测 (Object Detection)** | 在图像中找出“什么东西”以及“在哪” (Bounding Box) | `01_video_...`, `video_yolo.py` |
| **特征工程 (Feature Engineering)** | 从原始像素中提取有意义的模式（如边缘、纹理） | `03_image_...` |
| **卷积神经网络 (CNN)** | 通过卷积核提取局部特征，层层堆叠提取全局特征 | `04_lenet5...` |
| **迁移学习 (Transfer Learning)** | 利用别人训练好的模型（预训练权重），在自己的数据上进行微调 | `06_...`, `08_...` |
| **图像分割 (Segmentation)** | 将图像中的每个像素进行分类（不仅仅是整张图分类） | `09_...` |

---

## 💡 学习建议

1. **理解数据流向**: 无论是 CNN 还是 Transformer，核心都是 `输入 -> 模型提取特征 -> 输出预测 -> 计算损失 -> 更新权重`。
2. **对比传统与深度**: 对比 `03_image_processing` 和 `04_lenet5`，思考为什么深度学习能自动提取特征，而不需要人工设计滤波器。
3. **善用预训练模型**: 在实际工程中，很少从零训练 (Training from scratch)。重点掌握如何“加载预训练模型”并“替换分类头 (Head)”。

---

## 📅 更新记录

| 日期 | 更新内容 |
|------|----------|
| 2026-04-11 | 初始版本，梳理机器视觉学习路线与文件说明 |
