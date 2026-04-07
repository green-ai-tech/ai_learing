#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练脚本 - 语音指令识别系统
用于训练一个基于CNN的音频分类模型，支持Hugging Face Transformers格式

作者：Logic Ye
日期：2026-04-04
"""

import os
import sys
import json
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio.transforms as T
from transformers import PreTrainedModel, PretrainedConfig
from tqdm import tqdm
from typing import Optional, Tuple, Union

# ==================== 配置参数 ====================
# 数据集路径：存放按类别分类的音频文件（.wav格式）
# 目录结构示例：
#   系统控制/
#     ├── 动鼠标/
#     │   ├── sample1.wav
#     │   └── sample2.wav
#     ├── 截屏/
#     │   └── ...
#     └── 打开计算器/
#         └── ...
DATASET_PATH = "/Volumes/AI/my_Datasets/系统控制"

# 模型保存路径：训练完成后模型及相关文件保存到此目录
# 包含：config.json（配置）、pytorch_model.bin（权重）、label_mapping.json（标签映射）
MODEL_SAVE_PATH = "/Volumes/AI/models/my_trained/model_speech_control_transformer"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)  # 确保目录存在

# ==================== 双日志输出系统 ====================
# 创建一个Tee类，同时将输出打印到控制台和日志文件
# 这样既能实时查看训练进度，又能保存完整的训练记录
log_file = open(os.path.join(MODEL_SAVE_PATH, "training.log"), "w", encoding="utf-8")

class Tee:
    """
    自定义输出流，同时将内容写入多个文件对象
    用于实现控制台和日志文件同步输出
    """
    def __init__(self, *files):
        self.files = files  # 存储所有需要写入的文件对象
    
    def write(self, obj):
        """写入内容到所有文件对象"""
        for f in self.files:
            f.write(obj)
            f.flush()  # 立即刷新，确保内容立即写入磁盘
    
    def flush(self):
        """刷新所有文件对象的缓冲区"""
        for f in self.files:
            f.flush()

# 保存原始标准输出，并将Tee设置为新的标准输出
original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, log_file)

# ==================== 1. 准备标签 ====================
# 自动扫描数据集目录，识别所有有效的标签目录
# 标签目录定义：不以"_"开头的子目录（以下划线开头的目录会被忽略）
labels = [d for d in os.listdir(DATASET_PATH)
          if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith("_")]
if not labels:
    print(f"错误：未在 {DATASET_PATH} 中找到标签目录")
    sys.exit(1)

# 创建标签与数字ID的映射关系
# label2id: 将标签文本转换为数字ID（用于训练时的目标值）
# id2label: 将数字ID转换回标签文本（用于推理时的结果展示）
label2id = {lbl: i for i, lbl in enumerate(labels)}
id2label = {i: lbl for i, lbl in enumerate(labels)}
print(f"识别到的指令标签: {labels}")
print(f"标签映射: {label2id}")

# ==================== 2. 定义配置类 ====================
class SimpleCNNConfig(PretrainedConfig):
    """
    自定义 CNN 模型的配置类，继承自 PretrainedConfig
    
    作用：
    - 定义模型的所有超参数
    - 支持 Hugging Face 的 save_pretrained() 和 from_pretrained() 方法
    - 使模型配置可以序列化和复用
    """
    model_type = "simple_cnn"  # 模型类型标识，用于 AutoClass 支持

    def __init__(
        self,
        num_labels: int = 2,           # 分类类别数量
        hidden_size: int = 768,        # 隐藏层维度（兼容 transformers 标准）
        num_conv_layers: int = 3,      # 卷积层数量
        conv_channels: list = [16, 32, 64],  # 每层卷积输出通道数
        kernel_size: int = 3,          # 卷积核大小
        dropout_rate: float = 0.1,     # dropout 比率（防止过拟合）
        **kwargs,
    ):
        """
        初始化配置参数
        
        参数:
            num_labels: 分类数量，即语音指令的种类数
            hidden_size: 隐藏层维度，兼容 transformers 库的标准参数
            num_conv_layers: 卷积层数量
            conv_channels: 每层卷积通道数列表，决定特征提取能力
            kernel_size: 卷积核大小，影响感受野
            dropout_rate: dropout 比率，防止模型过拟合
        """
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_conv_layers = num_conv_layers
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

# ==================== 3. 定义模型类（继承 PreTrainedModel）====================
class SimpleCNNForAudioClassification(PreTrainedModel):
    """
    用于音频分类的 CNN 模型，完全兼容 Hugging Face Transformers 格式
    
    模型结构：
    1. 多层 2D 卷积层 + BatchNorm + ReLU激活 + MaxPooling
    2. 自适应平均池化层（将特征图统一为 4x4 大小）
    3. 全连接分类层
    
    支持 save_pretrained() 和 from_pretrained() 方法，
    可以方便地保存和加载模型
    """
    config_class = SimpleCNNConfig  # 关联配置类

    def __init__(self, config: SimpleCNNConfig):
        """
        初始化模型结构
        
        参数:
            config: SimpleCNNConfig 对象，包含模型的所有超参数
        """
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        # 构建卷积层序列
        # 每个卷积块包含：Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
        conv_layers = []
        in_channels = 1  # 输入通道数（Mel频谱图为单通道）
        for out_channels in config.conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=config.kernel_size, padding=1),  # 2D卷积层
                nn.BatchNorm2d(out_channels),   # 批归一化，加速训练并防止梯度消失
                nn.ReLU(),                      # ReLU激活函数，引入非线性
                nn.MaxPool2d(2)                 # 最大池化，降低特征图尺寸
            ])
            in_channels = out_channels  # 更新输入通道数，用于下一层

        # 自适应池化层：无论输入尺寸如何，都输出固定的 4x4 特征图
        # 这样分类层的输入维度始终是固定的
        conv_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.conv = nn.Sequential(*conv_layers)  # 将所有层组合成一个序列

        # 分类头：将提取的特征映射到各个类别
        # 输入维度 = 最后一个卷积层的通道数 * 4 * 4（池化后的尺寸）
        self.classifier = nn.Linear(config.conv_channels[-1] * 4 * 4, config.num_labels)

        # 初始化模型权重（使用 Hugging Face 的默认初始化策略）
        self.post_init()

    def forward(
        self,
        input_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, dict]:
        """
        前向传播方法 - 模型的核心计算逻辑
        
        Args:
            input_values: 输入的音频特征，形状为 (batch_size, 1, n_mels, time_steps)
                         即批次大小 x 通道数 x Mel频段数 x 时间步数
            labels: 标签，形状为 (batch_size,)，训练时提供用于计算损失
            return_dict: 是否返回字典格式的输出（兼容 Hugging Face 习惯）
        
        Returns:
            如果 return_dict=True：返回包含 "loss"（如果有标签）和 "logits" 的字典
            如果 return_dict=False：返回元组 (loss, logits) 或 (logits,)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 特征提取：通过卷积层提取音频特征
        features = self.conv(input_values)
        features = features.view(features.size(0), -1)  # 展平为 (batch_size, 特征维度)
        logits = self.classifier(features)  # 分类层，得到每个类别的得分

        # 如果提供了标签，计算交叉熵损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数
            # 将 logits 和 labels 调整为合适的形状
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 根据 return_dict 决定返回格式
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,      # 损失值（仅在有标签时提供）
            "logits": logits,  # 每个类别的得分（未经过softmax）
        }

    # 可选：支持返回字典的便捷方法
    def forward_with_dict(self, *args, **kwargs):
        """便捷方法，强制使用字典格式返回"""
        return self.forward(*args, **kwargs, return_dict=True)

# ==================== 4. 数据集类 ====================
class MelDataset(Dataset):
    """
    自定义数据集类，用于加载音频文件并转换为 Mel 频谱图
    
    工作流程：
    1. 扫描所有音频文件（.wav格式）
    2. 加载音频并进行预处理（重采样、截断/填充到固定长度）
    3. 将音频转换为 Mel 频谱图（2D特征表示）
    4. 对特征进行标准化（均值为0，标准差为1）
    
    为什么使用 Mel 频谱图？
    - Mel 频谱图能够更好地表示人类听觉感知的频率特征
    - 2D结构非常适合 CNN 处理
    """
    def __init__(self, root_dir, label2id, duration=2, sr=16000):
        """
        初始化数据集
        
        参数:
            root_dir: 数据集根目录路径
            label2id: 标签到数字ID的映射字典
            duration: 每个音频片段的持续时间（秒）
            sr: 目标采样率（Hz）
        """
        self.samples = []  # 存储所有样本文件路径和对应标签
        # 遍历每个标签目录
        for label_name, label_id in label2id.items():
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.exists(label_dir):
                continue
            # 收集该标签目录下的所有 .wav 文件
            for f in os.listdir(label_dir):
                if f.endswith('.wav'):
                    self.samples.append((os.path.join(label_dir, f), label_id))
        
        self.duration = duration  # 音频持续时间（秒）
        self.sr = sr              # 采样率（Hz）
        self.n_samples = duration * sr  # 总采样点数
        # Mel 频谱图转换器：将原始音频转换为 64 维 Mel 频谱图
        self.mel_spec = T.MelSpectrogram(
            sample_rate=sr,        # 采样率
            n_mels=64,            # Mel滤波器数量（频率维度）
            n_fft=1024,           # FFT窗口大小
            hop_length=512        # 帧移（相邻帧之间的采样点数）
        )

    def __len__(self):
        """返回数据集的总样本数"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取指定索引的样本
        
        参数:
            idx: 样本索引
        
        返回:
            mel: Mel 频谱图张量，形状为 (1, n_mels, time_steps)
            label_id: 对应的标签ID
        """
        wav_path, label_id = self.samples[idx]
        
        # 加载音频文件
        waveform, sr = torchaudio.load(wav_path)
        
        # 如果是立体声（多声道），转换为单声道（取平均）
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 如果采样率不匹配，进行重采样
        if sr != self.sr:
            waveform = torchaudio.transforms.Resample(sr, self.sr)(waveform)
        
        # 如果音频太短，进行零填充
        if waveform.shape[1] < self.n_samples:
            pad = torch.zeros(1, self.n_samples - waveform.shape[1])
            waveform = torch.cat([waveform, pad], dim=1)
        else:
            # 如果音频太长，截断到目标长度
            waveform = waveform[:, :self.n_samples]
        
        # 转换为 Mel 频谱图
        mel = self.mel_spec(waveform)
        
        # 标准化：减去均值，除以标准差（使特征分布更稳定）
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        
        return mel, label_id

# ==================== 5. 数据加载 ====================
# 创建完整数据集
full_dataset = MelDataset(DATASET_PATH, label2id)

# 划分训练集和验证集（80% 训练，20% 验证）
# 训练集：用于模型学习参数
# 验证集：用于评估模型性能，检测过拟合
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print(f"\n数据集划分：")
print(f"  训练集样本数: {train_size}")
print(f"  验证集样本数: {val_size}")

# ==================== 训练超参数配置 ====================
BATCH_SIZE = 32           # 批次大小：每批处理32个样本
                          # 较大批次训练更稳定，但需要更多内存
EPOCHS = 50               # 训练轮数：遍历整个训练集50次
                          # 更多轮数可能带来更好的性能，但也可能过拟合
LEARNING_RATE = 0.001     # 学习率：控制参数更新的速度
                          # 太大可能导致不收敛，太小训练会很慢
WEIGHT_DECAY = 1e-4       # 权重衰减（L2正则化）：防止模型过拟合
                          # 通过惩罚大权重值来简化模型

# 创建数据加载器
# train_loader: 训练集加载器，shuffle=True 打乱数据顺序
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# val_loader: 验证集加载器，不需要打乱
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ==================== 6. 初始化模型 ====================
# 创建配置实例，定义模型的具体参数
config = SimpleCNNConfig(
    num_labels=len(labels),           # 分类数量 = 标签种类数
    hidden_size=768,                  # 隐藏层维度（参考 BERT 标准）
    conv_channels=[16, 32, 64],       # 三层卷积，通道数逐渐增加
    kernel_size=3,                    # 3x3 的卷积核
)

# 使用配置初始化模型
model = SimpleCNNForAudioClassification(config)
device = torch.device("cpu")  # 使用 CPU 训练（可改为 "cuda" 使用 GPU 加速）
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数（多分类任务标准选择）
optimizer = torch.optim.Adam(      # Adam 优化器
    model.parameters(),            # 需要优化的参数
    lr=LEARNING_RATE,              # 学习率
    weight_decay=WEIGHT_DECAY      # 权重衰减
)

# 打印模型信息
print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"运行设备: {device}")

# ==================== 7. 训练循环 ====================
# 训练过程：
# 1. 前向传播：模型预测
# 2. 计算损失：预测结果与真实标签的差距
# 3. 反向传播：计算梯度
# 4. 参数更新：优化器调整模型参数
# 5. 验证评估：每个 epoch 后在验证集上测试性能

best_acc = 0.0  # 记录最佳验证准确率，用于保存最优模型
print("\n开始训练...")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):
    # ========== 训练阶段 ==========
    model.train()  # 设置为训练模式（启用 dropout 等）
    total_loss = 0.0
    # 使用 tqdm 显示进度条
    progress = tqdm(train_loader, desc=f"第 {epoch}/{EPOCHS} 轮 (训练)", unit="batch")
    for mel, label in progress:
        mel, label = mel.to(device), label.to(device)  # 将数据移到指定设备
        optimizer.zero_grad()  # 清空上一步的梯度
        
        # 前向传播：获取预测结果和损失
        outputs = model(input_values=mel, labels=label, return_dict=True)
        loss = outputs["loss"]
        
        loss.backward()  # 反向传播：计算梯度
        optimizer.step()  # 更新参数
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())  # 更新进度条显示

    avg_loss = total_loss / len(train_loader)  # 计算平均训练损失

    # ========== 验证阶段 ==========
    model.eval()  # 设置为评估模式（关闭 dropout 等）
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算（节省内存和加速）
        for mel, label in val_loader:
            mel, label = mel.to(device), label.to(device)
            outputs = model(input_values=mel, return_dict=True)
            logits = outputs["logits"]
            pred = torch.argmax(logits, dim=1)  # 取概率最大的类别
            correct += (pred == label).sum().item()  # 统计正确预测数量
            total += label.size(0)  # 累计样本数
    val_acc = 100.0 * correct / total  # 计算验证准确率

    print(f"第 {epoch} 轮 结束: 训练损失 = {avg_loss:.4f}, 验证准确率 = {val_acc:.2f}%")

    # ========== 保存最佳模型 ==========
    if val_acc > best_acc:
        best_acc = val_acc
        # 使用 Hugging Face 标准方法保存模型和配置
        model.save_pretrained(MODEL_SAVE_PATH)
        # 保存标签映射（推理时需要用到）
        with open(os.path.join(MODEL_SAVE_PATH, "label_mapping.json"), "w", encoding="utf-8") as f:
            json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
        print(f"  -> 保存最佳模型 (准确率 {val_acc:.2f}%)")

    # 早停机制：如果准确率达到 98%，提前结束训练
    # 避免过度训练和节省时间
    if val_acc >= 98.0:
        print(f"验证准确率已达到 {val_acc:.2f}%，提前停止训练。")
        break

# ==================== 8. 训练完成 ====================
print("\n" + "=" * 60)
print(f"训练完成！最佳验证准确率: {best_acc:.2f}%")
print(f"模型已保存至: {MODEL_SAVE_PATH}")
print(f"保存的文件包括:")
print(f"  - config.json          # 模型配置文件（包含所有超参数）")
print(f"  - model.safetensors     # 模型权重文件（安全格式的模型参数）")
print(f"  - label_mapping.json   # 标签映射文件（自定义，用于推理）")
print(f"  - training.log         # 训练日志（完整训练过程记录）")

# 恢复标准输出（关闭日志文件）
sys.stdout = original_stdout
log_file.close()
print("训练日志已同时保存至文件。")

# ==================== 9. 推理示例 ====================
# 以下提供两个便捷函数，用于加载训练好的模型并进行预测
# 这些函数仅供测试使用，实际推理请使用 model_validation.py 或 voice_controller.py

def load_model_for_inference(model_path):
    """
    加载训练好的模型用于推理
    
    参数:
        model_path: 模型保存路径（包含 config.json 和 model.safetensors）
    
    返回:
        model: 加载好的模型（eval 模式）
        id2label: ID 到标签的映射字典
    """
    # 使用 from_pretrained 加载模型（Hugging Face 标准方法）
    model = SimpleCNNForAudioClassification.from_pretrained(model_path)
    model.eval()  # 设置为评估模式
    
    # 加载标签映射
    with open(os.path.join(model_path, "label_mapping.json"), "r", encoding="utf-8") as f:
        mapping = json.load(f)
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    return model, id2label

def predict_audio(wav_path, model, id2label, duration=2, sr=16000):
    """
    预测单个音频文件的类别
    
    参数:
        wav_path: 音频文件路径（.wav 格式）
        model: 加载好的模型
        id2label: ID 到标签的映射
        duration: 音频持续时间（秒）
        sr: 采样率（Hz）
    
    返回:
        predicted_label: 预测的标签名称
    """
    # 加载音频
    waveform, sr_orig = torchaudio.load(wav_path)
    
    # 转换为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 重采样（如果需要）
    if sr_orig != sr:
        waveform = torchaudio.transforms.Resample(sr_orig, sr)(waveform)

    # 截断或填充到目标长度
    target_len = duration * sr
    if waveform.shape[1] < target_len:
        pad = torch.zeros(1, target_len - waveform.shape[1])
        waveform = torch.cat([waveform, pad], dim=1)
    else:
        waveform = waveform[:, :target_len]

    # 转换为 Mel 频谱图并标准化
    mel_spec = T.MelSpectrogram(sample_rate=sr, n_mels=64, n_fft=1024, hop_length=512)
    mel = mel_spec(waveform)
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)

    # 模型推理
    with torch.no_grad():  # 不需要梯度计算
        outputs = model(input_values=mel.unsqueeze(0), return_dict=True)
        logits = outputs["logits"]
        pred_id = torch.argmax(logits, dim=1).item()  # 取概率最大的类别

    return id2label[pred_id]

# 打印使用示例
print(f"\n推理示例: 使用 model.from_pretrained('{MODEL_SAVE_PATH}') 加载模型")