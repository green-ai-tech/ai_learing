#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""轻量级 CNN 分类器 - Hugging Face Transformers 标准格式
   完全遵循 transformers 的 save_pretrained / from_pretrained 规范
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

# ==================== 配置 ====================
DATASET_PATH = "/Users/logicye/Code/my_Datasets/系统控制"
MODEL_SAVE_PATH = "/Users/logicye/Code/my_trained_models/model_speech_control_transformer"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# 双日志输出
log_file = open(os.path.join(MODEL_SAVE_PATH, "training.log"), "w", encoding="utf-8")

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

original_stdout = sys.stdout
sys.stdout = Tee(original_stdout, log_file)

# ==================== 1. 准备标签 ====================
labels = [d for d in os.listdir(DATASET_PATH) 
          if os.path.isdir(os.path.join(DATASET_PATH, d)) and not d.startswith("_")]
if not labels:
    print(f"错误：未在 {DATASET_PATH} 中找到标签目录")
    sys.exit(1)

label2id = {lbl: i for i, lbl in enumerate(labels)}
id2label = {i: lbl for i, lbl in enumerate(labels)}
print(f"识别到的指令标签: {labels}")
print(f"标签映射: {label2id}")

# ==================== 2. 定义配置类 ====================
class SimpleCNNConfig(PretrainedConfig):
    """自定义 CNN 模型的配置类，继承自 PretrainedConfig"""
    model_type = "simple_cnn"  # 模型类型标识，用于 AutoClass 支持

    def __init__(
        self,
        num_labels: int = 2,
        hidden_size: int = 768,
        num_conv_layers: int = 3,
        conv_channels: list = [16, 32, 64],
        kernel_size: int = 3,
        dropout_rate: float = 0.1,
        **kwargs,
    ):
        """
        参数:
            num_labels: 分类数量
            hidden_size: 隐藏层维度（兼容 transformers 标准）
            num_conv_layers: 卷积层数量
            conv_channels: 每层卷积通道数列表
            kernel_size: 卷积核大小
            dropout_rate: dropout 比率
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
    用于音频分类的 CNN 模型，完全兼容 Hugging Face Transformers 格式。
    支持 save_pretrained() 和 from_pretrained() 方法。
    """
    config_class = SimpleCNNConfig  # 关联配置类

    def __init__(self, config: SimpleCNNConfig):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        # 构建卷积层
        conv_layers = []
        in_channels = 1
        for out_channels in config.conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=config.kernel_size, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
        
        # 自适应池化层
        conv_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.conv = nn.Sequential(*conv_layers)

        # 分类头
        self.classifier = nn.Linear(config.conv_channels[-1] * 4 * 4, config.num_labels)

        # 初始化权重
        self.post_init()

    def forward(
        self,
        input_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, dict]:
        """
        前向传播方法
        Args:
            input_values: 输入的音频特征，形状 (batch_size, 1, n_mels, time_steps)
            labels: 标签，形状 (batch_size,)
            return_dict: 是否返回字典格式
        Returns:
            输出 logits 和可选的 loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 特征提取
        features = self.conv(input_values)
        features = features.view(features.size(0), -1)
        logits = self.classifier(features)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
        }

    # 可选：支持返回字典的便捷方法
    def forward_with_dict(self, *args, **kwargs):
        return self.forward(*args, **kwargs, return_dict=True)

# ==================== 4. 数据集类（与之前保持一致）====================
class MelDataset(Dataset):
    """将音频文件转换为 Mel 频谱图，用于模型输入"""
    def __init__(self, root_dir, label2id, duration=2, sr=16000):
        self.samples = []
        for label_name, label_id in label2id.items():
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.exists(label_dir):
                continue
            for f in os.listdir(label_dir):
                if f.endswith('.wav'):
                    self.samples.append((os.path.join(label_dir, f), label_id))
        self.duration = duration
        self.sr = sr
        self.n_samples = duration * sr
        self.mel_spec = T.MelSpectrogram(sample_rate=sr, n_mels=64, n_fft=1024, hop_length=512)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wav_path, label_id = self.samples[idx]
        waveform, sr = torchaudio.load(wav_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.sr:
            waveform = torchaudio.transforms.Resample(sr, self.sr)(waveform)
        if waveform.shape[1] < self.n_samples:
            pad = torch.zeros(1, self.n_samples - waveform.shape[1])
            waveform = torch.cat([waveform, pad], dim=1)
        else:
            waveform = waveform[:, :self.n_samples]
        mel = self.mel_spec(waveform)
        mel = (mel - mel.mean()) / (mel.std() + 1e-9)
        return mel, label_id

# ==================== 5. 数据加载 ====================
full_dataset = MelDataset(DATASET_PATH, label2id)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print(f"\n数据集划分：")
print(f"  训练集样本数: {train_size}")
print(f"  验证集样本数: {val_size}")

# 超参数配置（批次大小和轮数）
BATCH_SIZE = 32      # 批次大小：每批处理16个样本
EPOCHS = 50         # 训练轮数：让模型充分收敛
LEARNING_RATE = 0.001  # 学习率
WEIGHT_DECAY = 1e-4    # 权重衰减

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ==================== 6. 初始化模型 ====================
# 创建配置实例
config = SimpleCNNConfig(
    num_labels=len(labels),
    hidden_size=768,
    conv_channels=[16, 32, 64],
    kernel_size=3,
)

# 使用配置初始化模型
model = SimpleCNNForAudioClassification(config)
device = torch.device("cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"运行设备: {device}")

# ==================== 7. 训练循环 ====================
best_acc = 0.0
print("\n开始训练...")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    progress = tqdm(train_loader, desc=f"第 {epoch}/{EPOCHS} 轮 (训练)", unit="batch")
    for mel, label in progress:
        mel, label = mel.to(device), label.to(device)
        optimizer.zero_grad()
        # 调用模型的 forward 方法
        outputs = model(input_values=mel, labels=label, return_dict=True)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(train_loader)
    
    # 验证阶段
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for mel, label in val_loader:
            mel, label = mel.to(device), label.to(device)
            outputs = model(input_values=mel, return_dict=True)
            logits = outputs["logits"]
            pred = torch.argmax(logits, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    val_acc = 100.0 * correct / total
    
    print(f"第 {epoch} 轮 结束: 训练损失 = {avg_loss:.4f}, 验证准确率 = {val_acc:.2f}%")
    
    # 保存最佳模型（使用 save_pretrained 标准方法）
    if val_acc > best_acc:
        best_acc = val_acc
        # 保存模型和配置到目录
        model.save_pretrained(MODEL_SAVE_PATH)
        # 同时保存标签映射（供推理时使用）
        with open(os.path.join(MODEL_SAVE_PATH, "label_mapping.json"), "w", encoding="utf-8") as f:
            json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
        print(f"  -> 保存最佳模型 (准确率 {val_acc:.2f}%)")
    
    if val_acc >= 98.0:
        print(f"验证准确率已达到 {val_acc:.2f}%，提前停止训练。")
        break

# ==================== 8. 训练完成 ====================
print("\n" + "=" * 60)
print(f"训练完成！最佳验证准确率: {best_acc:.2f}%")
print(f"模型已保存至: {MODEL_SAVE_PATH}")
print(f"保存的文件包括:")
print(f"  - config.json          # 模型配置文件")
print(f"  - pytorch_model.bin     # 模型权重文件")
print(f"  - label_mapping.json   # 标签映射文件（自定义）")
print(f"  - training.log         # 训练日志")

# 恢复标准输出
sys.stdout = original_stdout
log_file.close()
print("训练日志已同时保存至文件。")

# ==================== 9. 推理示例 ====================
def load_model_for_inference(model_path):
    """加载训练好的模型进行推理"""
    # 使用 from_pretrained 加载模型
    model = SimpleCNNForAudioClassification.from_pretrained(model_path)
    model.eval()
    # 加载标签映射
    with open(os.path.join(model_path, "label_mapping.json"), "r", encoding="utf-8") as f:
        mapping = json.load(f)
    id2label = {int(k): v for k, v in mapping["id2label"].items()}
    return model, id2label

def predict_audio(wav_path, model, id2label, duration=2, sr=16000):
    """预测单个音频文件"""
    waveform, sr_orig = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr_orig != sr:
        waveform = torchaudio.transforms.Resample(sr_orig, sr)(waveform)
    
    target_len = duration * sr
    if waveform.shape[1] < target_len:
        pad = torch.zeros(1, target_len - waveform.shape[1])
        waveform = torch.cat([waveform, pad], dim=1)
    else:
        waveform = waveform[:, :target_len]
    
    mel_spec = T.MelSpectrogram(sample_rate=sr, n_mels=64, n_fft=1024, hop_length=512)
    mel = mel_spec(waveform)
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)
    
    with torch.no_grad():
        outputs = model(input_values=mel.unsqueeze(0), return_dict=True)
        logits = outputs["logits"]
        pred_id = torch.argmax(logits, dim=1).item()
    
    return id2label[pred_id]

print(f"\n推理示例: 使用 model.from_pretrained('{MODEL_SAVE_PATH}') 加载模型")