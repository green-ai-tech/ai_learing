#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型验证脚本 - Hugging Face Transformers 格式模型（支持 .safetensors）
用于测试训练好的模型在实时录音和音频文件上的表现

作者：Logic Ye
日期：2026-04-04
"""

import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
import numpy as np

# ==================== 配置参数 ====================
# 模型路径：训练好的模型保存目录
# 应包含以下文件：
#   - config.json: 模型配置
#   - model.safetensors: 模型权重（安全格式）
#   - label_mapping.json: 标签映射
MODEL_PATH = "/Volumes/AI/models/my_trained/model_speech_control_transformer"
DEVICE = torch.device("cpu")  # 使用 CPU 进行推理

# ==================== 定义模型类（与训练时完全相同）====================
class SimpleCNNForAudioClassification(PreTrainedModel):
    """
    与训练脚本中完全一致的模型类定义
    
    注意：模型类必须与训练时保持一致，否则无法正确加载模型权重
    实际使用时会从 config.json 自动加载配置
    """
    config_class = PretrainedConfig  # 实际会从 config.json 加载自定义配置

    def __init__(self, config):
        """
        初始化模型结构
        
        参数:
            config: 配置对象，包含模型的所有超参数
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # 构建卷积层序列
        conv_layers = []
        in_channels = 1
        for out_channels in config.conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=config.kernel_size, padding=1),  # 2D卷积
                nn.BatchNorm2d(out_channels),   # 批归一化
                nn.ReLU(),                      # ReLU激活
                nn.MaxPool2d(2)                 # 最大池化
            ])
            in_channels = out_channels
        
        # 自适应平均池化层：输出固定尺寸 4x4
        conv_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.conv = nn.Sequential(*conv_layers)
        
        # 分类层
        self.classifier = nn.Linear(config.conv_channels[-1] * 4 * 4, config.num_labels)
        self.post_init()

    def forward(self, input_values, labels=None, return_dict=True):
        """
        前向传播
        
        参数:
            input_values: 输入特征（Mel频谱图）
            labels: 标签（可选，用于计算损失）
            return_dict: 是否返回字典格式
        
        返回:
            包含 logits 和可选 loss 的字典或元组
        """
        x = self.conv(input_values)
        x = x.view(x.size(0), -1)  # 展平
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        
        if return_dict:
            return {"loss": loss, "logits": logits}
        return (loss, logits) if loss is not None else (logits,)

# ==================== 加载模型 ====================
# 使用 from_pretrained 自动加载 config.json 和 model.safetensors
# 这是 Hugging Face Transformers 的标准加载方式
model = SimpleCNNForAudioClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()  # 设置为评估模式（关闭 dropout 等）
print("模型加载成功")

# 加载标签映射
# label_mapping.json 是自定义保存的文件，包含 label2id 和 id2label 两个映射
with open(os.path.join(MODEL_PATH, "label_mapping.json"), "r", encoding="utf-8") as f:
    mapping = json.load(f)
id2label = {int(k): v for k, v in mapping["id2label"].items()}  # ID -> 标签文本
label2id = mapping["label2id"]  # 标签文本 -> ID
print(f"标签: {id2label}")

# ==================== 预处理函数 ====================
def preprocess_audio(wav_path, duration=2, sr=16000):
    """
    预处理音频文件：加载 -> 转换为 Mel 频谱图 -> 标准化
    
    工作流程:
    1. 加载音频文件
    2. 转换为单声道（如果是立体声）
    3. 重采样到目标采样率（如果需要）
    4. 截断或填充到目标长度
    5. 计算 Mel 频谱图
    6. 标准化（均值为0，标准差为1）
    
    参数:
        wav_path: 音频文件路径（.wav 格式）
        duration: 目标持续时间（秒）
        sr: 目标采样率（Hz）
    
    返回:
        mel: 预处理后的 Mel 频谱图张量
    """
    # 加载音频
    waveform, sr_orig = torchaudio.load(wav_path)
    
    # 如果是立体声，转换为单声道（取平均值）
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 如果采样率不匹配，进行重采样
    if sr_orig != sr:
        waveform = torchaudio.transforms.Resample(sr_orig, sr)(waveform)
    
    # 计算目标采样点数
    target_len = duration * sr
    
    # 如果音频太短，进行零填充
    if waveform.shape[1] < target_len:
        pad = torch.zeros(1, target_len - waveform.shape[1])
        waveform = torch.cat([waveform, pad], dim=1)
    else:
        # 如果音频太长，截断到目标长度
        waveform = waveform[:, :target_len]
    
    # 计算 Mel 频谱图
    mel_spec = T.MelSpectrogram(
        sample_rate=sr,   # 采样率
        n_mels=64,        # Mel滤波器数量（频率维度）
        n_fft=1024,       # FFT窗口大小
        hop_length=512    # 帧移
    )
    mel = mel_spec(waveform)
    
    # 标准化：使特征分布更稳定
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)
    
    return mel

def predict_single_file(wav_path):
    """
    预测单个音频文件的类别
    
    参数:
        wav_path: 音频文件路径
    
    返回:
        predicted_label: 预测的标签名称
    """
    # 预处理音频
    mel = preprocess_audio(wav_path)
    
    # 模型推理（不需要梯度计算）
    with torch.no_grad():
        # 添加批次维度（模型期望输入为 batch x channels x mel x time）
        outputs = model(input_values=mel.unsqueeze(0), return_dict=True)
        logits = outputs["logits"]
        pred_id = torch.argmax(logits, dim=1).item()  # 取概率最大的类别
    
    return id2label[pred_id]

# ==================== 实时录音（需 pyaudio）====================
def record_audio(duration=2, sample_rate=16000):
    """
    使用 PyAudio 录制实时音频
    
    工作流程:
    1. 初始化 PyAudio
    2. 打开音频输入流（麦克风）
    3. 按块读取音频数据
    4. 停止并关闭音频流
    5. 将原始字节数据转换为 numpy 数组并归一化
    
    参数:
        duration: 录音持续时间（秒）
        sample_rate: 采样率（Hz）
    
    返回:
        waveform: 归一化后的音频张量，形状为 (1, 采样点数)
    """
    try:
        import pyaudio
        import numpy as np
    except ImportError:
        print("请先安装 pyaudio: pip install pyaudio")
        return None
    
    # 初始化 PyAudio
    p = pyaudio.PyAudio()
    
    # 打开音频输入流
    stream = p.open(format=pyaudio.paInt16,        # 16位整数格式
                    channels=1,                     # 单声道
                    rate=sample_rate,               # 采样率
                    input=True,                     # 输入模式
                    frames_per_buffer=1024)         # 每次读取1024个采样点
    
    frames = []  # 存储所有音频帧
    
    # 录制音频
    # 计算总共需要读取的帧数：采样率 / 每帧采样点 * 持续时间
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    
    # 停止并关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # 将字节数据转换为 numpy 数组
    # 1. 将所有帧连接成一个字节串
    # 2. 转换为 int16 类型的 numpy 数组
    # 3. 归一化到 [-1, 1] 范围（除以 32768.0，即 2^15）
    audio = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    
    # 转换为 PyTorch 张量并添加通道维度
    return torch.from_numpy(audio).float().unsqueeze(0)

def realtime_test():
    """
    实时录音测试
    
    工作流程:
    1. 等待用户按回车键
    2. 录制 2 秒音频
    3. 预处理音频（填充/截断、Mel频谱图、标准化）
    4. 模型推理
    5. 显示识别结果
    """
    print("\n=== 实时录音测试 ===")
    input("按回车开始录音（说完后等待2秒）...")
    
    # 录制音频
    waveform = record_audio(duration=2)
    if waveform is None:
        return  # 如果录音失败，直接返回
    
    # 预处理音频
    # 确保音频长度至少为 32000 个采样点（2秒 @ 16kHz）
    if waveform.shape[1] < 32000:
        pad = torch.zeros(1, 32000 - waveform.shape[1])
        waveform = torch.cat([waveform, pad], dim=1)
    else:
        waveform = waveform[:, :32000]
    
    # 计算 Mel 频谱图
    mel_spec = T.MelSpectrogram(
        sample_rate=16000,
        n_mels=64,
        n_fft=1024,
        hop_length=512
    )
    mel = mel_spec(waveform)
    
    # 标准化
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)
    
    # 模型推理
    with torch.no_grad():
        outputs = model(input_values=mel.unsqueeze(0), return_dict=True)
        logits = outputs["logits"]
        pred_id = torch.argmax(logits, dim=1).item()
    
    print(f"识别结果: {id2label[pred_id]}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 默认执行实时测试
    realtime_test()

    # 测试单个文件，以下代码：
    # result = predict_single_file("测试音频.wav")
    # print(f"预测结果: {result}")
    
    