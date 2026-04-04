#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 Hugging Face Transformers 格式模型（支持 .safetensors）"""

import os
import json
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
import numpy as np

# ==================== 配置 ====================
MODEL_PATH = "/Users/logicye/Code/my_trained_models/model_speech_control_transformer"
DEVICE = torch.device("cpu")

# ==================== 定义模型类（与训练时完全相同）====================
class SimpleCNNForAudioClassification(PreTrainedModel):
    """与训练脚本中完全一致的模型类"""
    config_class = PretrainedConfig  # 实际会从 config.json 加载自定义配置
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
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
        conv_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        self.conv = nn.Sequential(*conv_layers)
        self.classifier = nn.Linear(config.conv_channels[-1] * 4 * 4, config.num_labels)
        self.post_init()
    
    def forward(self, input_values, labels=None, return_dict=True):
        x = self.conv(input_values)
        x = x.view(x.size(0), -1)
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
model = SimpleCNNForAudioClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
print("模型加载成功")

# 加载标签映射
with open(os.path.join(MODEL_PATH, "label_mapping.json"), "r", encoding="utf-8") as f:
    mapping = json.load(f)
id2label = {int(k): v for k, v in mapping["id2label"].items()}
label2id = mapping["label2id"]
print(f"标签: {id2label}")

# ==================== 预处理函数 ====================
def preprocess_audio(wav_path, duration=2, sr=16000):
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
    return mel

def predict_single_file(wav_path):
    mel = preprocess_audio(wav_path)
    with torch.no_grad():
        outputs = model(input_values=mel.unsqueeze(0), return_dict=True)
        logits = outputs["logits"]
        pred_id = torch.argmax(logits, dim=1).item()
    return id2label[pred_id]

# ==================== 实时录音（需 pyaudio）====================
def record_audio(duration=2, sample_rate=16000):
    try:
        import pyaudio
        import numpy as np
    except ImportError:
        print("请先安装 pyaudio: pip install pyaudio")
        return None
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)
    frames = []
    for _ in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    audio = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32) / 32768.0
    return torch.from_numpy(audio).float().unsqueeze(0)

def realtime_test():
    print("\n=== 实时录音测试 ===")
    input("按回车开始录音（说完后等待2秒）...")
    waveform = record_audio(duration=2)
    if waveform is None:
        return
    # 预处理
    if waveform.shape[1] < 32000:
        pad = torch.zeros(1, 32000 - waveform.shape[1])
        waveform = torch.cat([waveform, pad], dim=1)
    else:
        waveform = waveform[:, :32000]
    mel_spec = T.MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512)
    mel = mel_spec(waveform)
    mel = (mel - mel.mean()) / (mel.std() + 1e-9)
    with torch.no_grad():
        outputs = model(input_values=mel.unsqueeze(0), return_dict=True)
        logits = outputs["logits"]
        pred_id = torch.argmax(logits, dim=1).item()
    print(f"识别结果: {id2label[pred_id]}")

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 实时测试（默认）
    realtime_test()
    
    # 如需测试单个文件，取消注释：
    # result = predict_single_file("测试音频.wav")
    # print(f"预测结果: {result}")
    