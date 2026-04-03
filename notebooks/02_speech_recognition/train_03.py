#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音指令采集工具 - 训练脚本
使用自己采集的数据集训练 AST 模型
"""

import os
import torch
import torchaudio
import numpy as np
from transformers import (
    ASTFeatureExtractor, 
    ASTConfig, 
    ASTForAudioClassification,
    TrainingArguments, 
    Trainer
)
from torch.utils.data import Dataset, Subset, random_split

# ==================== 全局路径配置 ====================
# 数据集路径（采集工具保存的路径）
DATASET_PATH = "/Users/logicye/Code/my_Datasets/方向数据集"

# 模型保存路径
MODEL_SAVE_PATH = "/Users/logicye/Code/my_trained_models/model_speech_up_down"

# ==================== 0. 检查并创建目录 ====================
print("正在检查数据集路径...")

# 检查数据集路径是否存在
if not os.path.exists(DATASET_PATH):
    print(f"警告：数据集路径不存在: {DATASET_PATH}")
    print("尝试查找可能的路径...")
    
    # 尝试其他可能的路径
    possible_paths = [
        "/Users/logicye/Code/Datasets/my_Datasets/方向数据集",
        "/Users/logicye/Code/Datasets/方向数据集",
        "/Users/logicye/Code/my_Datasets/方向数据集",
        "./dataset",  # 当前目录下的dataset
        "../dataset",  # 上级目录下的dataset
    ]
    
    found = False
    for path in possible_paths:
        if os.path.exists(path):
            DATASET_PATH = path
            print(f"找到数据集路径: {DATASET_PATH}")
            found = True
            break
    
    if not found:
        print("\n错误：找不到数据集目录！")
        print("请确保：")
        print("1. 已经运行过采集工具采集了一些音频")
        print("2. 或者手动创建目录并放入音频文件")
        print("\n可以运行以下命令创建目录：")
        print(f"mkdir -p {DATASET_PATH}/向上")
        print(f"mkdir -p {DATASET_PATH}/向下")
        print("\n或者修改代码中的 DATASET_PATH 变量为正确的路径")
        exit(1)

# 创建模型保存目录
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ==================== 1. 准备标签 ====================
print("\n正在加载数据集...")
labels = []  # 存放标签
for item in os.listdir(DATASET_PATH):
    # 遍历每个目录的名字
    item_path = os.path.join(DATASET_PATH, item)
    if os.path.isdir(item_path) and not item.startswith("_"):
        labels.append(item)

if len(labels) == 0:
    print(f"错误：在 {DATASET_PATH} 中未找到任何标签目录！")
    print("当前目录内容：")
    for item in os.listdir(DATASET_PATH):
        print(f"  - {item}")
    print("\n请确保目录结构如下：")
    print(f"{DATASET_PATH}/")
    print(f"  ├── 向上/")
    print(f"  │   ├── 向上_20231201_120000.wav")
    print(f"  │   └── ...")
    print(f"  └── 向下/")
    print(f"      ├── 向下_20231201_120001.wav")
    print(f"      └── ...")
    exit(1)

label2id = {d: idx for idx, d in enumerate(labels)}
id2label = {idx: d for idx, d in enumerate(labels)}

print(f"找到标签: {labels}")
print(f"标签映射: {label2id}")

# ==================== 2. 创建特征提取器 ====================
print("\n创建特征提取器...")
feature_extractor = ASTFeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    num_mel_bins=128,
    max_length=128,
    padding_value=0,
    do_normalize=True,
    mean=-6.845978,
    std=5.5654526,
    return_attention_mask=False
)

# ==================== 3. 自定义数据集类 ====================
class CustomAudioDataset(Dataset):
    """自定义音频数据集，用于加载采集工具生成的数据"""
    
    def __init__(self, root_dir, feature_extractor, label2id):
        """
        初始化自定义数据集
        
        Args:
            root_dir: 数据集根目录（包含 '向上'、'向下' 等子目录）
            feature_extractor: 特征提取器
            label2id: 标签到ID的映射
        """
        self.root_dir = root_dir
        self.feature_extractor = feature_extractor
        self.label2id = label2id
        self.samples = []
        
        # 遍历所有标签目录
        for label_name, label_id in label2id.items():
            label_dir = os.path.join(root_dir, label_name)
            if not os.path.exists(label_dir):
                print(f"警告：目录不存在 {label_dir}")
                continue
            
            # 遍历目录中的所有 wav 文件
            wav_files = [f for f in os.listdir(label_dir) if f.endswith('.wav')]
            print(f"在 {label_name} 目录中找到 {len(wav_files)} 个音频文件")
            
            for wav_file in wav_files:
                wav_path = os.path.join(label_dir, wav_file)
                self.samples.append((wav_path, label_id, label_name))
        
        print(f"\n总共加载了 {len(self.samples)} 个音频样本")
        
        # 统计每个类别的样本数
        if len(self.samples) > 0:
            label_counts = {}
            for _, label_id, label_name in self.samples:
                label_counts[label_name] = label_counts.get(label_name, 0) + 1
            
            print("\n类别分布:")
            for label_name, count in label_counts.items():
                print(f"  {label_name}: {count} 个样本")
        else:
            print("\n警告：没有找到任何音频文件！")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        wav_path, label_id, label_name = self.samples[idx]
        
        try:
            # 加载音频
            waveform, sample_rate = torchaudio.load(wav_path)
            
            # 如果是多声道，转换为单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # 重采样到16kHz（如果需要）
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # 确保音频长度足够（如果太短，进行填充）
            min_length = 16000  # 1秒
            if waveform.shape[1] < min_length:
                # 填充到最小长度
                padding = torch.zeros(1, min_length - waveform.shape[1])
                waveform = torch.cat([waveform, padding], dim=1)
            
            # 提取特征
            features = self.feature_extractor(
                waveform.squeeze().numpy(), 
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            
            return features["input_values"][0], torch.tensor(label_id, dtype=torch.long)
            
        except Exception as e:
            print(f"加载音频失败 {wav_path}: {e}")
            # 返回下一个样本
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(0)

# ==================== 4. 数据划分函数 ====================
def stratified_split(dataset, train_ratio=0.8, random_seed=42):
    """
    分层划分数据集，保持类别比例
    
    Args:
        dataset: 数据集
        train_ratio: 训练集比例
        random_seed: 随机种子
    
    Returns:
        train_indices, val_indices
    """
    # 获取所有样本的标签
    labels = [sample[1] for sample in dataset.samples]
    
    # 按类别分组
    class_indices = {}
    for idx, label in enumerate(labels):
        label_val = label.item() if torch.is_tensor(label) else label
        if label_val not in class_indices:
            class_indices[label_val] = []
        class_indices[label_val].append(idx)
    
    # 为每个类别分配索引
    train_indices = []
    val_indices = []
    
    np.random.seed(random_seed)
    
    for label, indices in class_indices.items():
        # 随机打乱
        shuffled_indices = np.random.permutation(indices)
        
        # 计算划分点
        split_point = int(len(shuffled_indices) * train_ratio)
        
        # 分配到训练集和验证集
        train_indices.extend(shuffled_indices[:split_point])
        val_indices.extend(shuffled_indices[split_point:])
    
    return train_indices, val_indices

# ==================== 5. 创建数据集 ====================
print("\n创建数据集...")
full_dataset = CustomAudioDataset(DATASET_PATH, feature_extractor, label2id)

if len(full_dataset) == 0:
    print("\n错误：没有找到任何音频文件！")
    print("\n请先运行采集工具采集一些音频样本。")
    print("采集工具会在这个路径下创建音频文件：")
    print(f"  {DATASET_PATH}/向上/")
    print(f"  {DATASET_PATH}/向下/")
    exit(1)

# 分层划分数据集
print("\n划分数据集...")
train_indices, val_indices = stratified_split(full_dataset, train_ratio=0.8)

# 创建子集
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

print(f"\n训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")

# 显示训练集和验证集的类别分布
def show_split_distribution(dataset, name):
    """显示数据集的类别分布"""
    if len(dataset) == 0:
        print(f"{name}: 空")
        return
    
    label_counts = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        label_val = label.item()
        label_name = id2label[label_val]
        label_counts[label_name] = label_counts.get(label_name, 0) + 1
    
    print(f"{name} 类别分布:")
    for label_name, count in label_counts.items():
        print(f"  {label_name}: {count} 个样本")

show_split_distribution(train_dataset, "训练集")
show_split_distribution(val_dataset, "验证集")

# ==================== 6. 创建模型 ====================
print("\n创建模型...")
config = ASTConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,  # 添加dropout防止过拟合（因为数据集小）
    attention_probs_dropout_prob=0.1,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    patch_size=16,
    qkv_bias=True,
    frequency_stride=10,
    time_stride=10,
    max_length=128,
    num_mel_bins=128,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

model = ASTForAudioClassification(config)
print(f"模型参数量: {model.num_parameters():,}")

# ==================== 7. 数据整理函数 ====================
def collate_fn(batch):
    """整理批次数据"""
    # 过滤掉None样本
    batch = [item for item in batch if item[0] is not None]
    
    if len(batch) == 0:
        return None
    
    # 取样本和标签
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    
    # 堆叠张量
    x = torch.stack(x)
    y = torch.stack(y)
    
    return {"input_values": x, "labels": y}

# ==================== 8. 评估函数 ====================
def compute_metrics(predict):
    """计算评估指标"""
    logits = predict.predictions
    label_ids = predict.label_ids
    pred = np.argmax(logits, axis=1)
    
    accuracy = (pred == label_ids).sum() * 100.0 / len(label_ids)
    
    return {
        "accuracy": accuracy,
    }

# ==================== 9. 训练参数配置 ====================
# 根据数据集大小动态调整训练参数
if len(train_dataset) < 10:
    print("\n警告：训练样本太少，建议至少采集10个以上样本")
    num_epochs = 200
    batch_size = 2
elif len(train_dataset) < 50:
    num_epochs = 100
    batch_size = 4
else:
    num_epochs = 50
    batch_size = 8

print(f"\n训练配置：")
print(f"  批次大小: {batch_size}")
print(f"  训练轮数: {num_epochs}")

args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    # 批次大小
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # 训练轮数
    num_train_epochs=num_epochs,
    # 优化器
    learning_rate=1e-4,
    optim="adamw_torch",
    weight_decay=1e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    # 保存策略
    save_strategy="epoch",
    save_total_limit=3,
    # 评估策略
    eval_strategy="epoch",
    # 日志
    logging_steps=5,
    logging_dir=os.path.join(MODEL_SAVE_PATH, "logs"),
    # 其他
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # 不使用早停（因为数据集小）
    dataloader_drop_last=False,
)

# ==================== 10. 创建Trainer并训练 ====================
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

print("\n开始训练...")
print("=" * 60)
trainer.train()

# ==================== 11. 保存模型 ====================
print("\n保存模型...")
model.save_pretrained(MODEL_SAVE_PATH)
feature_extractor.save_pretrained(MODEL_SAVE_PATH)
print(f"模型已保存到: {MODEL_SAVE_PATH}")

# ==================== 12. 最终评估 ====================
print("\n最终评估...")
eval_results = trainer.evaluate()
print(f"\n最终评估结果:")
print(f"  Loss: {eval_results['eval_loss']:.4f}")
print(f"  Accuracy: {eval_results['eval_accuracy']:.2f}%")

# ==================== 13. 模型推理测试 ====================
if len(val_dataset) > 0:
    print("\n测试模型推理...")
    model.eval()
    test_sample = val_dataset[0]
    input_values, label = test_sample
    
    with torch.no_grad():
        outputs = model(input_values.unsqueeze(0))
        predictions = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    print(f"测试样本预测: {id2label[predicted_class]} (置信度: {confidence:.2%})")
    print(f"实际标签: {id2label[label.item()]}")
    
print("\n训练完成！")

# ==================== 推理函数（供后续使用） ====================
def predict_audio(audio_path, model_path=MODEL_SAVE_PATH):
    """
    预测单个音频文件
    
    Args:
        audio_path: 音频文件路径
        model_path: 模型路径
    """
    # 加载模型和特征提取器
    model = ASTForAudioClassification.from_pretrained(model_path)
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_path)
    
    # 加载音频
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # 转换为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 重采样
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # 提取特征
    features = feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    )
    
    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(input_values=features["input_values"])
        predictions = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    return model.config.id2label[predicted_class], confidence

print("\n可以使用 predict_audio() 函数进行推理测试")
print(f"示例: result = predict_audio('path/to/audio.wav')")