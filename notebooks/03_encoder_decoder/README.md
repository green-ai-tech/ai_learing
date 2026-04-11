# 03_编码器、解码器与 Transformer

## 📚 学习主题

本模块深入学习**编码器-解码器架构**及其在深度学习中的应用，从传统的自编码器到现代 Transformer 模型。

---

## 📁 目录结构

```
03_encoder_decoder/
├── 01_图像编码解码/              # 主题 1: 图像自编码器
│   ├── 01_image_encoding_decoding.ipynb
│   ├── 01_image_encoding_decoding.py
│   ├── 01_sample.jpg
│   └── outputs/                  # 输出目录（图像生成结果）
│
├── 02_编码器与解码器/            # 主题 2: 基础编码器与解码器实现
│   ├── 02_encoder_and_decoder.ipynb
│   ├── 02_prediction_sample.jpg
│   ├── MNIST/                    # MNIST 手写数字数据集
│   └── outputs/                  # 输出目录
│
├── 03_Transformer文本分类/       # 主题 3: Transformer 文本分类（核心）
│   ├── 03_new.py                          # 学习草稿版（带详细中文注释）
│   ├── 03_my.py                           # 改进版（术语准确，结构清晰）
│   ├── 03_ai_code.py                      # 完整版（含训练、评估、推理）
│   ├── 03_transformer.py                  # HuggingFace Transformers 示例
│   ├── 03_transformer模型与分类.ipynb     # Jupyter Notebook 版本
│   ├── data/                              # 数据目录
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── sgns.wiki.char.bz2
│   ├── vocab/                             # 词表目录（自动生成）
│   │   ├── labels.pk
│   │   └── chars.pk
│   ├── models/                            # 模型目录（自动生成）
│   │   └── best_transformer_model.pth
│   └── outputs/                           # 输出目录（自动生成）
│       └── confusion_matrix.png
│
├── 04_自注意力机制/              # 主题 4: 自注意力机制详解
│   └── 04_自注意力机制.ipynb
│
└── README.md                     # 本文件
```

---

## 🎯 各主题学习内容

### 01_图像编码解码
**学习目标**: 理解图像自编码器的基本原理
- 编码器：将图像压缩为低维表示
- 解码器：从低维表示还原图像
- 训练过程：最小化重建误差
- 应用场景：图像压缩、去噪、特征提取

### 02_编码器与解码器
**学习目标**: 掌握基础的编码器-解码器架构
- 使用全连接层实现编码器和解码器
- 在 MNIST 数据集上的应用
- 特征空间的可视化
- 预测结果分析

### 03_Transformer文本分类 ⭐ **核心内容**
**学习目标**: 完整实现基于 Transformer 的文本分类系统

#### 📄 三个版本说明：
| 文件 | 说明 | 适合阶段 |
|------|------|----------|
| `03_new.py` | 学习草稿版，带详细中文注释 | 初次学习，理解每个步骤 |
| `03_my.py` | 改进版，术语准确，结构清晰 | 深入学习，理解架构设计 |
| `03_ai_code.py` | 完整版，包含训练、评估、推理 | 完整项目，实际运行 |

#### 🔧 核心模块：
1. **数据集预处理**
   - 从 CSV 提取高频字符和标签
   - 序列化保存为 `.pk` 文件
   
2. **数据集编码**
   - 文本 → 数字序列（词汇映射）
   - 统一序列长度（截断/填充）
   - PyTorch Dataset 封装

3. **词嵌入 (Embedding)**
   - 加载 Word2Vec 预训练模型
   - 构建嵌入矩阵
   - 语义向量查找表

4. **Transformer 模型**
   - 位置编码 (Positional Encoding)
   - 多头自注意力机制
   - Transformer Encoder 堆叠
   - 分类器设计

5. **模型训练**
   - 完整的训练循环
   - 学习率调度
   - 最佳模型自动保存
   - 训练进度条显示

6. **模型评估**
   - 分类报告 (Precision, Recall, F1)
   - 混淆矩阵可视化
   - 多维度指标分析

7. **模型推理**
   - 单条文本预测
   - 批量预测
   - Top-K 概率输出

### 04_自注意力机制
**学习目标**: 深入理解自注意力机制的数学原理和代码实现
- 注意力分数计算
- 多头注意力
- 位置编码的作用
- 与 RNN/CNN 的对比

---

## 🚀 快速开始

### 环境准备
```bash
# 激活虚拟环境
source /Users/logicye/Code/ai_learning/venv/bin/activate

# 安装依赖
pip install torch pandas numpy gensim scikit-learn tqdm matplotlib seaborn
```

### 运行顺序建议

#### 第一阶段：理解基础概念
```bash
# 1. 学习图像自编码器
cd 01_图像编码解码
jupyter notebook 01_image_encoding_decoding.ipynb

# 2. 学习基础编码器-解码器
cd ../02_编码器与解码器
jupyter notebook 02_encoder_and_decoder.ipynb
```

#### 第二阶段：Transformer 文本分类（核心）
```bash
cd 03_Transformer文本分类

# 步骤 1: 先学习草稿版（理解每一步）
python 03_new.py

# 步骤 2: 再看改进版（更好的架构）
python 03_my.py

# 步骤 3: 运行完整版（实际训练）
python 03_ai_code.py
```

#### 第三阶段：深入理论
```bash
cd ../04_自注意力机制
jupyter notebook 04_自注意力机制.ipynb
```

---

## 📊 数据集说明

### 文本分类数据集
- **来源**: CSV 格式的训练集和测试集
- **位置**: `ds/dataset/train.csv`, `ds/dataset/test.csv`
- **字段**:
  - `content`: 文本内容（一段话或一篇文章）
  - `label`: 标签类别（如"体育"、"娱乐"、"科技"等）

### 预训练模型
- **Word2Vec**: `ds/dataset/sgns.wiki.char.bz2`
- **说明**: 在维基百科上训练的字符级词向量
- **维度**: 300 维

---

## 🔑 核心概念

### 编码器 (Encoder)
将输入数据压缩为**低维表示**（特征向量）的模块。
- **图像**: CNN 压缩为特征图
- **文本**: Embedding + Transformer 压缩为上下文向量

### 解码器 (Decoder)
从**低维表示**还原为原始数据格式的模块。
- **图像**: 从特征图还原为像素值
- **文本**: 从上下文向量生成文本（生成任务）

### Transformer
基于**自注意力机制**的序列模型架构。
- **优势**: 并行计算、长距离依赖、上下文感知
- **应用**: 文本分类、机器翻译、大语言模型

### 词嵌入 (Word Embedding)
将离散的字符/词映射到**连续的向量空间**。
- **预训练**: Word2Vec, GloVe, FastText
- **微调**: 在特定任务中继续更新向量

---

## 💡 学习建议

1. **先看整体，再抠细节**
   - 先跑通 `03_ai_code.py` 看完整流程
   - 再回头看 `03_new.py` 理解每一步

2. **对比学习**
   - 对比图像编码器和文本编码器的异同
   - 对比传统 RNN 和 Transformer 的架构差异

3. **动手实验**
   - 修改超参数（学习率、批次大小、层数）
   - 观察训练曲线变化
   - 尝试不同的预训练模型

4. **理论结合实践**
   - 学完 `04_自注意力机制.ipynb` 后
   - 回到 `03_ai_code.py` 看代码如何实现理论

---

## 📝 常见问题

### Q1: 为什么要用预训练的词向量？
**A**: 预训练向量包含了从大规模语料中学到的语义信息，可以让模型从一开始就"理解"字符的含义，而不是从零开始瞎猜。这能显著提升训练效率和模型性能。

### Q2: 序列长度为什么要统一为 200？
**A**: 神经网络要求输入维度固定。短文本用 `PAD` (0) 填充，长文本截断。200 是一个经验值，覆盖了大部分文本的长度。

### Q3: `03_new.py`、`03_my.py`、`03_ai_code.py` 该看哪个？
**A**: 三个都看，但顺序很重要：
1. 先看 `03_new.py`（理解每一步在做什么）
2. 再看 `03_my.py`（学习更好的代码架构）
3. 最后跑 `03_ai_code.py`（看完整项目如何组织）

---

## 🔗 相关资源

- [PyTorch 官方文档](https://pytorch.org/docs/)
- [Transformers 库文档](https://huggingface.co/docs/transformers/)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Transformer 原论文)

---

## 📅 更新记录

| 日期 | 更新内容 |
|------|----------|
| 2026-04-10 | 初始版本，整理文件结构，统一路径配置 |
