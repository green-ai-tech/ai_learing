# ============================================================================
# 1. 数据集预处理模块
# ============================================================================
"""
功能说明:
    从原始 CSV 文件中提取高频字符和标签类别，并序列化保存为二进制文件 (.pk)
    
输出文件:
    - chars.pk:   高频字符列表 (词袋)
    - labels.pk:  所有不重复的标签类别
"""
import pickle
import os


# ============================================================================
# 1.1 文件操作工具类
# ============================================================================
class PickleFileOprator:
    """
    Pickle 文件读写器
    用于将 Python 对象序列化保存到文件，或从文件反序列化加载对象
    """

    def __init__(self, data=None, file_path=""):
        """
        初始化
        参数:
            data: 要保存的 Python 对象
            file_path: 文件路径
        """
        self.data = data
        self.file_path = file_path

    def save(self):
        """将 data 对象序列化保存到指定文件"""
        with open(self.file_path, "wb") as fd:
            pickle.dump(self.data, fd)

    def read(self):
        """从文件读取并反序列化为 Python 对象"""
        with open(self.file_path, "rb") as fd:
            self.data = pickle.load(fd)
            return self.data


# ============================================================================
# 1.2 全局路径配置
# ============================================================================
# 获取当前脚本所在目录（项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据目录（所有输入数据）
DATA_PATH    = os.path.join(PROJECT_ROOT, "data")
TRAIN_PATH   = os.path.join(DATA_PATH, "train.csv")
TEST_PATH    = os.path.join(DATA_PATH, "test.csv")
WORD2VEC_PATH = os.path.join(DATA_PATH, "sgns.wiki.char.bz2")

# 词表目录（保存 labels.pk, chars.pk）
VOCAB_PATH   = os.path.join(PROJECT_ROOT, "vocab")

# 输出目录（保存模型、混淆矩阵等）
OUTPUTS_PATH = os.path.join(PROJECT_ROOT, "outputs")
MODELS_PATH  = os.path.join(PROJECT_ROOT, "models")
LOGS_PATH    = os.path.join(PROJECT_ROOT, "logs")  # 日志保存目录

# 自动创建目录
os.makedirs(VOCAB_PATH, exist_ok=True)
os.makedirs(OUTPUTS_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

NUM_WORDS    = 6000  # 保留最高频的字符数量


# ============================================================================
# 1.3 双日志模块 (终端 + 文件)
# ============================================================================
import logging
from datetime import datetime

class DualLogger:
    """
    双日志系统：同时输出到终端和文件
    
    功能:
        - 终端输出：实时查看训练进度
        - 文件输出：保存完整训练日志，方便后续分析
    """
    
    def __init__(self, log_file=None):
        """
        初始化日志器
        
        参数:
            log_file: 日志文件路径，默认使用 logs/ 目录下的时间戳文件名
        """
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(LOGS_PATH, f"train_{timestamp}.log")
        
        # 确保日志文件父目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # 配置 logger
        self.logger = logging.getLogger("TransformerTraining")
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加 handler
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 日志格式
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 终端 Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件 Handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.log_file = log_file
    
    def info(self, msg):
        """输出 INFO 级别日志"""
        self.logger.info(msg)
    
    def warning(self, msg):
        """输出 WARNING 级别日志"""
        self.logger.warning(msg)
    
    def error(self, msg):
        """输出 ERROR 级别日志"""
        self.logger.error(msg)
    
    def close(self):
        """关闭所有 Handler"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


# 全局日志实例 (延迟初始化)
_logger = None

def get_logger(log_file=None):
    """获取或创建日志器 (单例模式)"""
    global _logger
    if _logger is None:
        _logger = DualLogger(log_file)
    return _logger


# ============================================================================
# 1.4 文件处理类
# ============================================================================
import pandas as pd
from collections import Counter, defaultdict
class FileProcess:
    """
    训练集文件处理器
    负责从 CSV 文件中统计字符频率、提取标签，并保存词表
    """

    def __init__(self, keep_n_word) -> None:
        """
        初始化
        参数:
            keep_n_word: 要保留的最高频字符数量
        """
        self.keep_n_word = keep_n_word

    def _read_train_file(self):
        """
        从训练集中统计字符频率
        
        处理流程:
            1. 读取 CSV 文件
            2. 提取所有不重复的标签
            3. 统计 content 列中每个字符出现的频次
            4. 按频次降序排列，取前 n 个高频字符
        
        返回:
            label_list:   标签列表，例如 ['体育', '娱乐', '科技', ...]
            top_n_chars:  高频字符列表，例如 ['的', '了', '是', ...]
        """
        # 读取训练集 CSV
        train_pd = pd.read_csv(TRAIN_PATH)
        
        # 提取所有不重复的标签类别
        label_list = train_pd["label"].unique().tolist()

        # 统计所有文本中字符的出现频次
        character_counter = Counter()
        for content in train_pd["content"]:
            character_counter.update(content)

        # 按频次降序排序
        sorted_chars = character_counter.most_common()

        # 取前 n 个最高频的字符
        top_n_chars = [char for char, freq in sorted_chars[:self.keep_n_word]]

        return label_list, top_n_chars

    def build_and_save_vocab(self):
        """
        构建词表并保存到文件
        
        输出:
            - VOCAB_PATH/labels.pk:  标签列表
            - VOCAB_PATH/chars.pk:   高频字符列表
        """
        label_list, top_n_chars = self._read_train_file()
        
        # 保存标签列表
        PickleFileOprator(
            data=label_list,
            file_path=os.path.join(VOCAB_PATH, "labels.pk")
        ).save()
        
        # 保存高频字符列表
        PickleFileOprator(
            data=top_n_chars,
            file_path=os.path.join(VOCAB_PATH, "chars.pk")
        ).save()


# ============================================================================
# 2. 数据集编码模块
# ============================================================================
"""
功能说明:
    将文字形式的标签和序列转换为数字 ID，以便神经网络能够处理
    
核心概念:
    - 词汇映射 (Vocab Mapping):  字符/标签 -> 数字 ID 的对照表
    - 序列编码 (Sequence Encoding): 将文本序列转换为固定长度的数字序列
"""

def load_vocab_mappings():
    """
    加载字符和标签的映射字典
    如果词表文件不存在，会自动从训练集生成
    
    返回:
        label_to_id: 标签映射字典,例如 {'体育': 0, '娱乐': 1, ...}
        char_to_id:  字符映射字典,例如 {'的': 0, '了': 1, '是': 2, ...}
    """
    labels_path = os.path.join(VOCAB_PATH, "labels.pk")
    chars_path = os.path.join(VOCAB_PATH, "chars.pk")
    
    # 如果词表文件不存在，自动从训练集生成
    if not os.path.exists(labels_path) or not os.path.exists(chars_path):
        print("⚠️  词表文件不存在，正在从训练集生成...")
        processor = FileProcess(keep_n_word=NUM_WORDS)
        processor.build_and_save_vocab()
        print("✅ 词表生成完成!")
    
    # 从文件加载标签列表
    labels = PickleFileOprator(file_path=labels_path).read()
    
    # 从文件加载字符列表
    chars = PickleFileOprator(file_path=chars_path).read()

    # 构建标签 -> ID 的映射
    label_to_id = dict(zip(labels, range(len(labels))))
    
    # 构建字符 -> ID 的映射
    char_to_id = dict(zip(chars, range(len(chars))))

    return label_to_id, char_to_id


# ============================================================================
# 2.1 特殊标记和超参数配置
# ============================================================================
PAD_NO = 0               # 填充标记的 ID (Padding) - 用于补齐短序列
UNK_NO = 1               # 未知字符的 ID (Unknown) - 用于词表外的生僻字
START_NO = UNK_NO + 1    # 实际字符的起始 ID (从 2 开始，为特殊标记预留空间)
SEQUENCE_LENGTH = 200    # 序列统一长度 (所有文本都会被截断或补齐到这个长度)
EMBEDDING_DIM = 300      # 词向量维度 (Word2Vec 预训练模型的向量维度)


# ============================================================================
# 2.2 序列编码函数
# ============================================================================
def text_to_vectors(labels_list, contents_list, label_to_id, char_to_id):
    """
    将文本序列和标签转换为数字向量
    
    参数:
        labels_list:   原始标签列表,例如 ['体育', '娱乐', '体育', ...]
        contents_list: 原始文本序列列表,例如 ['今天打球', '看电影听音乐', ...]
        label_to_id:   标签映射字典,例如 {'体育': 0, '娱乐': 1, ...}
        char_to_id:    字符映射字典,例如 {'今': 0, '天': 1, '打': 2, ...}
    
    返回:
        encoded_sequences: 编码后的序列向量列表,每个元素是长度为 200 的数字列表
        encoded_labels:    编码后的标签 ID 列表
    
    处理流程:
        1. 标签 -> 标签 ID
        2. 文本 -> 字符 ID 序列 (查表 + 预留偏移)
        3. 统一长度 (截断长序列 / 填充短序列)
    """
    encoded_sequences = []  # 存放所有编码后的序列向量
    encoded_labels = []     # 存放所有编码后的标签 ID

    # 遍历每一条数据 (标签, 文本)
    for label, text in zip(labels_list, contents_list):

        # ----- 第一步: 转换标签为数字 ID -----
        label_id = label_to_id[label]
        encoded_labels.append(label_id)

        # ----- 第二步: 转换文本为字符 ID 序列 -----
        char_ids = []  # 当前序列的字符编号列表

        for char in text:
            if char in char_to_id:
                # 字符在词表中: 使用它的 ID + 偏移量 (为 PAD/UNK 预留空间)
                char_id = char_to_id[char] + START_NO
                char_ids.append(char_id)
            else:
                # 字符不在词表中 (生僻字): 用 UNK 标记 (ID = 1)
                char_ids.append(UNK_NO)

        # ----- 第三步: 统一序列长度 -----
        if len(char_ids) > SEQUENCE_LENGTH:
            # 序列太长: 截断到固定长度 (只保留前面的字符)
            char_ids = char_ids[:SEQUENCE_LENGTH]
        elif len(char_ids) < SEQUENCE_LENGTH:
            # 序列太短: 用 PAD (0) 填充到固定长度
            padding_length = SEQUENCE_LENGTH - len(char_ids)
            char_ids = char_ids + [PAD_NO] * padding_length

        # 将处理好的序列加入结果列表
        encoded_sequences.append(char_ids)

    return encoded_sequences, encoded_labels


# ============================================================================
# 2.3 PyTorch 数据集封装
# ============================================================================
import numpy as np
import torch as T
from torch.utils.data import Dataset, random_split

class CSVDataset(Dataset):
    """
    CSV 数据集封装类 (继承自 PyTorch Dataset)
    
    功能:
        - 自动加载 CSV 文件并完成编码
        - 提供标准的 PyTorch 数据接口 (__len__, __getitem__)
        - 支持训练集/验证集随机划分
    """
    
    def __init__(self, file_path):
        """
        初始化数据集
        
        参数:
            file_path: CSV 文件路径 (例如 train.csv)
        
        处理流程:
            1. 加载词汇映射字典
            2. 读取 CSV 文件中的原始文本和标签
            3. 调用 text_to_vectors 编码为数字序列
            4. 转换为 PyTorch 张量 (Tensor)
        """
        # 加载词汇映射字典
        label_to_id, char_to_id = load_vocab_mappings()
        
        # 读取 CSV 文件
        contents, labels = load_csv_file(file_path)

        # 将文本编码为数字序列
        x, y = text_to_vectors(labels, contents, label_to_id, char_to_id)

        # 转换为 PyTorch 张量 (.long() 表示长整型，适用于分类任务)
        self.x = T.from_numpy(np.array(x)).long()  # 特征张量: [样本数, 序列长度]
        self.y = T.from_numpy(np.array(y)).long()  # 标签张量: [样本数]

    def __len__(self):
        """返回数据集中的样本总数"""
        return self.x.shape[0]

    def __getitem__(self, index):
        """
        根据索引获取单条数据
        
        返回:
            [特征序列, 标签ID] 的列表
        """
        return [self.x[index], self.y[index]]

    def get_splits(self, n_valid=0.3):
        """
        随机划分训练集和验证集
        
        参数:
            n_valid: 验证集所占比例 (默认 30%)
        
        返回:
            [训练集, 验证集] 的 Dataset 列表
        """
        # 计算验证集样本数
        valid_size = round(n_valid * len(self.x))
        train_size = len(self.x) - valid_size
        
        # 随机切分数据集
        return random_split(self, [train_size, valid_size])


# ============================================================================
# 3. 词嵌入 (Embedding) 模块
# ============================================================================
"""
功能说明:
    加载预训练的 Word2Vec 模型，为我们的字符构建语义向量查找表
    
核心概念:
    - 词嵌入 (Word Embedding): 将离散的字符映射到连续的向量空间
    - 预训练模型 (Pretrained Model): 在大规模语料上提前训练好的向量，包含语义信息
"""

def load_csv_file(file_path):
    """
    读取 CSV 文件，返回内容和标签列表
    
    参数:
        file_path: CSV 文件路径
    
    返回:
        contents: 文本内容列表,例如 ['今天打球', '看电影', ...]
        labels:   标签列表,例如 ['体育', '娱乐', ...]
    """
    df = pd.read_csv(file_path)
    contents = df["content"].tolist()
    labels = df["label"].tolist()
    return contents, labels


def load_pretrained_embeddings(char_to_id, embedding_dim=EMBEDDING_DIM, model_path=WORD2VEC_PATH):
    """
    加载预训练词向量，构建嵌入矩阵 (Embedding Matrix)
    
    参数:
        char_to_id:    字符到 ID 的映射字典,例如 {'今': 0, '天': 1, ...}
        embedding_dim: 词向量的维度 (默认 300)
        model_path:    Word2Vec 预训练模型文件路径
    
    返回:
        embedding_matrix: 嵌入矩阵,形状为 (词表大小, 词向量维度)
                         每一行是对应字符的 300 维语义向量
    
    处理流程:
        1. 加载预训练的 Word2Vec 模型
        2. 创建全零矩阵 (词表大小 x 向量维度)
        3. 遍历字符字典，从 Word2Vec 中查找并填入对应的向量
        4. 未找到的字符保持为全零向量
    """
    from gensim.models import KeyedVectors

    print(f"正在加载预训练词向量模型: {model_path}")
    
    # 加载预训练的 Word2Vec 模型
    pretrained_model = KeyedVectors.load_word2vec_format(
        model_path,
        binary=False,
        encoding="utf-8",
        unicode_errors="ignore"
    )
    print("预训练模型加载完成!")

    # 计算词表大小: 最大 ID + 1 + 预留空间 (PAD=0, UNK=1, START_NO=2, 以及额外预留)
    vocab_size = max(char_to_id.values()) + 1 + 4
    
    # 创建全零矩阵，用于存储所有字符的向量
    embedding_matrix = T.zeros(vocab_size, embedding_dim).float()

    # 统计在预训练模型中找到/未找到的字符数量
    found_count = 0
    not_found_count = 0

    print("开始构建嵌入矩阵...")
    for char, char_id in char_to_id.items():
        # 字符在矩阵中的实际行号 (需要加上预留偏移量)
        matrix_index = char_id + START_NO

        if char in pretrained_model.key_to_index:
            # 字符在预训练模型中: 获取它的 300 维语义向量
            vector = pretrained_model.get_vector(char)
            embedding_matrix[matrix_index, :] = T.from_numpy(vector.copy())
            found_count += 1
        else:
            # 字符不在预训练模型中 (生僻字): 保持为全零向量
            not_found_count += 1

    # 打印统计信息
    print(f"嵌入矩阵构建完成!")
    print(f"  - 词表大小: {vocab_size}")
    print(f"  - 向量维度: {embedding_dim}")
    print(f"  - 在预训练模型中找到的字符数: {found_count}")
    print(f"  - 未找到的字符数 (将使用全零向量): {not_found_count}")

    return embedding_matrix


# ============================================================================
# 4. Transformer 模型算法实现
# ============================================================================
"""
功能说明:
    实现基于 Transformer 的文本分类模型
    
架构概述:
    输入序列 -> 词嵌入层 -> 位置编码 -> Transformer Encoder -> 池化层 -> 全连接层 -> 分类输出

核心组件:
    - 自注意力机制 (Self-Attention): 让模型能够关注序列中不同位置的重要信息
    - 位置编码 (Positional Encoding): 为序列添加位置信息，因为 Transformer 本身不具备序列顺序感知
    - 编码器 (Encoder): 多层 Transformer Encoder Block，提取文本的高阶特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 4.1 位置编码层 (Positional Encoding)
# ============================================================================
class PositionalEncoding(nn.Module):
    """
    位置编码器
    
    作用:
        Transformer 的自注意力机制对所有位置一视同仁，无法感知字符的顺序。
        位置编码通过向嵌入向量中添加与位置相关的信号，让模型能够识别顺序。
    
    公式:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码器
        
        参数:
            d_model: 嵌入向量的维度 (需要和词嵌入维度一致)
            max_len: 支持的最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵 (max_len x d_model)
        pe = torch.zeros(max_len, d_model)
        
        # 生成位置索引 (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除以项: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度使用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度使用 cos
        
        # 增加批次维度 (1, max_len, d_model) 以便广播到任意 batch_size
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer (不会作为模型参数更新，但会随模型保存/加载)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 词嵌入张量, 形状为 (batch_size, seq_len, d_model)
        
        返回:
            添加了位置信息的词嵌入, 形状不变
        """
        # x: (batch_size, seq_len, d_model)
        # self.pe[:, :x.size(1), :]: 截取对应序列长度的位置编码
        x = x + self.pe[:, :x.size(1), :]
        return x


# ============================================================================
# 4.2 Transformer 文本分类模型
# ============================================================================
class TransformerTextClassifier(nn.Module):
    """
    基于 Transformer 的文本分类模型
    
    架构流程:
        1. 词嵌入层: 将字符 ID 转换为稠密向量
        2. 位置编码: 添加序列位置信息
        3. Dropout: 防止过拟合
        4. Transformer Encoder: 提取上下文特征
        5. 池化层: 将变长序列压缩为固定长度向量
        6. 全连接层: 映射到标签空间
        7. Softmax/Logits: 输出分类结果
    """
    
    def __init__(self, vocab_size, embedding_dim, num_classes, 
                 nhead=4, num_layers=2, dim_feedforward=256, 
                 dropout=0.1, max_len=5000, padding_idx=0,
                 pretrained_embeddings=None):
        """
        初始化模型
        
        参数:
            vocab_size:        词表大小 (嵌入矩阵的行数)
            embedding_dim:     词嵌入维度
            num_classes:       分类任务的类别数量
            nhead:             多头注意力的头数 (需能整除 embedding_dim)
            num_layers:        Transformer Encoder 的层数
            dim_feedforward:   前馈神经网络的隐藏层维度
            dropout:           Dropout 比率
            max_len:           最大序列长度
            padding_idx:       填充符的 ID (用于嵌入层忽略)
            pretrained_embeddings: 预训练词向量 (可选)
        """
        super(TransformerTextClassifier, self).__init__()
        
        # ----- 词嵌入层 -----
        if pretrained_embeddings is not None:
            # 使用预训练向量初始化
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=False,          # 允许微调
                padding_idx=padding_idx
            )
        else:
            # 随机初始化
            self.embedding = nn.Embedding(
                vocab_size, 
                embedding_dim, 
                padding_idx=padding_idx
            )
        
        # ----- 位置编码层 -----
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len)
        
        # ----- Dropout 层 -----
        self.dropout = nn.Dropout(dropout)
        
        # ----- Transformer Encoder 层 -----
        # PyTorch 内置的 TransformerEncoderLayer 已经包含了:
        #   - 多头自注意力机制 (Multi-Head Self-Attention)
        #   - 前馈神经网络 (Feed-Forward Network)
        #   - 残差连接 (Residual Connections)
        #   - 层归一化 (Layer Normalization)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,        # 输入维度
            nhead=nhead,                  # 注意力头数
            dim_feedforward=dim_feedforward,  # 前馈网络隐藏层维度
            dropout=dropout,
            batch_first=True              # 输入格式为 (batch, seq, feature)
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers         # 堆叠的 Encoder 层数
        )
        
        # ----- 池化层 + 分类层 -----
        # 使用全连接层将 Transformer 输出映射到类别空间
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward),  # 降维/升维
            nn.ReLU(),                                  # 激活函数
            nn.Dropout(dropout),                        # Dropout 防过拟合
            nn.Linear(dim_feedforward, num_classes)     # 输出 logits (未归一化的分数)
        )
        
        # ----- 初始化权重 -----
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化模型权重 (Xavier 均匀初始化)
        确保训练初期的梯度不会爆炸或消失
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_mask=None):
        """
        前向传播
        
        参数:
            src:      输入序列, 形状为 (batch_size, seq_len)
            src_mask: 注意力掩码 (可选，通常不需要)
        
        返回:
            logits: 分类 logits, 形状为 (batch_size, num_classes)
                    可通过 F.softmax(logits, dim=1) 转为概率，
                    或直接传入 nn.CrossEntropyLoss() 计算损失
        """
        # ----- 步骤 1: 词嵌入 -----
        # src: (batch_size, seq_len)
        # embedded: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(src)
        
        # ----- 步骤 2: 添加位置编码 -----
        # embedded: (batch_size, seq_len, embedding_dim)
        embedded = self.positional_encoding(embedded)
        
        # ----- 步骤 3: Dropout -----
        embedded = self.dropout(embedded)
        
        # ----- 步骤 4: Transformer Encoder -----
        # 自注意力机制会计算序列中每个字符对其他字符的注意力权重
        # 输出仍然保持形状: (batch_size, seq_len, embedding_dim)
        encoded = self.transformer_encoder(embedded, mask=src_mask)
        
        # ----- 步骤 5: 池化 (将序列压缩为固定长度向量) -----
        # 方案 1: 平均池化 (推荐) - 对所有位置的输出取平均
        # 这样能利用整个序列的信息，而不是只依赖第一个位置
        pooled = encoded.mean(dim=1)  # (batch_size, embedding_dim)
        
        # 方案 2: 取第一个位置 (类似 BERT 的 [CLS]，但需要特殊设计)
        # pooled = encoded[:, 0, :]  # 当前不推荐
        
        # ----- 步骤 6: 分类器 -----
        # pooled: (batch_size, embedding_dim)
        # logits: (batch_size, num_classes)
        logits = self.classifier(pooled)
        
        return logits


# ============================================================================
# 5. 模型训练模块
# ============================================================================
"""
功能说明:
    实现完整的训练循环，包括:
        - 损失计算 (Cross Entropy Loss)
        - 反向传播 (Backpropagation)
        - 优化器更新 (Adam / SGD)
        - 学习率调度 (Learning Rate Scheduler)
        - 训练日志记录
"""
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


class Trainer:
    """
    模型训练器
    
    职责:
        - 管理训练循环
        - 计算损失和梯度
        - 更新模型参数
        - 定期评估和保存模型
    """
    
    def __init__(self, model, train_loader, val_loader,
                 learning_rate=0.001, weight_decay=1e-4,
                 device=None, save_path="best_model.pth",
                 log_file=None):
        """
        初始化训练器

        参数:
            model:         待训练的 PyTorch 模型
            train_loader:  训练集 DataLoader
            val_loader:    验证集 DataLoader (可为 None)
            learning_rate: 初始学习率
            weight_decay:  权重衰减 (L2 正则化)
            device:        训练设备 (cuda / cpu)，默认自动检测
            save_path:     最佳模型保存路径（会自动添加时间戳）
            log_file:      日志文件路径 (可选，默认使用 logs/ 目录)
        """
        # 自动检测设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # 💡 新增：每次训练生成带时间戳的模型文件名，避免覆盖历史模型
        if save_path:
            dir_name, file_name = os.path.split(save_path)
            name, ext = os.path.splitext(file_name)
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_path = os.path.join(dir_name, f"{name}_{self.timestamp}{ext}")
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_path = save_path

        # 初始化日志器
        self.logger = get_logger(log_file)
        self.logger.info(f"📂 本次模型将保存至: {self.save_path}")
        self.logger.info(f"📄 日志已初始化，保存路径: {self.logger.log_file}")

        # 损失函数: CrossEntropyLoss 已经包含了 Softmax + NLLLoss
        # 适用于多分类任务，不需要在模型输出前手动加 Softmax
        self.criterion = nn.CrossEntropyLoss()

        # 优化器: AdamW (Adam with Weight Decay)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器: 当验证集损失不再改善时降低学习率
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',       # 监控指标越小越好 (损失)
            factor=0.5,       # 每次衰减为原来的 50%
            patience=2        # 等待 2 个 epoch 不改善后才衰减
        )

        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
    
    def train_epoch(self):
        """
        执行一个完整的训练 epoch
        
        返回:
            avg_loss: 平均训练损失
            avg_acc:  平均训练准确率
        """
        self.model.train()  # 切换到训练模式 (启用 Dropout)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 使用 tqdm 显示进度条
        progress_bar = tqdm(
            self.train_loader, 
            desc="训练", 
            leave=False
        )
        
        for batch_x, batch_y in progress_bar:
            # 数据移到设备
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # ----- 前向传播 -----
            # 计算模型的输出 (logits)
            logits = self.model(batch_x)
            
            # 计算损失
            loss = self.criterion(logits, batch_y)
            
            # ----- 反向传播 -----
            # 清空之前的梯度
            self.optimizer.zero_grad()
            
            # 计算当前损失的梯度
            loss.backward()
            
            # 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新模型参数
            self.optimizer.step()
            
            # ----- 统计信息 -----
            total_loss += loss.item() * batch_x.size(0)
            
            # 计算准确率: 取 logits 最大值的索引作为预测类别
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
            # 更新进度条显示
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct/total:.4f}"
            })
        
        # 计算平均值
        avg_loss = total_loss / total
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    @torch.no_grad()  # 禁用梯度计算 (节省内存和加速)
    def evaluate(self, loader):
        """
        在指定数据集上评估模型
        
        参数:
            loader: DataLoader (训练集或验证集)
        
        返回:
            avg_loss: 平均损失
            avg_acc:  平均准确率
        """
        self.model.eval()  # 切换到评估模式 (关闭 Dropout)
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # 前向传播
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            
            # 统计
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / total
        avg_acc = correct / total
        
        return avg_loss, avg_acc
    
    def train(self, num_epochs=10):
        """
        执行完整的训练流程 (多个 epoch)
        
        参数:
            num_epochs: 训练轮数
        
        返回:
            训练历史记录字典
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"开始训练 | 设备: {self.device} | 轮数: {num_epochs}")
        self.logger.info(f"{'='*60}\n")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # ----- 训练一个 epoch -----
            train_loss, train_acc = self.train_epoch()

            # ----- 验证 (如果有验证集) -----
            if self.val_loader is not None:
                val_loss, val_acc = self.evaluate(self.val_loader)

                # 记录历史
                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.train_accs.append(train_acc)
                self.val_accs.append(val_acc)

                # 学习率调度
                self.scheduler.step(val_loss)

                # 记录结果
                elapsed = time.time() - start_time
                log_msg = (f"Epoch [{epoch}/{num_epochs}] | "
                          f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
                          f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f} | "
                          f"耗时: {elapsed:.2f}s")
                self.logger.info(log_msg)

                # ----- 保存最佳模型 -----
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_checkpoint(epoch, val_acc)
                    self.logger.info(f"  ↳  ✓ 保存最佳模型 (准确率: {val_acc:.4f})")
            else:
                # 没有验证集
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)

                elapsed = time.time() - start_time
                log_msg = (f"Epoch [{epoch}/{num_epochs}] | "
                          f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
                          f"耗时: {elapsed:.2f}s")
                self.logger.info(log_msg)

                # 仅根据训练损失保存
                if train_acc > self.best_val_acc:
                    self.best_val_acc = train_acc
                    self.save_checkpoint(epoch, train_acc)

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"训练完成! 最佳准确率: {self.best_val_acc:.4f}")
        self.logger.info(f"{'='*60}\n")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
    
    def save_checkpoint(self, epoch, acc):
        """
        保存模型检查点
        
        参数:
            epoch: 当前轮数
            acc:   当前准确率
        """
        # 确保父目录存在
        parent_dir = os.path.dirname(self.save_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'accuracy': acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, self.save_path)
    
    def load_checkpoint(self, path=None):
        """
        加载模型检查点
        
        参数:
            path: 检查点路径 (默认使用保存路径)
        """
        path = path or self.save_path
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"已加载模型: Epoch {checkpoint['epoch']}, 准确率 {checkpoint['accuracy']:.4f}")


# ============================================================================
# 6. 模型评估模块
# ============================================================================
"""
功能说明:
    实现详细的评估指标计算，包括:
        - 准确率 (Accuracy)
        - 精确率 (Precision) - 宏平均/微平均
        - 召回率 (Recall)
        - F1 分数
        - 混淆矩阵 (Confusion Matrix)
"""
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    """
    模型评估器
    
    职责:
        - 计算多种评估指标
        - 生成分类报告
        - 绘制混淆矩阵
    """
    
    def __init__(self, model, dataloader, label_names=None, device=None):
        """
        初始化评估器
        
        参数:
            model:       已训练的模型
            dataloader:  测试集 DataLoader
            label_names: 标签名称列表 (用于显示报告)
            device:      设备 (cuda / cpu)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model = model.to(self.device)
        self.dataloader = dataloader
        self.label_names = label_names
    
    @torch.no_grad()
    def get_predictions(self):
        """
        在数据集上运行模型，获取所有预测结果
        
        返回:
            all_labels:    真实标签列表
            all_predictions: 预测标签列表
        """
        self.model.eval()
        
        all_labels = []
        all_predictions = []
        
        for batch_x, batch_y in self.dataloader:
            batch_x = batch_x.to(self.device)
            
            # 前向传播
            logits = self.model(batch_x)
            
            # 获取预测类别
            _, predicted = torch.max(logits, dim=1)
            
            all_labels.extend(batch_y.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
        
        return all_labels, all_predictions
    
    def print_classification_report(self):
        """
        打印详细的分类报告 (包括 Precision, Recall, F1)
        """
        y_true, y_pred = self.get_predictions()
        
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=self.label_names,
            zero_division=0
        )
        
        print("\n" + "="*60)
        print("分类报告")
        print("="*60)
        print(report)
        
        return report
    
    def plot_confusion_matrix(self, save_path="confusion_matrix.png"):
        """
        绘制并保存混淆矩阵
        
        参数:
            save_path: 图片保存路径
        """
        y_true, y_pred = self.get_predictions()
        
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        # 绘制
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.label_names if self.label_names else 'auto',
            yticklabels=self.label_names if self.label_names else 'auto'
        )
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"混淆矩阵已保存到: {save_path}")
        
        return cm


# ============================================================================
# 7. 模型推理模块
# ============================================================================
"""
功能说明:
    实现便捷的推理接口，允许用户输入新文本并获取预测结果
    
使用方式:
    predictor = Predictor(model, char_to_id, label_to_id)
    result = predictor.predict("这是一段新的文本")
"""

class Predictor:
    """
    模型推理器
    
    职责:
        - 预处理新文本 (编码为数字序列)
        - 运行模型预测
        - 返回人类可读的预测结果
    """
    
    def __init__(self, model, char_to_id, label_to_id, device=None):
        """
        初始化推理器
        
        参数:
            model:      已训练的模型
            char_to_id: 字符到 ID 的映射字典
            label_to_id: 标签到 ID 的映射字典
            device:     设备 (cuda / cpu)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.model = model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        self.char_to_id = char_to_id
        self.label_to_id = label_to_id
        
        # 创建 ID 到标签的反向映射
        self.id_to_label = {v: k for k, v in label_to_id.items()}
    
    def encode_text(self, text):
        """
        将输入文本编码为模型可接受的数字序列
        
        参数:
            text: 原始文本字符串
        
        返回:
            tensor: 形状为 (1, seq_len) 的张量
        """
        char_ids = []
        
        # 字符 -> ID
        for char in text:
            if char in self.char_to_id:
                char_ids.append(self.char_to_id[char] + START_NO)
            else:
                char_ids.append(UNK_NO)
        
        # 截断
        if len(char_ids) > SEQUENCE_LENGTH:
            char_ids = char_ids[:SEQUENCE_LENGTH]
        # 填充
        elif len(char_ids) < SEQUENCE_LENGTH:
            char_ids = char_ids + [PAD_NO] * (SEQUENCE_LENGTH - len(char_ids))
        
        # 转为张量 (添加批次维度)
        return torch.tensor([char_ids], dtype=torch.long).to(self.device)
    
    def predict(self, text, top_k=3):
        """
        对输入文本进行预测
        
        参数:
            text:    待预测的文本
            top_k:   返回概率最高的 k 个类别
        
        返回:
            results: 预测结果列表，格式为 [(标签, 概率), ...]
        """
        # 编码
        input_tensor = self.encode_text(text)
        
        # 模型推理
        with torch.no_grad():
            logits = self.model(input_tensor)
            
            # Softmax 转为概率
            probs = F.softmax(logits, dim=1)[0]  # 去掉批次维度
        
        # 获取 top-k 预测
        top_probs, top_indices = torch.topk(probs, top_k)
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            label = self.id_to_label[idx.item()]
            results.append((label, prob.item()))
        
        return results
    
    def predict_batch(self, texts, top_k=3):
        """
        批量预测
        
        参数:
            texts:   文本列表
            top_k:   返回概率最高的 k 个类别
        
        返回:
            results: 预测结果列表，每个元素为 [(标签, 概率), ...]
        """
        # 批量编码
        input_tensors = [self.encode_text(text) for text in texts]
        batch_tensor = torch.cat(input_tensors, dim=0)
        
        # 批量推理
        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs = F.softmax(logits, dim=1)
        
        # 解析结果
        all_results = []
        for i in range(len(texts)):
            top_probs, top_indices = torch.topk(probs[i], top_k)
            
            results = []
            for prob, idx in zip(top_probs, top_indices):
                label = self.id_to_label[idx.item()]
                results.append((label, prob.item()))
            
            all_results.append(results)
        
        return all_results


# ============================================================================
# 主程序入口 (完整演示)
# ============================================================================
if __name__ == "__main__":

    # 初始化日志系统
    logger = get_logger()
    
    logger.info("="*70)
    logger.info("Transformer 文本分类完整流程")
    logger.info("="*70)

    # ========================================================================
    # 步骤 1: 加载数据配置
    # ========================================================================
    logger.info("\n[1/6] 加载词汇映射...")
    label_to_id, char_to_id = load_vocab_mappings()
    num_classes = len(label_to_id)
    vocab_size = len(char_to_id) + 4  # +4 预留空间

    logger.info(f"  - 类别数量: {num_classes}")
    logger.info(f"  - 词表大小: {vocab_size}")

    # ========================================================================
    # 步骤 2: 加载预训练词向量
    # ========================================================================
    logger.info("\n[2/6] 加载预训练词向量...")
    embedding_matrix = load_pretrained_embeddings(char_to_id)
    logger.info(f"  - 嵌入矩阵形状: {embedding_matrix.shape}")

    # ========================================================================
    # 步骤 3: 准备数据集
    # ========================================================================
    logger.info("\n[3/6] 准备数据集...")
    full_dataset = CSVDataset(TRAIN_PATH)

    # 划分训练集和验证集 (70% 训练, 30% 验证)
    train_dataset, val_dataset = full_dataset.get_splits(n_valid=0.3)

    # 创建 DataLoader
    # 💡 改进: 减小 batch_size 可以增加训练步数，让模型学得更好
    BATCH_SIZE = 16  # 从 32 改为 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logger.info(f"  - 训练集样本数: {len(train_dataset)}")
    logger.info(f"  - 验证集样本数: {len(val_dataset)}")
    logger.info(f"  - Batch Size: {BATCH_SIZE}")

    # ========================================================================
    # 步骤 4: 初始化模型
    # ========================================================================
    logger.info("\n[4/6] 初始化 Transformer 模型...")

    # 💡 优化后的参数配置:
    # 1. nhead=5: 300/5=60，每头维度更充足（字符级任务不需要太多头）
    # 2. num_layers=2: 2层足够，3层容易过拟合
    # 3. dim_feedforward=256: 与300维嵌入匹配，避免参数量过大
    # 4. dropout=0.1: 适中的dropout，防止过拟合同时保留学习能力

    model = TransformerTextClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        num_classes=num_classes,
        nhead=5,                # 💡 300/5=60，每头维度更充足
        num_layers=2,           # 💡 2层足够，3层容易过拟合
        dim_feedforward=256,    # 💡 与300维嵌入匹配
        dropout=0.1,            # 💡 适中dropout
        pretrained_embeddings=embedding_matrix
    )

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  - 总参数数量: {total_params:,}")
    logger.info(f"  - 可训练参数: {trainable_params:,}")

    # ========================================================================
    # 步骤 5: 训练模型
    # ========================================================================
    logger.info("\n[5/6] 开始训练...")

    # 💡 优化训练参数:
    # 1. learning_rate=0.0003: 更温和的学习率，保护预训练词向量
    # 2. weight_decay=1e-4: 回到默认值，避免过度正则化
    # 3. epochs=50: 给模型足够时间学习（文本分类通常需要 30-50 轮）

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=0.0003,   # 💡 更温和的学习率
        weight_decay=1e-4,      # 💡 回到默认值
        save_path=os.path.join(MODELS_PATH, "best_transformer_model.pth")
    )

    # 执行训练 30 轮
    history = trainer.train(num_epochs=30)

    # ========================================================================
    # 步骤 6: 评估和推理
    # ========================================================================
    logger.info("\n[6/6] 模型评估与推理...")

    # 加载最佳模型
    trainer.load_checkpoint()

    # 评估 (使用验证集)
    evaluator = Evaluator(
        model=model,
        dataloader=val_loader,
        label_names=list(label_to_id.keys())
    )

    # 打印分类报告
    evaluator.print_classification_report()

    # 测试推理
    logger.info("\n" + "="*70)
    logger.info("推理测试")
    logger.info("="*70)

    predictor = Predictor(
        model=model,
        char_to_id=char_to_id,
        label_to_id=label_to_id
    )

    # 测试几条样本
    test_texts = [
        "中国铁腰与英超球队埃弗顿分道扬镳，闪电般转投谢联（本赛季成功升入英超）",
        "各国拥有的核弹头数量俄罗斯媒体认为，各国拥有的核弹头数量：中国以450枚排行第四，仅以50枚的差距落后于法国。",
        "参加成人高考时，７３岁的徐思华被拦在了大门外，工作人员问：“老同志，你到这儿来干什么？”当听到他也是考生时，",
        "我是一个兵，来自老百姓"
    ]
    
    for text in test_texts:
        results = predictor.predict(text, top_k=3)
        logger.info(f"\n文本: {text}")
        logger.info("预测结果:")
        for label, prob in results:
            logger.info(f"  - {label}: {prob:.4f}")

    logger.info("\n" + "="*70)
    logger.info("全部完成!")
    logger.info("="*70)