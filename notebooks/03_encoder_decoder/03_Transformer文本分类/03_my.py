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
# 获取当前脚本所在目录的父目录（项目根目录）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORK_PATH    = PROJECT_ROOT
DATASET_PATH = os.path.join(PROJECT_ROOT, "ds/dataset")
TRAIN_PATH   = os.path.join(DATASET_PATH, "train.csv")
TEST_PATH    = os.path.join(DATASET_PATH, "test.csv")
NUM_WORDS    = 6000  # 保留最高频的字符数量


# ============================================================================
# 1.3 文件处理类
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
            - WORK_PATH/labels.pk:  标签列表
            - WORK_PATH/chars.pk:   高频字符列表
        """
        label_list, top_n_chars = self._read_train_file()
        
        # 保存标签列表
        PickleFileOprator(
            data=label_list,
            file_path=os.path.join(WORK_PATH, "labels.pk")
        ).save()
        
        # 保存高频字符列表
        PickleFileOprator(
            data=top_n_chars,
            file_path=os.path.join(WORK_PATH, "chars.pk")
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
    
    返回:
        label_to_id: 标签映射字典,例如 {'体育': 0, '娱乐': 1, ...}
        char_to_id:  字符映射字典,例如 {'的': 0, '了': 1, '是': 2, ...}
    """
    # 从文件加载标签列表
    labels = PickleFileOprator(
        file_path=os.path.join(WORK_PATH, "labels.pk")
    ).read()
    
    # 从文件加载字符列表
    chars = PickleFileOprator(
        file_path=os.path.join(WORK_PATH, "chars.pk")
    ).read()

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

# Word2Vec 预训练模型文件路径 (包含字符的语义向量)
WORD2VEC_PATH = os.path.join(DATASET_PATH, "sgns.wiki.char.bz2")


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
# 4. Transformer 模型算法实现 (待开发)
# ============================================================================



# ============================================================================
# 5. 模型训练模块 (待开发)
# ============================================================================
# TODO: 实现训练循环 (损失计算、反向传播、优化器更新)


# ============================================================================
# 6. 模型评估模块 (待开发)
# ============================================================================
# TODO: 实现评估指标 (准确率、Precision、Recall、F1 等)


# ============================================================================
# 7. 模型推理模块 (待开发)
# ============================================================================
# TODO: 实现推理接口 (输入新文本，输出预测标签)


# ============================================================================
# 主程序入口 (测试区域)
# ============================================================================
if __name__ == "__main__":

    """
    # 测试 1: 文件处理模块
    processor = FileProcess(100)
    label_list, top_chars = processor._read_train_file()
    print("\n=== 测试结果 ===")
    print(f"标签类别数量: {len(label_list)}")
    print(f"标签列表: {label_list[:6]}...")
    print(f"\n选中的字符数量: {len(top_chars)}")
    print(f"前 20 个字符: {top_chars[:20]}")
    """

    """
    # 测试 2: 词汇映射加载
    label_to_id, char_to_id = load_vocab_mappings()
    print("测试！！")
    print(label_to_id)
    print(char_to_id)
    """

    """
    # 测试 3: 预训练词向量加载
    label_to_id, char_to_id = load_vocab_mappings()
    embedding_matrix = load_pretrained_embeddings(char_to_id)
    print(f"嵌入矩阵形状: {embedding_matrix.shape}")
    print(f"'猫' 的向量前 10 维: {embedding_matrix[char_to_id['猫'] + START_NO][:10]}")
    """

    # 测试 4: 数据集加载
    ds = CSVDataset(TRAIN_PATH)
    print(f"数据集大小: {len(ds)}")
    print(f"第一条数据 - 特征形状: {ds[0][0].shape}, 标签: {ds[0][1]}")




            