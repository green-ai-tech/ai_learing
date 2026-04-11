"""
1. 数据集预处理
    ----把train.csv（训练集）,test.csv（测试集） 处理后保存为二进制文件：
        ----样本：chars.pk(python序列化，即内存的数据保存到文件)
        ----标签：labels.pk
"""

import os                               #负责文件路径的处理
import pickle                           #序列化（把内存数据 存为文件）
import pandas as pd                     #读取csv 文件                  
from random import shuffle
from operator import itemgetter
from collections import Counter,defaultdict


"""
准备一个数据读写的对象：
    -保存数据
    -读取数据
"""
#封装：数据的读与写
class PickleFileOprator:
    def __init__(self,data=None,file_path=""):
        self.data = data
        self.file_path = file_path

    def save(self):
        #文件保存
        with open(self.file_path,"wb") as fd:       #上下文管理，文件自动关闭
            pickle.dump(self.data,fd)

    def read(self):
        #文件读取
        with open(self.file_path,"rb") as fd:
            content = pickle.load(fd)
            self.data=content
        return content

#op = PickleFileOprator(None,"/notebooks/03_encoder_decoder/ds/dataset/chars")


# 获取当前脚本所在目录（项目根目录）
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_PATH    = os.path.join(PROJECT_ROOT, "data")
TRAIN_PATH   = os.path.join(DATA_PATH, "train.csv")
TEST_PATH    = os.path.join(DATA_PATH, "test.csv")
VOCAB_PATH   = os.path.join(PROJECT_ROOT, "vocab")
NUM_WORDS    = 6000                # 处理词的多少，可根据情况而定

# 自动创建目录
os.makedirs(VOCAB_PATH, exist_ok=True)


"""
数据处理器
"""
class FileProcessing:
    def __init__(self,n) -> None:       #n,保留词频最高的单词数目
        self.__n = n                    #__n私有变量，外部不能直接访问

    def _read_train_file(self):

        #使用pandas 读取csv 文件
        train_pd = pd.read_csv(TRAIN_PATH)                  #读取为表格：DataFrame

        #标签列表
        label_list = train_pd["label"].unique().tolist()    #获取标签列中所有不重复的值，并转为列表

        #样本:content(统计词频)
        character_dict = defaultdict()                      #存放词频的字典

        for content in train_pd["content"]:                 #取出，训练统计每行的词频
            for k,v in Counter(content).items():
                character_dict[k] = v                       #字典中 key即是"词"，value是词频

        #把上面统计的数据集打乱
        sort_char_list = [(k,v) for k,v in character_dict.items()]
        shuffle(sort_char_list)                             #列表（堆：输出变量）与元组
        print("统计字符数：",len(character_dict))
        print("打乱后的 前十词：",sort_char_list[:10])


        #保留前n个
        top_n_chars = [_[0] for _ in sort_char_list[:self.__n]]

        return label_list,top_n_chars
    

    def run(self):

        label_list,top_n_chars = self._read_train_file()

        PickleFileOprator(data=label_list,file_path=os.path.join(VOCAB_PATH,"labels.pk")).save()
        PickleFileOprator(data=top_n_chars,file_path=os.path.join(VOCAB_PATH,"chars.pk")).save()
       

# processor = FileProcessing(NUM_WORDS)
# processor.run()

# labels = PickleFileOprator(file_path="/Users/logicye/Code/ai_learning/notebooks/03_encoder_decoder/labels.pk").read()
# print(labels)

# content = PickleFileOprator(file_path="/Users/logicye/Code/ai_learning/notebooks/03_encoder_decoder/chars.pk").read()[:20]
# print(content)



# 2. 数据集的数据处理
"""
- 把词袋编号，根据编号，把句子转换为一个数值向量（使用Dataset与Dataloader 来管理）
    - 加载标签，加载标签与词袋
    - 向量化（词向量）
"""

def load_file_file():
    labels = PickleFileOprator(file_path=os.path.join(VOCAB_PATH,"labels.pk")).read()
    chars  = PickleFileOprator(file_path=os.path.join(VOCAB_PATH,"chars.pk")).read()

    #编号
    label_dict = dict(zip(labels,range(len(labels))))
    #print(label_dict)
    char_dict = dict(zip(chars,range(len(chars))))

    return label_dict,char_dict


#读取数据集
def load_csv_file(file_path):
    df = pd.read_csv(file_path)
    samples,y_true = [],[]
    for index ,row in df.iterrows():
        samples.append(row["content"])
        y_true.append(row["label"])
    return samples,y_true

#print(load_csv_file(TRAIN_PATH)[0][0:2])


#预留机制

PAD ='<PAD>'
UNK = '<UNK>'

PAD_NO = 0
UNK_NO = 1
START_NO = UNK_NO +1        #预留编号
SENT_LENGTH=200             #保证长度一致

def text_feature(labels, contents, label_dict, char_dict):
    all_samples, y_true = [], []
    for label, content in zip(labels, contents):
        # 标签映射
        y_true.append(label_dict[label])
        
        # 当前文本的字符索引序列
        sample = []
        for char in content:
            if char in char_dict:
                sample.append(char_dict[char] + START_NO)
            else:
                sample.append(UNK_NO)
        
        # 截断与补齐
        if len(sample) > SENT_LENGTH:
            sample = sample[:SENT_LENGTH]
        else:
            sample = sample + [PAD_NO] * (SENT_LENGTH - len(sample))
        
        all_samples.append(sample)

    return all_samples, y_true


#使用torch.utils.data.Dataset, DataLoader
import numpy as np
import torch as T
from torch.utils.data import Dataset,random_split

class CSVDataset(Dataset):
    def __init__(self,file_path) :
        label_dict, char_dict = load_file_file()
        contents, labels = load_csv_file(file_path)

        x,y = text_feature(labels,contents,label_dict,char_dict)

        #转化为张量
        self.x = T.from_numpy(np.array(x)).long()
        self.y = T.from_numpy(np.array(y))

    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, index) :
        return [self.x[index],self.y[index]]
    
    def get_splits(self,n_vaild=0.3):
        #计算验证集的数目
        valid_size = round(n_vaild*len(self.x)) #验证集的数量
        train_size = len(self.x)- valid_size
        return random_split(self,[train_size,valid_size])
    

ds = CSVDataset(TRAIN_PATH)
print(ds[0])



# 3. 词嵌入的处理
"""- 词嵌入 ...

"""
from gensim.models import KeyedVectors
#读取词袋
label_dict, char_dict = load_file_file()
em_model = KeyedVectors.load_word2vec_format(
    os.path.join(DATA_PATH, "sgns.wiki.char.bz2"),
    binary=False,
    encoding="utf-8",
    unicode_errors="ignore"
)

pretraind_vector = T.zeros(NUM_WORDS+4,300).float()

for char ,index in char_dict.items():
    if char in em_model.key_to_index:
        vector = em_model.get_vector(char)
        pretraind_vector[index,:] = T.from_numpy(vector.copy())



# print(pretraind_vector[-1])
# print(pretraind_vector[0])








# 4. transformer模型算法实现
import torch
import torch.nn as nn
import math

EMBEDDING_SIZE = 300  # 每个词的特征向量的长度


# 位置编码类（03_new.py 中缺失的）
class PositionalEncoding(nn.Module):
    """位置编码器：为序列添加位置信息"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerForClassification(nn.Module):
    def __init__(
        self, 
        num_classes,            # 类别数量改为参数
        nhead=4,                # 注意力头:理论上越多效果越好
        num_layers=2,           # 特征抽取层
        dim_feedforward=2048,   # 网络维度
        dropout=0.1,            
        activation="relu"
    ):
        super(TransformerForClassification, self).__init__()
        self.d_model = EMBEDDING_SIZE
        vocab_size = NUM_WORDS + 4  # 与 pretraind_vector 保持一致 (+4: PAD, UNK, START, 额外预留)
        
        # nhead与d_model是整除关系
        assert self.d_model % nhead == 0, "nhead必须整除d_model"
        
        # 定义层
        # 1. 词嵌入（修复变量名：pretrained_vector -> pretraind_vector）
        self.emb = nn.Embedding.from_pretrained(
            pretraind_vector,     # ✅ 修复拼写错误
            freeze=False, 
            padding_idx=0
        )
        
        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_model,
            dropout=dropout
        )
        
        # 3. 编码器(核心是编码单元)
        # 编码单元
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            batch_first=True,  # 第一个维度就是批次数
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers
        )    
        
        # 4. 分类器（修复硬编码 + 添加池化）
        self.pooling = nn.AdaptiveAvgPool1d(1)  # 平均池化：(batch, d_model, seq_len) -> (batch, d_model, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.d_model, num_classes)  # num_classes 改为参数传入
        )

    def forward(self, x):
        # x: (batch, seq_len)
        
        # 词嵌入
        x = self.emb(x) * math.sqrt(self.d_model)   # (batch, seq_len, d_model) - 防止梯度消失
        
        # 位置编码
        x = self.pos_encoder(x)                     # (batch, seq_len, d_model)
        
        # 编码器
        x = self.transformer_encoder(x)             # (batch, seq_len, d_model)
        
        # 池化：将变长序列压缩为固定长度向量
        x = x.transpose(1, 2)                       # (batch, d_model, seq_len)
        x = self.pooling(x).squeeze(-1)             # (batch, d_model)
        
        # 分类器
        x = self.classifier(x)                      # (batch, num_classes)
        return x



# 5. 模型的训练

# 超参
BATCH_SIZE = 32
lr = 0.001
EPOCHES = 30
# 数据集
ds_train = CSVDataset(TRAIN_DIR)
ds_test  = CSVDataset(TEST_DIR)

ld_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
ld_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=True)
# 模型
model = model = TransformerForClassfication(
    nhead=10,
    dim_feedforward=128,
    num_layers=1,
    dropout=0.0, 
).to("cuda")
# 优化器
opt = optim.Adam(model.parameters(),lr=lr)
# 相关函数
loss_f = nn.CrossEntropyLoss()

@torch.no_grad
def eval_test():
    # 循环测试集
    model.eval()
    total_loss = 0.0
    num_batch = 0
    total_correct = 0
    for x, y in ld_test:
        x = x.to("cuda")
        y = y.to("cuda")
        # 预测
        y_ = model(x)
        # 计算损失
        loss = loss_f(y_, y)
        total_loss += loss.detach().cpu().item()
        num_batch += x.shape[0]
        # 统计准确数
        total_correct += (y_.argmax(dim=1) == y).sum().detach().cpu().item()
    avg_loss = total_loss / num_batch
    avg_correct = total_correct / num_batch
    return avg_loss, avg_correct

for epoch in range(EPOCHES):
    print(F"第{epoch + 1}轮")
    total_loss = 0.0
    num_batch = 0
    model.train()   # dropout,batchm
    for idx, (x, y) in enumerate(ld_train):
        x = x.to("cuda")
        y = y.to("cuda")
        # 梯度清零
        opt.zero_grad()
        #预测
        y_ = model(x)
        # 计算误差
        loss = loss_f(y_, y)
        # 自动求导
        loss.backward()
        #梯度更像
        opt.step()
        total_loss += loss.detach().cpu().item()
        num_batch += x.shape[0]
    # 评估
    avg_loss = total_loss / num_batch
    lss, accu = eval_test()
    print(f"\t|-训练损失：{avg_loss:.4f}，验证损失：{lss:.4f}，准确率:{accu*100:.2f}%")
    # 保存模型
    torch.save(model.state_dict(),"trans_cls.pth")
    # with torch.no_grad():


# 6. 模型的评估


# 7. 模型的推理


# ============================================================================
# 主程序入口（测试模型）
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Transformer 文本分类模型测试")
    print("=" * 70)

    # ---------------- 1. 检查词汇映射是否存在 ----------------
    print("\n[1/5] 检查词表文件...")
    if not os.path.exists(os.path.join(VOCAB_PATH, "labels.pk")):
        print("⚠️  词表文件不存在，正在生成...")
        processor = FileProcessing(NUM_WORDS)
        processor.run()
        print("✅ 词表生成完成!")
    else:
        print("✅ 词表文件已存在")

    # ---------------- 2. 加载映射 ----------------
    print("\n[2/5] 加载词汇映射...")
    label_dict, char_dict = load_file_file()
    num_classes = len(label_dict)
    print(f"  - 类别数量: {num_classes}")
    print(f"  - 字符词表大小: {len(char_dict)}")

    # ---------------- 3. 测试数据集加载 ----------------
    print("\n[3/5] 测试数据集加载...")
    ds = CSVDataset(TRAIN_PATH)
    print(f"  - 数据集大小: {len(ds)}")
    print(f"  - 第一条数据形状: x={ds[0][0].shape}, y={ds[0][1]}")

    # ---------------- 4. 初始化模型 ----------------
    print("\n[4/5] 初始化 Transformer 模型...")
    model = TransformerForClassification(
        num_classes=num_classes,
        nhead=4,
        num_layers=2,
        dim_feedforward=2048,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - 总参数数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")

    # ---------------- 5. 前向传播测试 ----------------
    print("\n[5/5] 前向传播测试...")
    model.eval()
    
    # 取一条数据测试
    x_sample, y_sample = ds[0]
    x_batch = x_sample.unsqueeze(0)  # 添加 batch 维度: (1, seq_len)
    
    with torch.no_grad():
        output = model(x_batch)  # (1, num_classes)
        predicted_class = torch.argmax(output, dim=1).item()
        
    # 反向映射获取标签名
    id_to_label = {v: k for k, v in label_dict.items()}
    predicted_label = id_to_label.get(predicted_class, "未知")
    true_label = id_to_label.get(y_sample.item(), "未知")
    
    print(f"  - 输入形状: {x_batch.shape}")
    print(f"  - 输出形状: {output.shape}")
    print(f"  - 真实标签: {true_label}")
    print(f"  - 预测标签: {predicted_label}")
    print(f"  - 预测概率分布: {torch.softmax(output, dim=1).squeeze().numpy()}")

    print("\n" + "=" * 70)
    print("✅ 模型测试完成！")
    print("=" * 70)
