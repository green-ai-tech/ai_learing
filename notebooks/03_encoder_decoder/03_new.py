"""
1. 数据集预处理
    ----把train.csv（训练集）,test.csv（测试集） 处理后保存为二进制文件：
        ----样本：chars.pk(python序列化，即内存的数据保存到文件)
        ----标签：lable.pk
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


DATASET_PATH = "/Users/logicye/Code/ai_learning/notebooks/03_encoder_decoder/ds/dataset"
TRAIN_PATH   = os.path.join(DATASET_PATH,"train.csv")
TEST_PATH    = os.path.join(DATASET_PATH,"test.csv")
NUM_WORDS    = 6000                # 处理词的多少，可根据情况而定


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
        lable_list = train_pd["label"].unique().tolist()    #获取标签列中所有不重复的值，并转为列表

        #样本:content(统计词频)
        character_dict = defaultdict()                      #存放词频的字典

        for content in train_pd["content"]:                 #取出，训练统计每行的词频
            for k,v in Counter(content).items():            
                character_dict[k] = v                       #字典中 key即是“词”，value是词频

        #把上面统计的数据集打乱
        sort_char_list = [(k,v) for k,v in character_dict.items()]
        shuffle(sort_char_list)                             #列表（堆：输出变量）与元组
        print("统计字符数：",len(character_dict))
        print("打乱后的 前十词：",sort_char_list[:10])


        #保留前n个      
        top_n_chars = [_[0] for _ in sort_char_list[:self.__n]]

        return lable_list,top_n_chars
    

    def run(self):              
  
        lable_list,top_n_chars = self._read_train_file()

        PickleFileOprator(data=lable_list,file_path="/Users/logicye/Code/ai_learning/notebooks/03_encoder_decoder/lables.pk").save()
        PickleFileOprator(data=top_n_chars,file_path="/Users/logicye/Code/ai_learning/notebooks/03_encoder_decoder/chars.pk").save()
       

# processor = FileProcessing(NUM_WORDS)
# processor.run()

# labels = PickleFileOprator(file_path="/Users/logicye/Code/ai_learning/notebooks/03_encoder_decoder/lables.pk").read()
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
    labels = PickleFileOprator(file_path="/Users/logicye/Code/ai_learning/notebooks/03_encoder_decoder/lables.pk").read()
    chars  = PickleFileOprator(file_path="/Users/logicye/Code/ai_learning/notebooks/03_encoder_decoder/chars.pk").read()

    #编号
    lable_dict = dict(zip(labels,range(len(labels))))
    #print(lable_dict)
    char_dict = dict(zip(chars,range(len(chars))))

    return lable_dict,char_dict


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

"""
# def text_feature(labels,contents,label_dict,char_dict):
#     all_samples,y_true = [],[]  # 使用更清晰的变量名
#     for lable,content in zip(labels,contents):
#         #标签替换
#         y_true.append(label_dict[lable])
#         sample = []  # 当前样本的处理结果

#         # 将字符转换为对应的编号
#         for char in content:
#             if char in char_dict:
#                 sample.append(char_dict[char]+START_NO)
#             else:
#                 sample.append(UNK_NO)

#         #截断与补齐
#         if len(sample) < SENT_LENGTH:
#             # 补齐到固定长度
#             sample.extend([PAD_NO] * (SENT_LENGTH - len(sample)))
#         else:
#             sample = sample[:SENT_LENGTH]
        
#         all_samples.append(sample)  # 添加处理后的样本
    
#     return all_samples,y_true



# label_dict,char_dict = load_file_file()
# contents,labels = load_csv_file(TRAIN_PATH)

# samples,label_ture = text_feature(labels,contents,label_dict,char_dict)

# print(samples[0])
# print(label_ture[0])

"""


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
    "/Users/logicye/Code/ai_learning/notebooks/03_encoder_decoder/ds/dataset/sgns.wiki.char.bz2",
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


#===========================================
import torch.nn as nn
from torch.utils.data import DataLoader

embedding = nn.Embedding.from_pretrained(pretraind_vector,freeze=False,padding_idx=0)

ds = CSVDataset(TRAIN_PATH)

train_loader = DataLoader(ds,batch_size=5,shuffle=True)
for x_b,y_b in train_loader:
    v_emb = embedding(x_b)







# 4. transformer模型算法实现


# 5. 模型的训练 


# 6. 模型的评估


# 7. 模型的推理
