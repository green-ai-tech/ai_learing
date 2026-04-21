import os
import math
import torch
import torch.nn as nn
from tokenizers import Tokenizer                        #分词
from torchtext.vocab import build_vocab_from_iterator   #构建词袋
from torch.utils.data import dataset,dataloader         #数据集与批次数据集
from torch.nn.functional import pad,log_softmax         #pad补边与对齐,log_softmax(转换为概率)
from torch.utils.data import Dataset

#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")   #标准的torch.device对象


#==========================1. 语料库（即数据集）处理==============================（tokenizer）
## 1.1  英文的分词处理()
#加载分词器
tokenizer= Tokenizer.from_file("./tokenizer.json")
def en_tokenizer(line):
    """
    """
    tokens = tokenizer.encode(line,add_special_tokens=False).tokens
    return tokens
# print(en_tokenizer("hello,this a dog"))

# #再封装（生成器版本）:使用协程，循环中使用的时候再加载（节省空间）
# en_filepath = "./datasets/train.en"

# def yield_en_tokens():
#     with open(en_filepath,encoding="utf-8") as fd:     #file descreption,打开文件
#         for line in fd:
#             yield en_tokenizer(line)  #节省空间！！！

# #构建英文分词器
# en_tokens=yield_en_tokens()

# #构建词袋（词袋的保存）
# en_vocab_file = "vocab_en.pt"           #词袋文件
# en_vocab = build_vocab_from_iterator(en_tokens)

# #默认的索引
# #en_vocab.set_default_index(en_vocab["<unk>"])   #0编号<unk> = (代表所有僻词)

# torch.save(en_vocab,en_vocab_file)    #需要消耗时间，下一次

# en_vocab = torch.load("vocab_en.pt",weights_only=False)
# print(en_vocab["apple"],len(en_vocab))





## 1.2  中文的分词处理（依赖第三方框架）
zh_filepath = "./datasets/train.zh"
def zh_tokenizer(line):
    return list(line.strip().replace("",""))    #strip去掉前后两边的空格，使用replace把中间的空格替换掉

# def yield_zh_tokens():
#     with open(zh_filepath,encoding="utf-8") as fd:
#         for line in fd:
#             yield zh_tokenizer(line)

# zh_token = yield_zh_tokens()

# zh_vocab_file = "vocab_zh.pt"
# zh_vocab = build_vocab_from_iterator(zh_token)
# torch.save(zh_vocab,zh_vocab_file)

# zh_vocab= torch.load("vocab_zh.pt",weights_only=False)
# print(zh_vocab.stoi["我"])
# print(zh_vocab["我"])
# print(zh_vocab.itos[5])

#数据集
zh_vocab = torch.load("vocab_zh.pt",weights_only=False)         #加载对象，存储的对象必须支持序列与反序列
en_vocab = torch.load("vocab_en.pt",weights_only=False)

zh_filepath = "./datasets/train.zh"
en_filepath = "./datasets/train.en"

class TranslateDataset(Dataset):
    def __init__(self):
        #初始化
        self.zh_tokens=self._load_tokens(zh_filepath,zh_tokenizer,zh_vocab)
        self.en_tokens=self._load_tokens(en_filepath,en_tokenizer,en_vocab)

        self.len_tokens = len(self.zh_tokens)

    def __getitem__(self, idx):
        #根据索引返回数据
        return self.en_tokens[idx],self.zh_tokens[idx]

    def __len__(self):
        # 返回数据集长度
        return self.len_tokens

    # def _load_tokens(self,filepath,tokenizer,vocab):
    #     tokens_list = []    #存放 向量化的词
    #     with open(filepath,encoding="utf-8") as fd:
    #         for line in fd:
    #             tokens = tokenizer(line)    #分词
    #             #把token 转为编号（向量化）
    #             num_tokens = [vocab[token] for token in tokens]
    #             #存储到列表
    #             tokens_list.append(num_tokens)
    #     return tokens_list

    def _load_tokens(self, filepath, tokenizer, vocab):
        tokens_list = []
        with open(filepath, encoding="utf-8") as fd:
            for line in fd:
                line = line.strip()
                if not line:
                    continue
                    
                tokens = tokenizer(line)
                num_tokens = [vocab.stoi[token] if token in vocab.stoi else vocab.default_index for token in tokens]
                tokens_list.append(num_tokens)
                
        return tokens_list
                



#测试

ds = TranslateDataset()
print(ds[0])

#==========================2. 构建Transformer模型==============================（基本被原生作者实现）






#==========================3. 模型训练========================================（Pytorch,Trainer）




##模型保存




#==========================4. 模型推理========================================


##加载模型（单独编辑为一个模块)