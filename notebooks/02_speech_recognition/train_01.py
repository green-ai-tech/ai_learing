from transformers import ASTFeatureExtractor, ASTConfig, ASTForAudioClassification
from transformers import TrainingArguments, Trainer

from torchaudio.datasets import SPEECHCOMMANDS
import os
import torch
import numpy as np
# -----------------------------1. 模型------------------------------------
# 准备读取数据集的信息（分类的类别数量，id2label，label2id）
ds_dir = "/Users/logicye/Code/my_Datasets/方向数据集"     
labels = []  # 存放35个标签
for item in os.listdir(ds_dir):
    # 遍历每个目录的名字
    if os.path.isdir(os.path.join(ds_dir, item)) and not item.startswith("_"):  # 过滤：文件不要，_开头的目录不要
        labels.append(item)
label2id = {d:idx for idx, d in enumerate(labels)}
id2label = {idx:d for idx, d in enumerate(labels)}

# print(labels)
# print(label2id)
# print(id2label)
# 1.1. 构建配置
config = ASTConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,             # 决定了模型的大小与参数的个数   
    hidden_act="gelu",                  # 正是因为激活函数的非线性，整个网络最后才是非线性（分类能力高）。
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,   # 特征的激活方式
    initializer_range=0.02,             # 权重的初始值的大小。（数据标准化的原理一样）
    layer_norm_eps=1e-12,               # 标准化的时候，除数容易为0，防止为0，一般会添加epsilon  （x-mean）/std
    patch_size=16,          
    qkv_bias=True,
    frequency_stride=10,                # 音频特征处理
    time_stride=10,
    max_length=128,                     # 必须与特征处理器保持一致
    num_mel_bins=128,
    num_labels=len(labels),             # 与数据集有关- 分类的类别数量
    id2label=id2label,         
    label2id=label2id,
)

model = ASTForAudioClassification(config)  # 配置是构建模型的必备条件

# model.save_pretrained("./models")

# 1.2. 构建模型（重新构建）

# -----------------------------2. 数据集（5字段 -> 2字段，使用处理器做特征抽取）------------------------------------
# 1.1. 准备处理器：语音特征预处理
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

# 1.2. 实现数据集
class ASTDataset(SPEECHCOMMANDS):
    # 构造器
    def __init__(self, extractor, ds_path="/Users/logicye/Code/Datasets/", subset="training", download=True):
        # 利用父类的功能
        super(ASTDataset, self).__init__(root=ds_path, download=download, subset=subset)
        # 保存处理器（抽取特征）
        self.extractor = extractor
        
    # __getitem__()
    def __getitem__(self, idx):
        # 调用父类的__getitem__获取数据，取出样本，取出标签
        data, sample_rate, label, speaker_id, wav_id = super(ASTDataset, self).__getitem__(idx)
        # 抽取样本特征（张量）
        feature = self.extractor(data[0], sample_rate, return_tensors="pt")
        # 标签从字符串转换为id，然后转换为张量 
        return feature["input_values"][0], torch.tensor([label2id[label]])   # feature[1, 128, 128], label[1]

#########测试
ds_train = ASTDataset(extractor=feature_extractor, subset="training")
ds_valid = ASTDataset(extractor=feature_extractor, subset="validation")

# print(ds_valid[0])

# -----------------------------3. 调整（x, y的数据集 -> 字典[transformers]）------------------------------------
def collate_fn(batch):  # batch就是ds_train数据集，按照批次分别
    # 把x, y的风格的批次数据集，转变为{"input_values"：...， "labels":...}字典的批次数据集
    # [(x1, y1), ....]  长度批次的长度  
    # 取样本，取标签
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    # 堆叠成为[B, M, M]
    x = torch.stack(x)  # 把二维x数组，堆叠成3为张量（batch_size, max_length, num_mel_bins）
    y = torch.stack(y)
    # 形成字典
    return {"input_values":x, "labels":y}

# -----------------------------4. 使用TrainingArguments与Trainer实现训练（准确率评估）------------------------------------

args = TrainingArguments(
    output_dir="/Users/logicye/Code/my_trained_models/model_speeh_sub_UP_DOWM",
    per_device_eval_batch_size=24, 
    num_train_epochs=10, 
    # 优化器
    learning_rate=5e-5,
    optim="adamw_torch",
    weight_decay=1e-4,
    lr_scheduler_type="cosine",
    # 保存策略
    save_strategy="epoch",
    # 评估
    eval_strategy="steps",  # 500步，会计算训练损失值（没有到500步，显示no Log）：默认三个值：steps, Training Loss, Eval Loss(额外加准确度)
    eval_steps=100,
    # 限制（通用的模型）
    save_only_model=True  # 只保存HuggingFace的标准的模型部分（仅仅3个）。
)


def compute_metrics(predict):
    # 计算准确率
    logits = predict.predictions
    label_ids = predict.label_ids
    losses = predict.losses
    pred = np.argmax(logits, axis=1)
    
    accuary = (pred == label_ids).sum() * 100.0 / len(label_ids)
    print(len(label_ids))
    return {
        "准确度": F"{accuary:.2f}%",
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_valid,
    data_collator=collate_fn,   # 基本掌握这个五个参数的使用（优化使用参数配置）
    compute_metrics=compute_metrics  # 评估准确率
)

trainer.train() # 开始训练