from transformers import ASTFeatureExtractor, ASTConfig, ASTForAudioClassification
from transformers import TrainingArguments, Trainer

from torchaudio.datasets import SPEECHCOMMANDS
import os
import torch
import numpy as np

# ====================== 3 个标签 =================================
SELECTED_LABELS = ["yes", "no", "up"]

# 你的路径
ds_dir = "/Users/logicye/Code/Datasets/SpeechCommands/speech_commands_v0.02"

# 标签映射
labels = SELECTED_LABELS
label2id = {d: idx for idx, d in enumerate(labels)}
id2label = {idx: d for idx, d in enumerate(labels)}

# ====================== 模型配置 ================================
config = ASTConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    initializer_range=0.02,
    layer_norm_eps=1e-12,
    patch_size=16,
    qkv_bias=True,
    frequency_stride=10,
    time_stride=10,
    max_length=128,
    num_mel_bins=128,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
)

#model = ASTForAudioClassification(config)

model = ASTForAudioClassification.from_pretrained("/Users/logicye/Code/my_trained_models/model_speeh_sub/checkpoint-41877")

# ====================== 特征提取器 ===============================
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

# ====================== 安全数据集（无递归！） ======================
class ASTDataset(SPEECHCOMMANDS):
    def __init__(self, extractor, ds_path="/Users/logicye/Code/Datasets/", subset="training", download=True):
        super().__init__(root=ds_path, download=download, subset=subset)
        self.extractor = extractor
        self.allowed_labels = SELECTED_LABELS

        # 提前过滤所有合法索引
        self._filtered_indices = [
            i for i in range(len(self._walker))
            if self._get_label(i) in self.allowed_labels
        ]

    # 辅助函数：获取标签
    def _get_label(self, idx):
        file_path = self._walker[idx]
        label = os.path.basename(os.path.dirname(file_path))
        return label

    def __len__(self):
        return len(self._filtered_indices)

    def __getitem__(self, idx):
        real_idx = self._filtered_indices[idx]
        data, sample_rate, label, speaker_id, wav_id = super().__getitem__(real_idx)

        feature = self.extractor(data[0], sample_rate, return_tensors="pt")
        return feature["input_values"][0], torch.tensor(label2id[label])  # ✅ 修复标签形状

# 加载数据集
ds_train = ASTDataset(extractor=feature_extractor, subset="training")
ds_valid = ASTDataset(extractor=feature_extractor, subset="validation")

# ====================== 批次处理 =============================
def collate_fn(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    x = torch.stack(x)
    y = torch.tensor(y)  # ✅ 修复标签形状
    return {"input_values": x, "labels": y}

# ====================== 极小批次训练配置 ========================
args = TrainingArguments(
    output_dir="/Users/logicye/Code/my_trained_models/model_speeh_sub",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    learning_rate=5e-5,
    optim="adamw_torch",
    weight_decay=1e-4,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
    eval_strategy="steps",
    eval_steps=50,
    save_only_model=True
)

# ✅ 正确计算准确率
def compute_metrics(predict):
    logits = predict.predictions
    label_ids = predict.label_ids

    # 预测类别
    pred = np.argmax(logits, axis=1)

    # 正确数量 / 总数
    correct = np.sum(pred == label_ids)
    total = len(label_ids)
    accuracy = correct / total * 100

    return {"准确度": f"{accuracy:.2f}%"}

# 训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_valid,
    data_collator=collate_fn,
    compute_metrics=compute_metrics
)

trainer.train()