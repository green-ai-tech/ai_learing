import os
import torch
import pickle
import numpy as np
import importlib.util

# ============================================================================
# 1. 动态导入模型类
# ============================================================================
# 因为文件名以数字开头，不能直接用 import 语句，需要使用 importlib
current_dir = os.path.dirname(os.path.abspath(__file__))
ai_code_path = os.path.join(current_dir, "03_ai_code.py")

spec = importlib.util.spec_from_file_location("ai_code_module", ai_code_path)
ai_code_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ai_code_module)

# 从加载的模块中获取模型类
TransformerTextClassifier = ai_code_module.TransformerTextClassifier

# ============================================================================
# 2. 加载配置和映射
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH   = os.path.join(PROJECT_ROOT, "vocab")
MODELS_PATH  = os.path.join(PROJECT_ROOT, "models")

# 使用 03_ai_code.py 中定义的常量，保证与训练时完全一致
SEQUENCE_LENGTH = ai_code_module.SEQUENCE_LENGTH
START_NO = ai_code_module.START_NO
UNK_NO = ai_code_module.UNK_NO
PAD_NO = ai_code_module.PAD_NO

print("正在加载词表...")
with open(os.path.join(VOCAB_PATH, "labels.pk"), "rb") as f:
    labels = pickle.load(f)
with open(os.path.join(VOCAB_PATH, "chars.pk"), "rb") as f:
    chars = pickle.load(f)

label_to_id = {l: i for i, l in enumerate(labels)}
id_to_label = {i: l for i, l in enumerate(labels)}
char_to_id  = {c: i for i, c in enumerate(chars)}
num_classes = len(labels)

# ============================================================================
# 3. 加载模型
# ============================================================================
print("正在初始化模型结构...")
# 注意：这里的参数必须与 03_ai_code.py 训练时完全一致！
model = TransformerTextClassifier(
    vocab_size=len(chars) + 4,
    embedding_dim=300,
    num_classes=num_classes,
    nhead=5,                # 💡 必须与训练时一致
    num_layers=2,           # 💡 必须与训练时一致
    dim_feedforward=256,    # 💡 必须与训练时一致
    dropout=0.1
)

# 加载权重
model_path = os.path.join(MODELS_PATH, "best_transformer_model.pth")
print(f"正在加载模型权重: {model_path}")
checkpoint = torch.load(model_path, map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # 切换到评估模式
print("✅ 模型加载完成！")

# ============================================================================
# 4. 推理函数
# ============================================================================
def predict(text):
    """输入一段文本，输出预测标签"""
    # 1. 文本转 ID
    char_ids = []
    for char in text:
        if char in char_to_id:
            char_ids.append(char_to_id[char] + START_NO)
        else:
            char_ids.append(UNK_NO)

    # 2. 截断与填充
    if len(char_ids) > SEQUENCE_LENGTH:
        char_ids = char_ids[:SEQUENCE_LENGTH]
    else:
        char_ids = char_ids + [PAD_NO] * (SEQUENCE_LENGTH - len(char_ids))

    # 3. 转 Tensor
    x = torch.tensor([char_ids], dtype=torch.long)

    # 4. 推理
    with torch.no_grad():
        output = model(x)
        prob = torch.softmax(output, dim=1)
        pred_id = torch.argmax(prob, dim=1).item()

    return id_to_label[pred_id], prob[0][pred_id].item()

# ============================================================================
# 5. 开始测试
# ============================================================================
print("\n" + "="*50)
print("模型推理测试 (输入 'quit' 退出)")
print("="*50)

while True:
    try:
        text = input("\n请输入文本: ")
    except EOFError:
        break
        
    if text.lower() == 'quit':
        break
    if not text.strip():
        continue

    label, confidence = predict(text)
    print(f"👉 预测结果: 【{label}】 (置信度: {confidence:.4f})")
