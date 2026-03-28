import torch
# 训练一次
def train_one(model, dataloader, optimizer, scaler, device):
    # 损失函数（属于YOLO网络的一部分，不需要，在forward计算直接返回损失值）
    # 优化器
    model.train()
    model.to(device)
    v_loss = 0  # 一轮的总的损失
    total = 0   # 训练的所有样本数量
    for data in dataloader:
        total += len(data)
        optimizer.zero_grad()
        # 把数据迁移到device（data是字典）
        data["pixel_values"] = data["pixel_values"].to(device)   # 把样本图像，迁移到device
        # 标签迁移到GPU
        labels = []
        for label in data["labels"]:
            label = {k:v.to(device) for k, v in label.items()}   # 小技巧
            labels.append(label)
        # 使用labels覆盖data中的labels
        data["labels"] = labels
        # 前向预测（顺带计算损失值）- 混合精度
        with torch.autocast(device_type=device):
            outputs = model(**data)
        # 获取损失值
        loss = outputs.loss
        # 自动求导
        scaler.scale(loss).backward()
        # 梯度更新
        scaler.step(optimizer)
        scaler.update()
        # 记录损失值
        v_loss += loss.detach().cpu().item()
    print(F"\t平均损失：{v_loss / total:.6f}")


def train(model, dataloader, epoches= 1000, lr=2.5e-5, weight_decay=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )
    scaler = torch.GradScaler()
    print('开始训练')
    for e in range(epoches):
        # 训练
        print(F'第{e:04d}轮')
        train_one(model=model, dataloader=dataloader, optimizer=optimizer, scaler=scaler, device=device)
        # 评估
        # 保存模型
        model.save_pretrained("./models")


    