from models.Lenet5 import Lenet5
from utils.dataset import load_minist
import torch.nn as nn
import torch.optim as optim
import torch

# 训练一轮
def train_one(model, loader, loss_fun, optimizer, device="cpu"):
    model.train()  # 目前没有，对复杂网络有意义。（有两个会执行dropout, batchnorm）
    model = model.to(device)
    for x, y in loader:
        # 梯度清零
        optimizer.zero_grad()  
        x = x.to(device)
        y = y.to(device)
        # 推理
        y_ = model(x)
        # 计算损失值
        loss = loss_fun(y_, y)
        # 自动求导
        loss.backward()
        # 梯度更新
        optimizer.step()
    
# 评估
@torch.no_grad()    # 标注：保护函数中的所有运算，不会被自动求导跟踪
def evaluate(model, loader, loss_fun, device="cpu"):
    model = model.eval()
    model = model.to(device)
    total = 0
    num_correct = 0
    num_loss = 0.0
    for x, y in  loader:
        x = x.to(device)
        y = y.to(device)
        # 预测
        y_ = model(x)
        # 计算损失
        loss = loss_fun(y_, y)
        num_loss += loss.detach().cpu().item()              # 累加损失度
        # 计算准确率
        p, c = torch.max(y_, dim=1)
        num_correct += (c==y).sum().detach().cpu().item()   # 累加正确数量
        total += y.size()[0]
        
    print(F"\t|-损失：{num_loss:.6f}, 精确度：{num_correct / total * 100:0.2f}%")
        
        

# 所有训练
def train(epoches=100, learning_rate=0.0001):
    # 创建模型
    _model = Lenet5()
    # 加载数据集
    _ld_train, _ld_valid = load_minist()
    # 定义损失函数
    _loss_fun = nn.CrossEntropyLoss()
    # 定义优化器  
    _optimizer = optim.Adam(_model.parameters(), lr=learning_rate)
    
    # 判断当前环境是否支持gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 训练
    for epoch in range(epoches):
        print(F"第{epoch:04d}轮")
        train_one(_model, _ld_train, _loss_fun, _optimizer, device=device)
        # 评估
        evaluate(_model, _ld_valid, _loss_fun, device=device)
        # 保存
        torch.save(_model.state_dict(), "lenet5.pth")

if __name__ == "__main__":
    train(epoches=100, learning_rate=0.0005)