"""
Author  : Louis Young
Date    : 2026-03-24
    主要数据集加载数据集
"""
from torchvision.transforms import Compose, ToTensor  # 加载数据集负责转为张量
from torch.utils.data import DataLoader               # 批次数据集
from torchvision.datasets import MNIST                # 数据集

def load_minist(root="./minist", batch_size=10000):
    # 定义转换
    transform = Compose([ToTensor()])
    # 加载数据集
    _ds_train = MNIST(root=root, train=True, download=True, transform=transform)
    _ds_valid = MNIST(root=root, train=False, download=True, transform=transform)
    # 构建批次数据集
    _loader_train = DataLoader(_ds_train, shuffle=True, batch_size=batch_size)
    _loader_valid = DataLoader(_ds_valid, shuffle=False, batch_size=batch_size)
    return _loader_train, _loader_valid

if __name__ == "__main__":
    # 测试
    loader_train, loader_valid = load_minist()
    for x, y in loader_train:
        print(x.shape, y.shape)
        break

     