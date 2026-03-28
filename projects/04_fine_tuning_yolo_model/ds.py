from torchvision.datasets import CocoDetection
from transformers import YolosImageProcessor
import os
from torch.utils.data import DataLoader

class YOLODataset(CocoDetection):
    # 构造器
    def __init__(self, ds_path, processor, train=True):
        # ds_path指定根
        img_dir  = os.path.join(ds_path, "train" if train else "val") 
        ann_file = os.path.join(ds_path, "train.json" if train else "val.json")
        super(YOLODataset, self).__init__(root=img_dir, annFile=ann_file)
        
        self.processor = processor  # 在后面进行预处理
    
    # 索引(运算符：[])
    def __getitem__(self, index):
        # 使用父类的__getitem__返回没哟预处理的数据样本与标签
        img, target = super(YOLODataset, self).__getitem__(index)
        # 预处理
        target = {
            "image_id": self.ids[index],    # 样本从新编号
            "annotations": target
        }
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        return  encoding["pixel_values"].squeeze(), encoding["labels"][0]


def get_loader(ds_path, processor, batch_size=2):
    ds_train = YOLODataset(ds_path, processor, True)
    ds_val   = YOLODataset(ds_path, processor, False)
    
    def collate_fn(batch):
        
        # 取值
        x = [ item[0] for item in  batch]   # 取样本
        y = [ item[1] for item in  batch]   # 取标签
        # 对齐
        x = processor.pad(x, return_tensors="pt")
        # 重新构建字典
        batch = {}
        batch["pixel_values"] = x["pixel_values"]
        batch["labels"]       = y
        
        return batch
    
    ld_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    ld_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return ld_train, ld_val

# if __name__ == "__main__":
#     processor = YolosImageProcessor.from_pretrained("F:/03Models/yolos-tiny")
#     # ds = YOLODataset(ds_path="F:/04Datasets/balloon", processor=processor, train=True)
#     # print(ds[0][1])
#     # print(ds[0][0].shape)
#     tr, va = get_loader(ds_path="F:/04Datasets/balloon", processor=processor)
#     for data in tr:
#         # print(data)
#         print(data["pixel_values"].shape)
#         print(data["labels"])
#         break
    