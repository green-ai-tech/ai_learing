from train import train
if __name__ == "__main__":
    # 加载模型
    from transformers import YolosForObjectDetection, YolosImageProcessor
    model = YolosForObjectDetection.from_pretrained(
        "E:/ai_learning/models/yolos-tiny", 
        num_labels = 1,   # len(ds.coco.cats)
        ignore_mismatched_sizes=True   # 加载时候其分类器就输出1个向量 
    )
    processor = YolosImageProcessor.from_pretrained("E:/ai_learning/models/yolos-tiny")
    processor.save_pretrained("./models")
    
    # 加载数据集
    from ds import get_loader
    ld_train, ld_val = get_loader(ds_path="E:/Datasets/balloon", processor=processor)


    # 训练 - 微调
    train(model=model, dataloader=ld_train, epoches=100)
    
    # YOLO加入attention在YOLO 13以后，最新YOLO26