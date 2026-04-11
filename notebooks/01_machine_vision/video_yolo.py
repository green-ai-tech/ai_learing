def detect_objects_by_yolo(img, model_file="models/yolo26l-seg.pt"):
    """
        文档注释
        作者：Louis Young     # 模块文档注释。
        日期：2026-03-20
        功能：加载yolo的分割模型，实现图片中的目标分割
        参数：
            img：输入的需要分割目标的图像
            model_file:离线下载yolo分隔模型文件，要求是pytorch格式
        返回：
            obj_names（目标名）, obj_confs（置信度）, obj_boxes（目标边框）, obj_masks（分隔的目标遮罩）
    """
    # from ultralytics import YOLO
    import ultralytics  # 网络模型
    import numpy        # 图像运算
    import cv2          # 图像处理
    import torch        # 暂时不直接使用
    
    model_seg = ultralytics.YOLO(model_file)
    # 选择使用gpu还是cpu
    model_seg = model_seg.cuda()  # 使用gpu
    # model_seg.cpu()   # 使用cpu
    results = model_seg(img)[0]   # 基本上都支持（方便）
    # results = model_seg.forward("jiaotong.jpg")[0]  # forward 在YOLO没有实现。 pytorch的标准
    # results = model_seg.predict("jiaotong.jpg")[0]  # 是为了以scikit-learning模块兼容。（机器学习库）
    boxes = results.boxes  # 侦测的目标
    masks = results.masks  # 分隔
    names = results.names  # 用来把id转换为名字
    
    # 名字
    # boxes.cls
    obj_names = [names[int(c)] for c in boxes.cls.cpu().numpy()]   # 生成表达式（逐次生成：效率高）
    # 概率
    obj_confs = boxes.conf.cpu().numpy()   # .tolist()  # numpy数组，python数组
    # 边框
    obj_boxes = boxes.xyxy.cpu().numpy().astype(int)
    # # 分隔目标
    if  masks.data is None:
        obj_masks = None
    else:
        obj_masks = masks.data.cpu().numpy() * 255 # 与原图不一致
        
    return obj_names, obj_confs, obj_boxes, obj_masks

def main():
    """
        ...
    """
    import cv2
    import numpy
    dev = cv2.VideoCapture(0)
    model_file="../../models/yolo26l-seg.pt"
    # 返回第一个值：读取的状态，True成功。False表示失败，frame返回图像。
    fengjing = cv2.imread("assets/images/05_landscape.jpg")
    while True:
        status, frame = dev.read()
        if status == True:
            #######################################################
            cls, conf, box, mask = detect_objects_by_yolo(frame)
            # 标注
            # for c, f, b, m in zip(cls, conf, box, mask):
            #     # 绘制矩形
            #     cv2.rectangle(frame, pt1=(b[0], b[1]), pt2=(b[2], b[3]), color=(255, 0,  0), thickness=1)
            #     # 绘制信息
            #     cv2.putText(frame, F"{c}:{f:.2f}", org=(b[0], b[1]), color=(255, 0, 255), fontFace=0, fontScale=0.5, thickness=1)
            # 换背景
            result_img = frame
            if mask is None:
                pass
            else:
                h, w, c = frame.shape # 获取原图的大小
                for c, m in zip(cls, mask):  # 处理每个目标
                    if c == "person":
                        # 挖空
                        # 1. 合并成彩色图像（注意缩放）
                        color_m = numpy.stack([m, m, m], axis=2)
                        color_m = cv2.resize(color_m, (w, h))
                        # 2. 与原图做且运算
                        new_img = frame & color_m
                        # 处理背景
                        # 1. 读取背景(放在循环外，读取一次)
                        fengjing_new = cv2.resize(fengjing, (w, h)) # 缩放
                        color_m = 255 - color_m
                        new_img2 = fengjing_new & color_m
                        # 得到最后效果
                        result_img = new_img + new_img2
            ########################################################
            cv2.imshow("Image", result_img)
            # 按键退出
            if cv2.waitKey(80) == ord('q'):
                break
        else:
            print("读取失败！")
            break
    
    # 完成任务释放资源
    dev.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":  # 做为模块调用，条件为False，main不会执行。否则作为独立程序使用。
    main()

b




