import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('./config/yolov8l.yaml')   # 根据yaml文件初始化模型
    model.load('yolov8l.pt')  # 加载预训练权重
    model.train(data='./datasets/coco128.yaml',   # 数据集yaml文件
                imgsz=640,
                epochs=10,
                batch=4,
                device=0
    )