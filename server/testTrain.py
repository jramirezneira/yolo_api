from ultralytics import YOLO
import torch


model = YOLO("best.pt") 

if __name__ == '__main__':
    results = model.train(
        data="C:\proyectos\yolo\yolo_dataset\Facial Data.v2i.yolov11\data.yaml", 
        epochs=10,  
        imgsz=[384,640], 
        batch=16, 
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    results = model.val()  # evaluate model performance on the validation set