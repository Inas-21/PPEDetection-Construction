from ultralytics import YOLO

def train_ppe_model():
    # Load a pre-trained model 
    model = YOLO("yolov8s.pt")  

    # Train the model on your custom dataset
    results = model.train(
        data="data\data.yaml",  
        epochs=50,                 
        imgsz=640,                 
        batch=16,                 
        name="ppe_detection",  
    )

    return model, results

if __name__ == "__main__":
    model, results = train_ppe_model()