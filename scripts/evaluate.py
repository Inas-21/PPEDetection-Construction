from ultralytics import YOLO
import multiprocessing

def main():
    # Load the best model
    model = YOLO("models/best.pt")
    
    # Run validation
    metrics = model.val()
    
    # Run inference on test image
    results = model("data/test/images/ppe_0139_jpg.rf.18e36dc211ea77c46e230ffc30583593.jpg")
    results[0].show()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()