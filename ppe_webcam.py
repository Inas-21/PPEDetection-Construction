import cv2
from ultralytics import YOLO
import time

class PPEDetector:
    def __init__(self, model_path="runs/detect/ppe_detection/weights/best.pt"):
        """Initialize the PPE detector with your trained model"""
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
        # Define colors for each class (BGR format)
        self.colors = {
            'helmet': (0, 255, 0),      
            'no-helmet': (0, 0, 255),   
            'vest': (0, 255, 0),       
            'no-vest': (0, 0, 255),     
            'person': (255, 0, 0)       
        }
        
        # Safety compliance tracking
        self.compliance_status = "CHECKING..."
        
    def check_safety_compliance(self, detections):
        """Check if detected person is wearing proper PPE"""
        detected_classes = [self.class_names[int(cls)] for cls in detections.boxes.cls]
        
        has_person = 'person' in detected_classes
        has_helmet = 'helmet' in detected_classes
        has_vest = 'vest' in detected_classes
        has_no_helmet = 'no-helmet' in detected_classes
        has_no_vest = 'no-vest' in detected_classes
        
        if not has_person:
            return "NO PERSON DETECTED", (128, 128, 128)  # Gray
        
        # Check compliance
        if has_helmet and has_vest:
            return "✓ COMPLIANT", (0, 255, 0)  # Green
        elif has_no_helmet or has_no_vest:
            return "✗ NON-COMPLIANT", (0, 0, 255)  # Red
        elif has_helmet or has_vest:
            return "⚠ PARTIALLY COMPLIANT", (0, 165, 255)  # Orange
        else:
            return "? UNCLEAR", (0, 255, 255)  # Yellow
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on the frame"""
        if detections.boxes is not None:
            for box, conf, cls in zip(detections.boxes.xyxy, detections.boxes.conf, detections.boxes.cls):
                # Get coordinates
                x1, y1, x2, y2 = map(int, box)
                
                # Get class name and color
                class_name = self.class_names[int(cls)]
                color = self.colors.get(class_name, (255, 255, 255))
                confidence = float(conf)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with confidence
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                # Draw label background
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def draw_safety_status(self, frame, status, color):
        """Draw safety compliance status on frame"""
        height, width = frame.shape[:2]
        
        # Draw status background
        cv2.rectangle(frame, (10, 10), (width - 10, 80), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (width - 10, 80), color, 3)
        
        # Draw status text
        cv2.putText(frame, "PPE SAFETY CHECK", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, status, (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    def run_webcam_detection(self, camera_index=0):
        """Run real-time PPE detection on webcam feed"""
        # Initialize webcam
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting PPE Detection...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        # FPS calculation
        fps_counter = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Run YOLO detection
            results = self.model(frame, verbose=False)
            
            # Process detections
            for result in results:
                # Draw detections
                frame = self.draw_detections(frame, result)
                
                # Check safety compliance
                status, status_color = self.check_safety_compliance(result)
                self.draw_safety_status(frame, status, status_color)
            
            # Calculate and display FPS
            fps_counter += 1
            fps_time = time.time() - fps_start_time
            if fps_time >= 1.0:
                fps = fps_counter / fps_time
                fps_counter = 0
                fps_start_time = time.time()
                
                # Draw FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('PPE Detection System', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"ppe_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("PPE Detection stopped")

def main():
    """Main function to run the PPE detection system"""
    # Initialize detector with your model
    detector = PPEDetector("runs/detect/ppe_detection/weights/best.pt")
    
    # Run webcam detection
    detector.run_webcam_detection(camera_index=0)  # Use 0 for default camera

if __name__ == "__main__":
    main()