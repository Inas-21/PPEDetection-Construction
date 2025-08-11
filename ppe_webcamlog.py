import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import json
from datetime import datetime
import threading
import pygame  # For audio alerts (optional)

class AdvancedPPEDetector:
    def __init__(self, model_path="runs/detect/ppe_detection/weights/best.pt"):
        """Initialize the advanced PPE detector"""
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
        # Colors for visualization
        self.colors = {
            'helmet': (0, 255, 0),      # Green
            'no-helmet': (0, 0, 255),   # Red
            'vest': (0, 255, 0),        # Green
            'no-vest': (0, 0, 255),     # Red
            'person': (255, 0, 0)       # Blue
        }
        
        # Logging setup
        self.setup_logging()
        
        # Alert system
        self.alert_cooldown = 5  # seconds
        self.last_alert_time = 0
        
        # Statistics tracking
        self.stats = {
            'total_detections': 0,
            'compliant_count': 0,
            'non_compliant_count': 0,
            'session_start': datetime.now()
        }
        
        # Initialize audio (optional)
        try:
            pygame.mixer.init()
            self.audio_enabled = True
        except:
            self.audio_enabled = False
            print("Audio alerts disabled (pygame not available)")
    
    def setup_logging(self):
        """Setup logging directory and files"""
        self.log_dir = "ppe_logs"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(f"{self.log_dir}/screenshots", exist_ok=True)
        
        self.log_file = f"{self.log_dir}/ppe_log_{datetime.now().strftime('%Y%m%d')}.json"
    
    def log_detection(self, status, detections_info):
        """Log detection results"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'detections': detections_info,
            'frame_count': self.stats['total_detections']
        }
        
        # Append to log file
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2)
        except Exception as e:
            print(f"Logging error: {e}")
    
    def play_alert_sound(self, alert_type):
        """Play alert sound for violations"""
        if not self.audio_enabled:
            return
            
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
            
        try:
            # Create a simple beep sound
            duration = 200  # milliseconds
            freq = 800 if alert_type == "violation" else 400
            
            # Generate sine wave
            sample_rate = 22050
            frames = int(duration * sample_rate / 1000)
            arr = np.zeros((frames, 2))
            
            for i in range(frames):
                time_val = float(i) / sample_rate
                wave = np.sin(2 * np.pi * freq * time_val)
                arr[i] = [wave, wave]
            
            arr = (arr * 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(arr)
            sound.play()
            
            self.last_alert_time = current_time
        except Exception as e:
            print(f"Audio alert error: {e}")
    
    def analyze_detections(self, detections):
        """Analyze detections for safety compliance"""
        detected_objects = []
        
        if detections.boxes is not None:
            for cls, conf in zip(detections.boxes.cls, detections.boxes.conf):
                class_name = self.class_names[int(cls)]
                detected_objects.append({
                    'class': class_name,
                    'confidence': float(conf)
                })
        
        # Determine compliance status
        has_person = any(obj['class'] == 'person' for obj in detected_objects)
        has_helmet = any(obj['class'] == 'helmet' for obj in detected_objects)
        has_vest = any(obj['class'] == 'vest' for obj in detected_objects)
        has_no_helmet = any(obj['class'] == 'no-helmet' for obj in detected_objects)
        has_no_vest = any(obj['class'] == 'no-vest' for obj in detected_objects)
        
        if not has_person:
            status = "NO_PERSON"
            color = (128, 128, 128)
            alert_needed = False
        elif has_helmet and has_vest:
            status = "COMPLIANT"
            color = (0, 255, 0)
            alert_needed = False
            self.stats['compliant_count'] += 1
        elif has_no_helmet or has_no_vest:
            status = "NON_COMPLIANT"
            color = (0, 0, 255)
            alert_needed = True
            self.stats['non_compliant_count'] += 1
            self.play_alert_sound("violation")
        else:
            status = "UNCLEAR"
            color = (0, 255, 255)
            alert_needed = False
        
        self.stats['total_detections'] += 1
        
        # Log the detection
        self.log_detection(status, detected_objects)
        
        return status, color, detected_objects, alert_needed
    
    def draw_enhanced_ui(self, frame, status, color, detected_objects):
        """Draw enhanced UI with statistics"""
        height, width = frame.shape[:2]
        
        # Main status panel
        cv2.rectangle(frame, (10, 10), (width - 10, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (width - 10, 100), color, 3)
        
        cv2.putText(frame, "PPE SAFETY MONITORING", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(frame, f"Detections: {len(detected_objects)}", (20, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Statistics panel
        stats_y_start = height - 120
        cv2.rectangle(frame, (10, stats_y_start), (300, height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, stats_y_start), (300, height - 10), (255, 255, 255), 2)
        
        cv2.putText(frame, "SESSION STATS", (20, stats_y_start + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Total: {self.stats['total_detections']}", (20, stats_y_start + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Compliant: {self.stats['compliant_count']}", (20, stats_y_start + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, f"Violations: {self.stats['non_compliant_count']}", (20, stats_y_start + 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Detection details panel
        details_x_start = width - 250
        cv2.rectangle(frame, (details_x_start, 120), (width - 10, 300), (0, 0, 0), -1)
        cv2.rectangle(frame, (details_x_start, 120), (width - 10, 300), (255, 255, 255), 2)
        
        cv2.putText(frame, "DETECTED OBJECTS", (details_x_start + 10, 145), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i, obj in enumerate(detected_objects[:8]):  # Show max 8 objects
            y_pos = 165 + i * 20
            text = f"{obj['class']}: {obj['confidence']:.2f}"
            color_obj = self.colors.get(obj['class'], (255, 255, 255))
            cv2.putText(frame, text, (details_x_start + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_obj, 1)
    
    def save_violation_screenshot(self, frame):
        """Save screenshot when violation is detected"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.log_dir}/screenshots/violation_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename
    
    def run_monitoring_system(self, camera_index=0, save_violations=True):
        """Run the advanced PPE monitoring system"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("=== PPE MONITORING SYSTEM STARTED ===")
        print("Controls:")
        print("  'q' - Quit system")
        print("  's' - Save screenshot")
        print("  'r' - Reset statistics")
        print("  'l' - Show logs location")
        
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                results = self.model(frame, verbose=False)
                
                for result in results:
                    # Draw detections
                    frame = self.draw_detections(frame, result)
                    
                    # Analyze compliance
                    status, status_color, detected_objects, alert_needed = self.analyze_detections(result)
                    
                    # Draw enhanced UI
                    self.draw_enhanced_ui(frame, status, status_color, detected_objects)
                    
                    # Save violation screenshots
                    if alert_needed and save_violations:
                        screenshot_path = self.save_violation_screenshot(frame)
                        print(f"Violation detected! Screenshot saved: {screenshot_path}")
                
                # FPS calculation
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Draw FPS
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (frame.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Advanced PPE Monitoring System', frame)
                
                # Handle controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{self.log_dir}/screenshots/manual_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot saved: {filename}")
                elif key == ord('r'):
                    self.stats = {
                        'total_detections': 0,
                        'compliant_count': 0,
                        'non_compliant_count': 0,
                        'session_start': datetime.now()
                    }
                    print("Statistics reset!")
                elif key == ord('l'):
                    print(f"Logs directory: {os.path.abspath(self.log_dir)}")
        
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("=== PPE MONITORING SYSTEM STOPPED ===")
            self.print_final_stats()
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels"""
        if detections.boxes is not None:
            for box, conf, cls in zip(detections.boxes.xyxy, detections.boxes.conf, detections.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                class_name = self.class_names[int(cls)]
                color = self.colors.get(class_name, (255, 255, 255))
                confidence = float(conf)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"{class_name}: {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def print_final_stats(self):
        """Print final session statistics"""
        duration = datetime.now() - self.stats['session_start']
        total = self.stats['total_detections']
        compliant = self.stats['compliant_count']
        violations = self.stats['non_compliant_count']
        
        print(f"\n=== SESSION SUMMARY ===")
        print(f"Duration: {duration}")
        print(f"Total detections: {total}")
        print(f"Compliant: {compliant} ({compliant/max(total,1)*100:.1f}%)")
        print(f"Violations: {violations} ({violations/max(total,1)*100:.1f}%)")
        print(f"Logs saved to: {os.path.abspath(self.log_dir)}")

def main():
    """Main function"""
    detector = AdvancedPPEDetector("runs/detect/ppe_detection/weights/best.pt")
    detector.run_monitoring_system(camera_index=0, save_violations=True)

if __name__ == "__main__":
    main()