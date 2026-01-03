#!/usr/bin/env python3
"""
Raspberry Pi Camera with YOLO Trash Detection
Detects trash vs recyclables using YOLO trained on TACO dataset
Press 'q' to quit, 's' to save snapshot with detections
"""

from picamera2 import Picamera2
import cv2
import time
from datetime import datetime
import numpy as np
from ultralytics import YOLO

# TACO dataset categories mapping to trash/recyclable
# Update this based on your specific TACO model classes
TACO_CATEGORIES = {
    # Recyclables
    'Aluminium foil': 'recyclable',
    'Battery': 'hazardous',
    'Aluminium blister pack': 'recyclable',
    'Bottle': 'recyclable',
    'Bottle cap': 'recyclable',
    'Can': 'recyclable',
    'Carton': 'recyclable',
    'Cup': 'recyclable',
    'Glass bottle': 'recyclable',
    'Metal bottle cap': 'recyclable',
    'Plastic bottle cap': 'recyclable',
    'Pop tab': 'recyclable',
    'Scrap metal': 'recyclable',
    'Plastic film': 'recyclable',
    'Six pack rings': 'recyclable',
    'Aluminium foil': 'recyclable',
    
    # Trash
    'Cigarette': 'trash',
    'Food waste': 'trash',
    'Tissues': 'trash',
    'Paper': 'recyclable',
    'Plastic bag & wrapper': 'trash',
    'Disposable plastic cup': 'trash',
    'Styrofoam piece': 'trash',
    'Unlabeled litter': 'trash',
    'Other plastic': 'trash',
}

# Colors for different categories (BGR format)
COLORS = {
    'recyclable': (0, 255, 0),    # Green
    'trash': (0, 0, 255),          # Red
    'hazardous': (0, 165, 255),    # Orange
    'unknown': (255, 255, 0)       # Yellow
}

def get_category(class_name):
    """Determine if item is trash, recyclable, or hazardous"""
    return TACO_CATEGORIES.get(class_name, 'unknown')

def draw_detections(frame, results):
    """Draw bounding boxes and labels on frame"""
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            # Determine category
            category = get_category(class_name)
            color = COLORS.get(category, COLORS['unknown'])
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{class_name}: {confidence:.2f}"
            category_label = f"[{category.upper()}]"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cat_size, _ = cv2.getTextSize(category_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(frame, (x1, y1 - label_size[1] - cat_size[1] - 10), 
                         (x1 + max(label_size[0], cat_size[0]) + 10, y1), color, -1)
            
            # Draw text
            cv2.putText(frame, category_label, (x1 + 5, y1 - cat_size[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def main():
    # Load YOLO model
    print("Loading YOLO model...")
    # Replace 'taco_yolo.pt' with your actual model path
    # You can also use 'yolov8n.pt' for testing
    try:
        model = YOLO('taco_yolo.pt')  # Your TACO-trained model
        print("Model loaded successfully!")
    except:
        print("TACO model not found. Using YOLOv8n for demonstration...")
        print("Download a TACO-trained model and replace 'taco_yolo.pt'")
        model = YOLO('yolov8n.pt')  # Fallback to default YOLO
    
    # Initialize the camera
    print("Initializing camera...")
    picam2 = Picamera2()
    
    # Create camera configuration
    camera_config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}  # Smaller size for faster inference
    )
    picam2.configure(camera_config)
    
    # Start the camera
    picam2.start()
    print("Camera started!")
    print("Press 'q' to quit, 's' to save snapshot")
    
    # Allow camera to warm up
    time.sleep(2)
    
    # FPS calculation
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Run YOLO inference
            results = model(frame_bgr, conf=0.5, verbose=False)
            
            # Draw detections
            frame_bgr = draw_detections(frame_bgr, results)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count >= 10:
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Display FPS
            cv2.putText(frame_bgr, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Trash Detection - TACO Dataset', frame_bgr)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame_bgr)
                print(f"Snapshot saved as {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        picam2.stop()
        cv2.destroyAllWindows()
        print("Camera stopped and windows closed")

if __name__ == "__main__":
    main()
