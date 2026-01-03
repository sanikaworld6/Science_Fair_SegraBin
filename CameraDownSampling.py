#!/usr/bin/env python3
"""
Smart Waste Sorting Camera
Full frame capture (1920x1080) with downsampled detection (640x480)
Press 'q' to quit, 's' to save snapshot
"""

from picamera2 import Picamera2
import cv2
import time
from datetime import datetime
from ultralytics import YOLO

# Comprehensive mapping of all 60 classes to recyclable/trash/hazardous
CATEGORY_MAPPING = {
    # RECYCLABLE - Metals
    'Aluminium foil': 'recyclable',
    'Aluminium blister pack': 'recyclable',
    'Metal bottle cap': 'recyclable',
    'Food Can': 'recyclable',
    'Aerosol': 'recyclable',
    'Drink can': 'recyclable',
    'Metal lid': 'recyclable',
    'Scrap metal': 'recyclable',
    'Pop tab': 'recyclable',
    
    # RECYCLABLE - Glass
    'Glass bottle': 'recyclable',
    'Broken glass': 'recyclable',
    'Glass cup': 'recyclable',
    'Glass jar': 'recyclable',
    
    # RECYCLABLE - Clear Plastics & Bottles
    'Clear plastic bottle': 'recyclable',
    'Other plastic bottle': 'recyclable',
    'Plastic bottle cap': 'recyclable',
    
    # RECYCLABLE - Paper & Cardboard
    'Corrugated carton': 'recyclable',
    'Other carton': 'recyclable',
    'Egg carton': 'recyclable',
    'Drink carton': 'recyclable',
    'Magazine paper': 'recyclable',
    'Normal paper': 'recyclable',
    'Paper bag': 'recyclable',
    'Wrapping paper': 'recyclable',
    'Toilet tube': 'recyclable',
    'Pizza box': 'recyclable',
    'Meal carton': 'recyclable',
    
    # TRASH - Non-recyclable plastics
    'Carded blister pack': 'trash',
    'Other plastic': 'trash',
    'Plastic film': 'trash',
    'Six pack rings': 'trash',
    'Garbage bag': 'trash',
    'Other plastic wrapper': 'trash',
    'Single-use carrier bag': 'trash',
    'Polypropylene bag': 'trash',
    'Crisp packet': 'trash',
    'Plastic glooves': 'trash',
    'Plastic straw': 'trash',
    'Plastified paper bag': 'trash',
    
    # TRASH - Disposable items
    'Disposable plastic cup': 'trash',
    'Foam cup': 'trash',
    'Paper cup': 'trash',
    'Other plastic cup': 'trash',
    'Disposable food container': 'trash',
    'Foam food container': 'trash',
    'Other plastic container': 'trash',
    'Styrofoam piece': 'trash',
    'Plastic utensils': 'trash',
    'Paper straw': 'trash',
    
    # TRASH - Food & organic
    'Food waste': 'trash',
    'Tissues': 'trash',
    
    # TRASH - Containers with residue
    'Spread tub': 'trash',
    'Tupperware': 'trash',
    'Squeezable tube': 'trash',
    'Plastic lid': 'trash',
    
    # TRASH - Misc
    'Rope & strings': 'trash',
    'Shoe': 'trash',
    'Unlabeled litter': 'trash',
    'Cigarette': 'trash',
    
    # HAZARDOUS
    'Battery': 'hazardous',
}

# Colors (RGB format)
COLORS = {
    'recyclable': (0, 255, 0),    # Green
    'trash': (255, 0, 0),          # Red
    'hazardous': (255, 165, 0),    # Orange
    'unknown': (255, 255, 0)       # Yellow
}

def get_category(class_name):
    """Determine if item is recyclable, trash, or hazardous"""
    return CATEGORY_MAPPING.get(class_name, 'unknown')

def draw_detections(frame, results, scale_x, scale_y):
    """Draw bounding boxes and labels on full resolution frame"""
    for result in results:
        boxes = result.boxes
        
        for box in boxes:
            # Get coordinates from downsampled detection
            x1, y1, x2, y2 = box.xyxy[0]
            
            # Scale up to full frame resolution
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            
            category = get_category(class_name)
            color = COLORS.get(category, COLORS['unknown'])
            
            # Draw bounding box (thicker for full res)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            
            # Prepare labels
            label = f"{class_name}: {confidence:.2f}"
            category_label = f"[{category.upper()}]"
            
            # Draw label background (larger font for full res)
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cat_size, _ = cv2.getTextSize(category_label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            
            cv2.rectangle(frame, (x1, y1 - label_size[1] - cat_size[1] - 20), 
                         (x1 + max(label_size[0], cat_size[0]) + 20, y1), color, -1)
            
            # Draw text (larger font)
            cv2.putText(frame, category_label, (x1 + 10, y1 - cat_size[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(frame, label, (x1 + 10, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    return frame

def main():
    print("Loading waste sorting model...")
    model = YOLO('waste_sorting.pt')
    print("✅ Model loaded!")
    print(f"Total classes: {len(model.names)}")
    
    print("Initializing camera (full frame)...")
    picam2 = Picamera2()
    
    # Capture full frame at 1920x1080
    camera_config = picam2.create_preview_configuration(
        main={"size": (1280, 960), "format": "RGB888"}
    )
    picam2.configure(camera_config)
    picam2.start()
    
    print("✅ Camera started at 1920x1080!")
    print("Detection runs at 640x480 for speed")
    print("Press 'q' to quit, 's' to save snapshot\n")
    
    time.sleep(2)
    
    # Calculate scaling factors
    DETECTION_WIDTH = 640
    DETECTION_HEIGHT = 480
    
    fps = 0
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture full resolution frame
            frame = picam2.capture_array()
            
            # Calculate scale factors
            scale_x = frame.shape[1] / DETECTION_WIDTH
            scale_y = frame.shape[0] / DETECTION_HEIGHT
            
            # Downsample for detection (faster)
            frame_small = cv2.resize(frame, (DETECTION_WIDTH, DETECTION_HEIGHT))
            
            # Run detection on small frame
            results = model(frame_small, conf=0.4, verbose=False)
            
            # Draw detections on full frame with scaled coordinates
            frame = draw_detections(frame, results, scale_x, scale_y)
            
            # Calculate FPS
            frame_count += 1
            if frame_count >= 10:
                fps = frame_count / (time.time() - start_time)
                frame_count = 0
                start_time = time.time()
            
            # Display FPS (larger for full res)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            
            # Display resolution info
            cv2.putText(frame, f"Display: {frame.shape[1]}x{frame.shape[0]}", (20, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Detection: {DETECTION_WIDTH}x{DETECTION_HEIGHT}", (20, 160),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Waste Sorting Camera - Full Frame', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"waste_sort_fullframe_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Snapshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("✅ Camera stopped")

if __name__ == "__main__":
    main()
