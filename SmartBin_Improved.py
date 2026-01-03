#!/usr/bin/env python3
"""
Smart Waste Sorting Bin - Improved Model
22-class model with better trash detection
Button press -> Camera detects -> Servo sorts -> Return to idle
LIVE CAMERA VIEW - Always shows what camera sees
"""

from picamera2 import Picamera2
import cv2
import time
from datetime import datetime
from ultralytics import YOLO
import RPi.GPIO as GPIO

# GPIO Pin Configuration
SERVO_PIN = 18   # GPIO 18 (Physical Pin 12)
BUTTON_PIN = 17  # GPIO 17 (Physical Pin 11) - POWER SW
LED_PIN = 27     # GPIO 27 (Physical Pin 13) - LED control

# Servo positions
POSITION_CENTER = 7   # 90 degrees (idle/flat)
POSITION_LEFT = 3     # 0 degrees (TRASH)
POSITION_RIGHT = 11   # 180 degrees (RECYCLABLE)

# Detection settings
DETECTION_WIDTH = 640
DETECTION_HEIGHT = 480
CONFIDENCE_THRESHOLD = 0.35  # Lower threshold for better detection

# Category mapping for 22 classes
CATEGORY_MAPPING = {
    # RECYCLABLE - Metals
    'battery': 'hazardous',  # Special handling
    'can': 'recyclable',
    
    # RECYCLABLE - Cardboard/Paper
    'cardboard_bowl': 'recyclable',
    'cardboard_box': 'recyclable',
    'reuseable_paper': 'recyclable',
    'scrap_paper': 'recyclable',
    
    # HAZARDOUS - Chemical containers (treat as trash for safety)
    'chemical_plastic_bottle': 'hazardous',
    'chemical_plastic_gallon': 'hazardous',
    'chemical_spray_can': 'hazardous',
    'light_bulb': 'hazardous',
    'paint_bucket': 'hazardous',
    
    # RECYCLABLE - Clean plastics
    'plastic_bottle': 'recyclable',
    'plastic_bottle_cap': 'recyclable',
    'plastic_cup_lid': 'recyclable',
    
    # TRASH - Non-recyclable plastics and items
    'plastic_bag': 'trash',
    'plastic_box': 'trash',
    'plastic_cultery': 'trash',
    'plastic_cup': 'trash',
    'scrap_plastic': 'trash',
    'snack_bag': 'trash',
    'stick': 'trash',
    'straw': 'trash',
}

def get_category(class_name):
    """Determine if item is recyclable, trash, or hazardous"""
    return CATEGORY_MAPPING.get(class_name, 'trash')  # Default to trash if unknown

def setup_gpio():
    """Initialize GPIO pins"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Setup servo
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    pwm = GPIO.PWM(SERVO_PIN, 50)
    pwm.start(0)
    
    # Setup button with internal pull-up resistor
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    # Setup LED (turn it on to show system is ready)
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.output(LED_PIN, GPIO.HIGH)  # Turn LED ON
    
    return pwm

def set_angle(pwm, duty_cycle):
    """Move servo to position"""
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(0)  # Stop signal to reduce jitter

def draw_live_detections(frame, results, scale_x, scale_y):
    """Draw detection boxes on live camera view"""
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates and scale to full frame
            x1, y1, x2, y2 = box.xyxy[0]
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
            
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            category = get_category(class_name)
            
            # Color based on category
            if category == 'recyclable':
                color = (0, 255, 0)  # Green
            elif category == 'hazardous':
                color = (0, 165, 255)  # Orange
            else:  # trash
                color = (255, 0, 0)  # Red
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{class_name} [{category.upper()}]"
            conf_label = f"{confidence:.2f}"
            
            # Label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-35), (x1 + label_size[0] + 10, y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1+5, y1-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, conf_label, (x1+5, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def analyze_detections(results, model):
    """
    Analyze detection results and return category decision
    Logic: If ANY recyclable detected -> RIGHT
           If hazardous detected -> LEFT (treat as trash for safety)
           Otherwise -> LEFT (trash)
    """
    detected_items = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            category = get_category(class_name)
            
            detected_items.append({
                'name': class_name,
                'category': category,
                'confidence': confidence
            })
            
            print(f"  Detected: {class_name} [{category.upper()}] ({confidence:.2f})")
    
    # Look for recyclable items first
    recyclable_items = [item for item in detected_items if item['category'] == 'recyclable']
    
    if recyclable_items:
        # Found recyclable items - use the highest confidence one
        best_recyclable = max(recyclable_items, key=lambda x: x['confidence'])
        if best_recyclable['confidence'] >= CONFIDENCE_THRESHOLD:
            print(f"✅ Found recyclable: {best_recyclable['name']} ({best_recyclable['confidence']:.2f})")
            return 'recyclable'
    
    # Check for hazardous items (treat as trash for safety)
    hazardous_items = [item for item in detected_items if item['category'] == 'hazardous']
    if hazardous_items:
        best_hazardous = max(hazardous_items, key=lambda x: x['confidence'])
        print(f"⚠️ Hazardous item detected: {best_hazardous['name']} - going to TRASH")
        return 'trash'
    
    # Check for trash items
    trash_items = [item for item in detected_items if item['category'] == 'trash']
    if trash_items:
        best_trash = max(trash_items, key=lambda x: x['confidence'])
        print(f"🗑️ Trash detected: {best_trash['name']}")
        return 'trash'
    
    # Nothing clear detected - default to trash (safer)
    if detected_items:
        print(f"⚠️ Unclear detection - defaulting to TRASH")
    else:
        print(f"⚠️ Nothing detected - defaulting to TRASH")
    
    return 'trash'

def main():
    print("=" * 60)
    print("🗑️ SMART WASTE SORTING BIN - IMPROVED MODEL")
    print("=" * 60)
    
    # Setup hardware
    print("\n1. Setting up GPIO...")
    pwm = setup_gpio()
    
    # Load improved model
    print("2. Loading improved AI model...")
    try:
        model = YOLO('improved_model.pt')  # Your new model file
        print(f"   ✅ Model loaded ({len(model.names)} classes)")
        print(f"   Classes: {list(model.names.values())}")
    except:
        print("   ❌ Error: 'improved_model.pt' not found!")
        print("   Make sure the model file is in the project directory")
        return
    
    # Initialize camera
    print("3. Initializing camera...")
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(
        main={"size": (1280, 960), "format": "RGB888"}
    )
    picam2.configure(camera_config)
    picam2.start()
    time.sleep(2)
    print("   ✅ Camera ready")
    
    # Move to center position
    print("4. Moving lid to idle position...")
    set_angle(pwm, POSITION_CENTER)
    
    print("\n" + "=" * 60)
    print("✅ SYSTEM READY!")
    print("=" * 60)
    print("\nInstructions:")
    print("  - LIVE CAMERA VIEW shows what AI sees")
    print("  - Green boxes = Recyclable items")
    print("  - Red boxes = Trash items")
    print("  - Orange boxes = Hazardous items (go to trash)")
    print("  - Place item on lid, then press button to sort")
    print("\nLogic: IF recyclable detected → RIGHT, ELSE → LEFT (trash)")
    print("\nPress 'q' in camera window to exit\n")
    
    # For button debouncing
    button_was_pressed = False
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            
            # Calculate scale factors
            scale_x = frame.shape[1] / DETECTION_WIDTH
            scale_y = frame.shape[0] / DETECTION_HEIGHT
            
            # Downsample for detection
            frame_small = cv2.resize(frame, (DETECTION_WIDTH, DETECTION_HEIGHT))
            
            # Run detection continuously
            results = model(frame_small, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            # Draw detections on frame
            display_frame = draw_live_detections(frame.copy(), results, scale_x, scale_y)
            
            # Add status text
            cv2.putText(display_frame, "LIVE VIEW - Waiting for button press", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show live camera view
            cv2.imshow('Smart Bin - Improved Model', display_frame)
            
            # Check for button press
            button_pressed = GPIO.input(BUTTON_PIN) == GPIO.LOW
            
            if button_pressed and not button_was_pressed:
                print("\n" + "=" * 60)
                print("🔘 BUTTON PRESSED - Making decision...")
                print("=" * 60)
                
                # Analyze current detections
                category = analyze_detections(results, model)
                
                if category == 'recyclable':
                    print("\n♻️ RECYCLABLE DETECTED")
                    print("→ Flipping RIGHT to recyclable bin...")
                    set_angle(pwm, POSITION_RIGHT)
                    time.sleep(2)
                    
                else:  # trash (includes hazardous)
                    print("\n🗑️ TRASH DETECTED")
                    print("→ Flipping LEFT to trash bin...")
                    set_angle(pwm, POSITION_LEFT)
                    time.sleep(2)
                
                # Return to center
                print("→ Returning to idle position...")
                set_angle(pwm, POSITION_CENTER)
                
                print("=" * 60)
                print("✅ Ready for next item!")
                print("=" * 60 + "\n")
            
            button_was_pressed = button_pressed
            
            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n\n🛑 'q' pressed - Shutting down...")
                break
            
            time.sleep(0.05)  # Small delay for smoother display
    
    except KeyboardInterrupt:
        print("\n\n🛑 Ctrl+C pressed - Shutting down...")
    
    finally:
        # Cleanup
        set_angle(pwm, POSITION_CENTER)
        pwm.stop()
        GPIO.cleanup()
        picam2.stop()
        cv2.destroyAllWindows()
        print("✅ System stopped cleanly\n")

if __name__ == "__main__":
    main()
