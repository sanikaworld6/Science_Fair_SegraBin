#!/usr/bin/env python3
"""
Button + Servo Test
Button on GPIO 17 (Pin 11) with Ground on Pin 9
Servo on GPIO 18 (Pin 12)
Press button → Servo flips right
"""

import RPi.GPIO as GPIO
import time

# Pin Configuration
SERVO_PIN = 18   # GPIO 18 (Physical Pin 12)
BUTTON_PIN = 17  # GPIO 17 (Physical Pin 11)

# Servo positions
POSITION_CENTER = 7.5   # 90 degrees (idle/flat)
POSITION_RIGHT = 12.5   # 180 degrees (flip right)

def setup():
    """Initialize GPIO"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    # Setup servo
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    pwm = GPIO.PWM(SERVO_PIN, 50)
    pwm.start(0)
    
    # Setup button with internal pull-up resistor
    # Button connects GPIO 17 (Pin 11) to Ground (Pin 9)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    
    return pwm

def set_angle(pwm, duty_cycle):
    """Move servo to position"""
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(0)  # Stop signal

def main():
    print("=" * 50)
    print("Button + Servo Test")
    print("=" * 50)
    print("\nWiring:")
    print("  BUTTON:")
    print("    One leg → Pin 11 (GPIO 17)")
    print("    Other leg → Pin 9 (Ground)")
    print("\n  SERVO:")
    print("    Brown  → Pin 6 (Ground)")
    print("    Red    → Pin 2 (5V)")
    print("    Yellow → Pin 12 (GPIO 18)")
    print("\n✅ Ready! Press button to flip servo right")
    print("Press Ctrl+C to exit\n")
    
    pwm = setup()
    
    # Move to center position
    print("Moving to center position...")
    set_angle(pwm, POSITION_CENTER)
    time.sleep(1)
    
    try:
        while True:
            # Check if button is pressed (goes LOW when pressed)
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                print("\n🔘 Button pressed!")
                
                # Flip right
                print("→ Flipping RIGHT (180°)")
                set_angle(pwm, POSITION_RIGHT)
                time.sleep(2)
                
                # Return to center
                print("→ Returning to CENTER (90°)")
                set_angle(pwm, POSITION_CENTER)
                
                print("✅ Ready for next press\n")
                
                # Wait for button release (debounce)
                while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                    time.sleep(0.1)
                time.sleep(0.3)
            
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n⏹  Test stopped")
    
    finally:
        # Return to center
        set_angle(pwm, POSITION_CENTER)
        pwm.stop()
        GPIO.cleanup()
        print("✅ Cleanup complete\n")

if __name__ == "__main__":
    main()