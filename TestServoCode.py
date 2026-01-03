#!/usr/bin/env python3
"""
SG90 Servo Test Script for Raspberry Pi 5
Tests servo rotation back and forth
"""

import RPi.GPIO as GPIO
import time

# Configuration
SERVO_PIN = 18  # GPIO 18 (Physical Pin 12)

# Servo positions (duty cycle values)
# SG90: 2.5% = 0°, 7.5% = 90°, 12.5% = 180°
POSITION_0 = 2.5      # 0 degrees
POSITION_90 = 7.5     # 90 degrees (middle)
POSITION_180 = 12.5   # 180 degrees

def setup_servo():
    """Initialize GPIO and PWM"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(SERVO_PIN, GPIO.OUT)
    
    # 50Hz PWM signal (standard for servos)
    pwm = GPIO.PWM(SERVO_PIN, 50)
    pwm.start(0)
    
    return pwm

def set_angle(pwm, duty_cycle):
    """Set servo to specific angle"""
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)  # Wait for servo to reach position
    pwm.ChangeDutyCycle(0)  # Stop signal to prevent jitter

def main():
    print("=" * 50)
    print("SG90 Servo Test - Raspberry Pi 5")
    print("=" * 50)
    print("\nWiring Check:")
    print("  Brown wire  → Pin 6  (Ground)")
    print("  Red wire    → Pin 2  (5V)")
    print("  Yellow wire → Pin 12 (GPIO 18)")
    print("\nStarting test in 3 seconds...")
    print("Press Ctrl+C to stop\n")
    
    time.sleep(3)
    
    pwm = setup_servo()
    
    try:
        # Move to starting position
        print("Moving to 0° position...")
        set_angle(pwm, POSITION_0)
        time.sleep(1)
        
        cycle = 1
        while True:
            print(f"\n--- Cycle {cycle} ---")
            
            # Move to 0 degrees
            print("→ Moving to 0° (far left)")
            set_angle(pwm, POSITION_0)
            time.sleep(1)
            
            # Move to 90 degrees
            print("→ Moving to 90° (center)")
            set_angle(pwm, POSITION_90)
            time.sleep(1)
            
            # Move to 180 degrees
            print("→ Moving to 180° (far right)")
            set_angle(pwm, POSITION_180)
            time.sleep(1)
            
            # Back to center
            print("→ Moving to 90° (center)")
            set_angle(pwm, POSITION_90)
            time.sleep(1)
            
            cycle += 1
    
    except KeyboardInterrupt:
        print("\n\n⏹  Test stopped by user")
    
    finally:
        # Return to center position
        print("Returning to center position...")
        set_angle(pwm, POSITION_90)
        time.sleep(0.5)
        
        # Cleanup
        pwm.stop()
        GPIO.cleanup()
        print("✅ GPIO cleaned up")
        print("Test complete!\n")

if __name__ == "__main__":
    main()