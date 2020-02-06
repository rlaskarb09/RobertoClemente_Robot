import time
from adafruit_servokit import ServoKit

kit = ServoKit(channels=16)

def stop():
    kit.continuous_servo[0].throttle = 0.0
    kit.continuous_servo[1].throttle = 0.0

def forward(speed):
    kit.continuous_servo[0].throttle = speed
    kit.continuous_servo[1].throttle = -speed

def backward(speed):
    kit.continuous_servo[0].throttle = -speed
    kit.continuous_servo[1].throttle = speed

def move(leftSpeed, rightSpeed):
    kit.continuous_servo[0].throttle = leftSpeed
    kit.continuous_servo[1].throttle = -rightSpeed

if __name__ == '__main__':
    kit = ServoKit(channels=16)
    stop(kit)
    intervals = []
    for i in range(-402, 1000, 2):
        command = input("iteration " + str(i) + ": Enter anything to start: ")
        start = time.time()
        move(kit, -0.43, i * 0.001)

        command = input("iteration " + str(i) + ": Enter anything to stop: ")
        end = time.time()
        stop(kit)
        intervals.append(end - start)
    
    stop(kit)
    
    for i in range(1, 11):
        print('iteration', i, intervals[i - 1])

