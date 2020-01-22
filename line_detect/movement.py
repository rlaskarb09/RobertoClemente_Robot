from adafruit_servokit import ServoKit
import conf
import numpy as np
import time

import pdb
kit = ServoKit(channels=16)
# left =kit.servo[0]
# right = kit.servo[1]
# left.angle = 0
# right.angle = 180
#
# # setup GPIO pins
# GPIO.setmode(GPIO.BCM)
# GPIO.setwarnings(False)
#
# GPIO.setup(12, GPIO.OUT)
# GPIO.setup(16, GPIO.OUT)
# GPIO.setup(20, GPIO.OUT)
# GPIO.setup(21, GPIO.OUT)

def moveForward(kit):
    kit.continuous_servo[0].throttle = 0.5
    kit.continuous_servo[1].throttle = -0.5

def moveBack(kit):
    kit.continuous_servo[0].throttle = -0.5
    kit.continuous_servo[1].throttle = 0.5

def moveLeft(kit):
    kit.continuous_servo[0].throttle = -0.05
    kit.continuous_servo[1].throttle = -0.4

def moveRight(kit):
    kit.continuous_servo[0].throttle = 0.4
    kit.continuous_servo[1].throttle = 0.05

def moveStop(kit):
    kit.continuous_servo[0].throttle = 0
    kit.continuous_servo[1].throttle = 0

def motorSpeed(a, b):
    kit.continuous_servo[0].throttle = a
    kit.continuous_servo[1].throttle = b



def move(action):
    if action == "LEFT":
        moveLeft(kit)
    elif action == "RIGHT":
        moveRight(kit)
    elif action == "FORWARD":
        moveForward(kit)
    elif action == "BACKWARD":
        moveBack(kit)
    elif action == "STOP":
        moveStop(kit)

def move2(a,b):
    motorSpeed(kit, a, b)

#
# def check_shift_turn(angle, shift):
#     turn_state = 0
#     if angle < conf.turn_angle or angle > 180 - conf.turn_angle:
#         turn_state = np.sign(90 - angle)    #양수 오른쪽
#
#     shift_state = 0
#     if abs(shift) > conf.shift_max:
#         shift_state = np.sign(shift)    # 양수: 오른쪽
#     return turn_state, shift_state
#
# def get_turn(turn_state, shift_state):
#     turn_dir = 0
#     turn_val = 0
#     if shift_state != 0:
#         turn_dir = shift_state
#         turn_val = conf.shift_step if shift_state != turn_state else conf.turn_step
#     elif turn_state != 0:
#         turn_dir = turn_state
#         turn_val = conf.turn_step
#     return turn_dir, turn_val

def turn(turn_dir, turn_val):
    turn_cmd = "RIGHT" if turn_dir > 0 else "LEFT"
    ret_cmd = "FORWARD" if turn_dir > 0 else "FORWARD"
    print(turn_cmd)
    move(turn_cmd)
    time.sleep(turn_val)
    move("STOP")
    # move("FORWARD")
#
# def Action(turn_state, shift_state, time_delay):
#
#     # time delay
#
#     # decide whether to turn
#     turn_dir, turn_val = get_turn(turn_state, shift_state)
#
#     if turn_dir != 0:  # make turn
#         turn(turn_dir, turn_val)
#         # time.sleep(time.delay-turn_val)
#         last_turn = turn_dir
#     else:  # forward
#         print("FORWARD")
#         move("FORWARD")
#         time.sleep(conf.straight_run)
#
#         last_turn = 0
#
#     return last_turn
