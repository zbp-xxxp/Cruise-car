from serial_port import serial_connection

from ctypes import *
import time
import serial
import math
from widgets import Button

comma_head_01_motor = bytes.fromhex('77 68 06 00 02 0C 01 01')
comma_head_02_motor = bytes.fromhex('77 68 06 00 02 0C 01 02')
comma_head_03_motor = bytes.fromhex('77 68 06 00 02 0C 01 03')
comma_head_04_motor = bytes.fromhex('77 68 06 00 02 0C 01 04')
comma_trail = bytes.fromhex('0A')

class Cart:
    def __init__(self):
        self.velocity = 25
        self.Kx=0.85
        portx = "/dev/ttyUSB0"
        bps = 115200
        self.serial = serial.Serial(portx, int(bps), timeout=1, parity=serial.PARITY_NONE, stopbits=1)
        # self.serial = serial_connection
        self.p = 0.8;
        self.full_speed = self.velocity
        self.slow_ratio = 0.97;
        self.min_speed = 20

    def steer(self, angle):
        print(angle)
        speed = int(self.velocity);
        if abs(angle) > 0.12:
            speed = int(self.velocity * self.slow_ratio);
        # angle = angle * 0.9;
        angle = angle * self.Kx
        delta = angle - 0
        

        leftwheel = speed+1
        rightwheel = speed-2
        
        scale = 1;
        if (delta < 0):
            leftwheel = int((1 + delta * scale) * speed);
        if (delta > 0):
            rightwheel = int((1 - delta * scale) * speed);
        # leftwheel_back=int(leftwheel*1.1)
        # rightwheel_back=int(rightwheel*1.1)
        self.move([leftwheel, rightwheel, leftwheel, rightwheel])


    def stop(self):
        self.move([0, 0, 0, 0])

    def exchange(self, speed):
        if speed > 100:
            speed = 100
        elif speed < -100:
            speed = -100
        else:
            speed = speed
        return speed

    def move(self, speeds):
        left_front = int(speeds[0])
        right_front = -int(speeds[1])
        left_rear = int(speeds[2])
        right_rear = -int(speeds[3])
        self.min_speed = int(min(speeds))
        # print(speeds)
        left_front=self.exchange(left_front)
        right_front = self.exchange(right_front)
        left_rear=self.exchange(left_rear)
        right_rear = self.exchange(right_rear)
        send_data_01_motor = comma_head_01_motor + left_front.to_bytes(1, byteorder='big', signed=True) + comma_trail
        send_data_02_motor = comma_head_02_motor + right_front.to_bytes(1, byteorder='big', signed=True) + comma_trail
        send_data_03_motor = comma_head_03_motor + left_rear.to_bytes(1, byteorder='big', signed=True) + comma_trail
        send_data_04_motor = comma_head_04_motor + right_rear.to_bytes(1, byteorder='big', signed=True) + comma_trail

        self.serial.write(send_data_01_motor)
        self.serial.write(send_data_02_motor)
        self.serial.write(send_data_03_motor)
        self.serial.write(send_data_04_motor)
        # self.serial.flush()

    def turn_left(self):
        speed = self.velocity 
        leftwheel = -speed;
        rightwheel = speed;
        self.move([leftwheel, rightwheel, leftwheel, rightwheel])

    def turn_right(self):
        speed = self.velocity 
        leftwheel = speed;
        rightwheel = -speed;

        self.move([leftwheel, rightwheel, leftwheel, rightwheel])
        print("L:{} R:{}".format(leftwheel, rightwheel))

    def reverse(self):
        speed = self.velocity 
        self.move([-speed,-speed,-speed,-speed])
        

def test():
    c = Cart();
    while True:
        c.move([50,0,50,0])
        time.sleep(4);
        c.stop();
        time.sleep(1);
    # c.move([-20,-20,-20,-20])
    # time.sleep(2);
    # c.stop();
    # c.steer(0);
    # time.sleep(10);
    # c.steer(0.2);
    # time.sleep(10);
    # c.steer(0.5);
    # time.sleep(1);
    # c.stop();
if __name__ == "__main__":
    c = Cart()
    #程序开启运行开关
    start_button = Button(1, "UP")
    #程序关闭开关
    stop_button = Button(1, "RIGHT")
    while True:
        if start_button.clicked():
            c.move([20,20,20,20])
        else:
        # time.sleep(2)
            c.move([0, 0, 0, 0])

    # test();
    # # testmove()
    # # test();
    # #     turntest()
    # c = Cart()
    # while True:
    #     # c.move([5,5,5,5])
    #     c.move([10, 10, 10, 10])
    #     time.sleep(3)
    #     c.move([0, 0, 0, 0])
    #     time.sleep(3)
