import serial
import time

from threading import Lock

class Serial:
	def __init__(self):
		portx = "/dev/ttyUSB0"
		bps = 115200
		self.res = None
		self.serial = serial.Serial(portx, int(bps), timeout=1, parity=serial.PARITY_NONE, stopbits=1)
		time.sleep(1)

	def write(self, data):
		lock = Lock()
		lock.acquire()
		try:
			self.serial.write(data)
			self.serial.flush()
			self.res = self.serial.readline()
		finally:
			lock.release()

	def read(self):
		return self.res

serial_connection = Serial()
serial_connection.write(b'77 68 06 00 02 0C 01 01 50 0A')
print("finish!")
