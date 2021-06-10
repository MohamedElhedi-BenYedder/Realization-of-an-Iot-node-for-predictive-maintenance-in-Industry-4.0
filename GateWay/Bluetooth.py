import time
import serial
class Bluetooth:
    
    
    
    def __init__(self,port=1,baudrate=9600,parity=serial.PARITY_ODD,stopbits=serial.STOPBITS_TWO,bytesize=serial.SEVENBITS):
        self.__port='/dev/rfcomm'+str(port)
        self.__baudrate=baudrate
        self.__parity=parity
        self.__stopbits=stopbits
        self.__bytesize=bytesize
        self.__serial = serial.Serial(port=self.__port,baudrate=self.__baudrate,parity=self.__parity,stopbits=self.__stopbits,bytesize=self.__bytesize)
    def getSerial(self):
        return self.__serial
    def readResponse(self,delay=0):
        if(delay!=0):
            tic = time.time()
            while time.time() - tic < delay and bluetooth.inWaiting() == 0: 
                time.sleep(1)
        if self.__serial.inWaiting() >= 0:
            return self.__serial.readline().decode()
    def sendMessage(self,message):
        self.__serial.write(message.encode())
    
        

    

