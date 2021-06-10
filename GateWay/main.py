import time
import serial
import requests 
import json
import Bluetooth as bt
from Controller import *
import util
"""



"""
L=[]
bluetooth = bt.Bluetooth()

bluetooth.getSerial().isOpen()

print('Enter your message below.\r\nInsert "exit" to leave the application.')
delay = 10000
while 1 :
    user_input = input("Server message >> ")
    if user_input == 'exit':
        bluetooth.getSerial().close()
        obj={
    "engineCycle": {
        "id": {
            "engine": {
                "id": 1
            },
            "maintenanceIndex": 0,
            "cycle": 0
        }
    },
    "measurements": L
}
       
        requests.post("http://192.168.1.15:8080/api/engineCycle/addMeasurement",json=obj.__dict__)
        exit()
    else:
        user_input = user_input + '\r'
        bluetooth.sendMessage(user_input)
        recv = bluetooth.readResponse()
        if recv != None:
            print(recv)
            val= util.Sensors.fromString(recv)
            L.append(val.__dict__)
            obj={
    "engineCycle": {
        "id": {
            "engine": {
                "id": 1
            },
            "maintenanceIndex": 0,
            "cycle": 0
        }
    },
    "measurements": L
    }
            print("Response from server ",EngineCycle.update(obj))