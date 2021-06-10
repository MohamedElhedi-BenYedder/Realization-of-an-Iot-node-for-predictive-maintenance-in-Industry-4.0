from typing import overload
class Sensors:
    
    def __init__(self,ax: float,ay: float,az: float,temp: float,hygro: float,tempIndex: float):
        self.ax = ax
        self.ay = ay
        self.az = az
        self.temperature = temp
        self.hygrometry  = hygro
        self.temperatureIsndex = tempIndex
    
    def fromString(string :str):
        values = str(string).split(",");
        ax = float(values[0])
        ay = float(values[1])
        az = float(values[2])
        temperature = float(values[3])
        hygrometry  = float(values[4])
        temperatureIndex = float(values[5])
        return Sensors(ax,ay,az,temperature,hygrometry,temperatureIndex)
