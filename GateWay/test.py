import requests 
import Controller
"""
import Bodies 

obj={
    "engineCycle": {
        "id": {
            "engine": {
                "id": 10
            },
            "maintenanceIndex": 0,
            "cycle": 0
        }
    },
    "measurements": [
        {
            "ax": 5,
            "ay": 11.0,
            "az": 11.0,
            "temperature": 11.0,
            "hygrometry": 11.0,
            "temperatureIndex": 11.0
        }
    ]
}
r = requests.post("http://192.168.1.15:8080/api/engineCycle/addMeasurement", json=obj)
"""
print(Controller.EngineCycle.getAll().json())