import requests  as http
def toDict(data):
    if not(isinstance(data,dict)):
               data = data.__dict__
    return data
"""
        EngineCycleController
"""
class EngineCycle:
    __baseUrl = "http://192.168.1.15:8080/api/engineCycle/"
    def getBaseUrl():
        return __baseUrl
    def getAll():
        return http.get(EngineCycle.__baseUrl)
    def create(data):
        return http.post(EngineCycle.__baseUrl+"create",json=toDict(data))
    def update(data):
        return http.post(EngineCycle.__baseUrl+"update",json=toDict(data))
    def delete(data):
        return http.delete(EngineCycle.__baseUrl+"delete",json=toDict(data))
"""
        EngineController
"""
class Engine:
    __baseUrl = "http://192.168.1.15:8080/api/engine/"
    def getBaseUrl():
        return __baseUrl
    def getAll():
        return http.get(Engine.__baseUrl+"getAll")
    def create(data):
        return http.post(Engine.__baseUrl+"create",json=toDict(data))
    def update(data):
        return http.post(Engine.__baseUrl+"update",json=toDict(data))
    def delete(data):
        return http.delete(Engine.__baseUrl+"delete",json=toDict(data))
    

    