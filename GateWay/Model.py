class Engine:
    def __init__(self,eid: int,description:str=None ,maintenanceIndex:int=None):
        self.id=eid
        if(description!=None):
            self.description=description
        if(maintenanceIndex!=None):
            self.maintenanceIndex=maintenanceIndex

class EngineCycle:
    class CompositeId:
        def __init__(self,engine :Engine,cycle: int,maintenanceIndex :int):
            self.engine=engine
            self.cycle=cycle
            self.maintenanceIndex=maintenanceIndex
    def __init__(self,ecid :CompositeId):
        self.id=ecid
    