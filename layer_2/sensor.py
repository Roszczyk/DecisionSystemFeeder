from datetime import datetime

class Sensor:
    def __init__(self, name : str, precision : float, measure_function : function):
        self.name = name
        self.precision = precision
        self.measure_function = measure_function

    def measure(self):
        return self.measure_function()
    
    def get_measurement_value(self):
        return {
            "timestamp" : datetime.now(),
            "sensor" : self.name,
            "value" : self.measure()
        }