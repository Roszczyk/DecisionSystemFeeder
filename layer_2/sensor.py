from datetime import datetime

class TruthTable:
    def __init__(self, true_positive, true_negative, false_positive, false_negative):
        self.true_positive = true_positive
        self.true_negative = true_negative
        self.false_positive = false_positive
        self.false_negative = false_negative

class Sensor:
    def __init__(self, name : str, truth_table : TruthTable, precision : float,
                 measure_function : function):
        self.name = name
        self.truth_table = truth_table
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