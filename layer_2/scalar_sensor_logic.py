from sensor import Sensor

from scipy.stats import norm

class LabelThreshold:
    def __init__(self, label_name : str, lower_bound : float, upper_bound : float):
        self.name = label_name
        self.lower = lower_bound
        self.upper = upper_bound


class ScalarSensor(Sensor):
    def __init__(self, name : str, precision : float, measure_function : function, 
                 uncertainty_logic : function, labels_thresholds : list[LabelThreshold]):
        super().__init__(name, precision, measure_function)
        self.uncertainty_logic = uncertainty_logic
        self.labels_thresholds = labels_thresholds

    def calculate_uncertainty(self, measured_value):
        return self.uncertainty_logic(measured_value)
    
    def gauss_probability(self, value, uncertainty, label_threshold : LabelThreshold):
        # Z = N(value, uncertainty^2)
        p = norm.cdf(label_threshold.upper, loc=value, scale=uncertainty) \
            - norm.cdf(label_threshold.lower, loc=value, scale=uncertainty)
        return float(p)
    
    def get_classes_probability(self):
        measurement = self.measure()
        uncertainty = self.calculate_uncertainty(measurement)
        score = dict()
        for label in self.labels_thresholds:
                score[label.name] =  self.gauss_probability(measurement, uncertainty, label)
        return score
    
#                                       #
#       =   TESTING SECTION     =       #
#                                       #

def dummy_run():
    import random

    def measure_dummy():
        value = random.random() * 100
        print(f"measured value: {value}")
        return value
    
    def uncertainty_func_dummy(value):
        rand = random.random() * 0.2 + 0.5
        unc = value * rand
        unc = min(unc, 15.0)
        print(f"uncertainty: {unc}")
        return unc
    
    labels = [LabelThreshold("optionA", 0, 25), LabelThreshold("optionB", 25, 50),
              LabelThreshold("optionC", 50, 75), LabelThreshold("optionD", 75, 100)]

    sensor = ScalarSensor("dummy", None, measure_dummy, uncertainty_func_dummy, labels)

    print(sensor.get_classes_probability())

if __name__ == "__main__":
    dummy_run()