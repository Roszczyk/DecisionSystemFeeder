from sensor import Sensor
from scalar_sensor_logic import ScalarSensor


class TruthTable:
    def __init__(self, true_positive, true_negative, false_positive, false_negative):
        self.true_positive = true_positive
        self.true_negative = true_negative
        self.false_positive = false_positive
        self.false_negative = false_negative


class BinaryClassifier:
    def __init__(self, sensor : ScalarSensor, class_true : str, truth_table : TruthTable):
        # SCALAR SENSOR TO BE DEVELOPED TO GENERAL SENSOR (TODO)
        self.sensor = sensor
        self.class_true = class_true
        self.truth_table = truth_table

    def get_boolean_probability(self):
        results = self.sensor.get_classes_probability() 
        # get_classes_probability() is scalar sensor's function. to be generalized.
        class_true_probability = results[self.class_true]
        return class_true_probability


#                                       #
#       =   TESTING SECTION     =       #
#                                       #

def dummy_run():
    import random
    from scalar_sensor_logic import LabelThreshold

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

    class_true = "optionC"

    binary_classifier = BinaryClassifier(sensor, class_true, None)

    print(binary_classifier.get_boolean_probability() )

if __name__ == "__main__":
    dummy_run()