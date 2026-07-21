from states import StateMeasured, State
from environment import Environment

from datetime import datetime

class Sensor:
    def __init__(self, name : str, states_drafts : list[StateMeasured]):
        self.name = name
        self.states_to_measure = states_drafts

    def get_states_probabilities(self, environment : Environment):
        return self.states_to_measure
    
    def get_measurements_dict(self, environment : Environment):
        return dict({
            "results" : self.get_states_probabilities(environment),
            "timestamp" : datetime.now(),
            "sensor_name" : self.name
        })


################################
####### SCALAR SENSORS   #######
################################
# Scalar sensors are sensors that measure physical value of the environment
# e.g. temperature, humidity, vibrations, pressure etc.

from scipy.stats import norm

class LabelThreshold:
    def __init__(self, state_represented : StateMeasured, lower_bound : float, upper_bound : float):
        self.name = state_represented.friendly_name
        self.state = state_represented
        self.lower = lower_bound
        self.upper = upper_bound


class ScalarSensor(Sensor):
    def __init__(self, name : str, measure_function : function, uncertainty_relative : float, 
                 uncertainty_absolute : float, labels_thresholds : list[LabelThreshold]):
        super().__init__(name, [x.state for x in labels_thresholds])
        # measurement
        self.measure_function = measure_function
        # uncertainty calculation
        self.uncertainty_relative = uncertainty_relative
        self.uncertainty_absolute = uncertainty_absolute
        # probabilities assignment
        self.labels_thresholds = labels_thresholds

    def calculate_uncertainty(self, measured_value):
        return measured_value * self.uncertainty_relative + self.uncertainty_absolute
    
    def gauss_probability(self, value, uncertainty, label_threshold : LabelThreshold):
        # Z = N(value, uncertainty^2)
        p = norm.cdf(label_threshold.upper, loc=value, scale=uncertainty) \
            - norm.cdf(label_threshold.lower, loc=value, scale=uncertainty)
        return float(p)
    
    def measure(self, environment : Environment):
        return self.measure_function(environment)
    
    def get_states_probabilities(self, environment : Environment):
        measurement = self.measure(environment)
        uncertainty = self.calculate_uncertainty(measurement)
        result = []
        for label in self.labels_thresholds:
            probability = self.gauss_probability(measurement, uncertainty, label)
            label.state.mass = probability
            result.append(label.state)
        return result