from src.states import StateMeasured, State
from src.environment import Environment

from datetime import datetime
from copy import deepcopy

class Sensor:
    def __init__(self, name : str, states_drafts : list[StateMeasured],
                 conditional_probabilities : ConditionalProbabilitiesMatrix = None   # mostly for Bayes inference, optional
                 ):
        self.name = name
        self.states_to_measure = states_drafts
        self.conditional_probabilities = conditional_probabilities

    def get_states_probabilities(self, environment : Environment) -> list[StateMeasured]:
        # this function should be overwritten by a subclass
        return self.states_to_measure
    
    def get_measurements_dict(self, environment : Environment) -> SensorOutputDict:
        return SensorOutputDict(
            results = self.get_states_probabilities(environment),
            sensor = self
        )

class SensorOutputDict:
    def __init__(self, results : list[StateMeasured], sensor : Sensor, timestamp : datetime = None):
        self.results = results
        self.sensor = sensor
        self.timestamp = timestamp if timestamp is not None else datetime.now()

### UTILS FOR BAYES INFERENCE - Conditional Probabilities

class ConditionalProbability:
    def __init__(self, real_state : State, state_measured : StateMeasured, value : float):
        self.friendly_name = f"P({state_measured.friendly_name}|{real_state.name})"
        self.real = real_state
        self.measured = state_measured
        assert value <= 1 and value >= 0, f"0 <= value ({value}) <= 1 not fulfilled"
        self.value = value

class ConditionalProbabilitiesMatrix:
    def __init__(self, real_states : list[State], measured_states : list[StateMeasured],
                    conditional_probabilities : list[ConditionalProbability]):
        self.matrix = dict()
        for state in real_states:
            self.matrix[state] = dict()
        for prob in conditional_probabilities:
            self.matrix[prob.real][prob.measured] = prob
        # validate:
        for real_s in real_states:
            validating_sum = 0
            for meas_s in measured_states:
                assert meas_s in list(self.matrix[real_s].keys()), f"P({meas_s.friendly_name}|{real_s.name}) is not given"
                validating_sum = validating_sum + self.matrix[real_s][meas_s].value
            validating_sum = round(validating_sum, 3)
            assert validating_sum == 1, \
                    f"Sum for conditional probabilities for state {real_s.name} does not sum up to 1 (={validating_sum})"
            
    def get_value(self, real_state : State, measured_state : StateMeasured):
        return self.matrix[real_state][measured_state]
    
    def get_value_by_friendly_name(self, real_state_name : str, meas_state_name : str):
        real_state = None
        meas_state = None
        for state in self.matrix.keys():
            if state.name == real_state_name:
                real_state = state
                break
        assert real_state != None, f"State {real_state_name} wasn't found"
        for state in self.matrix[real_state]:
            if state.friendly_name == meas_state_name:
                meas_state = state
                break
        assert meas_state != None
        return self.matrix[real_state][meas_state]
    
def conditional_probabilities_matrix_helper(probs_as_list : list[list[float]], real_states : list[State],
                                            measured_states : list[StateMeasured]):
    temp_list = []
    assert len(probs_as_list) == len(real_states), \
        f"matrix rows ({len(probs_as_list)}) and number of states ({len(real_states)}) are not equal"
    for row_no in range(len(probs_as_list)):
        assert len(probs_as_list[row_no]) == len(measured_states),\
            f"columns ({len(probs_as_list[row_no])}) and number of meas states ({len(measured_states)}) are not equal for row {row_no}"
        for column_no in range(len(probs_as_list[row_no])):
            temp_list.append(ConditionalProbability(real_states[row_no], measured_states[column_no], probs_as_list[row_no][column_no]))
    return ConditionalProbabilitiesMatrix(real_states, measured_states, temp_list)
    

################################
####### SCALAR SENSORS   #######
################################
# Scalar sensors are sensors that measure physical value of the environment
# e.g. temperature, humidity, vibrations, pressure etc.

from scipy.stats import norm

class LabelThreshold:
    def __init__(self, state_represented : StateMeasured, lower_bound : float, upper_bound : float):
        self.name = state_represented.friendly_name
        self.state = deepcopy(state_represented)
        self.lower = lower_bound
        self.upper = upper_bound


class ScalarSensor(Sensor):
    def __init__(self, name : str, measure_function : function, uncertainty_relative : float, 
                 uncertainty_absolute : float, labels_thresholds : list[LabelThreshold],
                 conditional_probabilities : ConditionalProbabilitiesMatrix = None ):
        super().__init__(name, [x.state for x in labels_thresholds], 
                         conditional_probabilities=conditional_probabilities)
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
            result.append(deepcopy(label.state))
        return result
    
################################
####### COMPUTER VISION  #######
################################
# Computer Vision sensors use CV models to detect objects

class BoundingBox:
    def __init__(self, x1, y1, x2, y2, confidence, class_id):
        self.coords1 = [x1,y1]
        self.coords2 = [x2,y2]
        self.confidence = confidence
        self.cls = class_id

    def calculate_area(self):
        x_vec = abs(self.coords1[0] - self.coords2[0])
        y_vec = abs(self.coords1[1] - self.coords2[1])
        return x_vec * y_vec

class ComputerVision(Sensor):
    def __init__(self, name : str, acquire_bounding_boxes : function[[Environment], list[BoundingBox]], 
                 analyze_bounding_boxes : function[[list[BoundingBox]], list[StateMeasured]], 
                 states : list[StateMeasured]):
        super().__init__(name, states)
        # measurement
        self.acquire_bb = acquire_bounding_boxes
        self.analyze_bb = analyze_bounding_boxes

    def get_bounding_boxes(self, environment : Environment) -> list[BoundingBox]:
        return self.acquire_bb(environment)

    def analyze_bounding_boxes(self, bounding_boxes : list[BoundingBox]) -> list[StateMeasured]:
        return self.analyze_bb(bounding_boxes)

    def get_states_probabilities(self, environment : Environment) -> list[StateMeasured]:
        return self.analyze_bounding_boxes(self.get_bounding_boxes(environment))
