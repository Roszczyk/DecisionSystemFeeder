from states import StateMeasured, State
from environment import Environment

from datetime import datetime
from copy import deepcopy

class Sensor:
    def __init__(self, name : str, states_drafts : list[StateMeasured],
                 conditional_probabilities : ConditionalProbabilitiesMatrix = None   # mostly for Bayes inference, optional
                 ):
        self.name = name
        self.states_to_measure = states_drafts
        self.conditional_probabilities = conditional_probabilities

    def get_states_probabilities(self, environment : Environment):
        return self.states_to_measure
    
    def get_measurements_dict(self, environment : Environment):
        return dict({
            "results" : self.get_states_probabilities(environment),
            "timestamp" : datetime.now(),
            "sensor_name" : self.name
        })
    
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
    

# TESTING 

if __name__ == "__main__":
    states = [
        State("X", 0.4), 
        State("Y", 0.6)
    ]
    measured_states = [
        StateMeasured(states, 0, "XvY"),
        StateMeasured([states[0]], 0, "X"),
        StateMeasured([states[1]], 0, "Y")
    ]
    conditional_probs = [
        ConditionalProbability(states[0], measured_states[0], 0.4),
        ConditionalProbability(states[1], measured_states[0], 0.3),
        ConditionalProbability(states[0], measured_states[1], 0.5),
        ConditionalProbability(states[1], measured_states[1], 0.6),
        ConditionalProbability(states[0], measured_states[2], 0.1),
        ConditionalProbability(states[1], measured_states[2], 0.1)
    ]
    matrix = ConditionalProbabilitiesMatrix(states, measured_states, conditional_probs)

    print(matrix.get_value(states[0],measured_states[0]).value)
    print(matrix.get_value_by_friendly_name("X", "XvY").value)

    array_for_helper = [
        [0.4, 0.5, 0.1],
        [0.3, 0.6, 0.1]
    ]
    matrix2 = conditional_probabilities_matrix_helper(array_for_helper, states, measured_states)
    print(matrix2.get_value(states[0],measured_states[0]).value)
    print(matrix2.get_value_by_friendly_name("X", "XvY").value)