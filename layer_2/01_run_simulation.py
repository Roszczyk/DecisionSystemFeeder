from sensor import ScalarSensor, LabelThreshold
from environment import Environment, Metric
from states import State, StateMeasured

import random

# SETTING UP AN ENVIRONMENT

states = [
    State("A", 0.3),
    State("B", 0.4),
    State("C", 0.2),
    State("D", 0.1)
]

env = Environment(states)   # environment with 4 states, not changing the state

# SCALAR SENSOR TYPE 1 

# Set up metric measured by type 1 sensor
def type_1_measure_metric(state : State):
    if state.name == "A":
        base = 0
        multiplier = 60
    elif state.name == "B":
        base = 20
        multiplier = 40
    elif state.name == "C":
        base = 60
        multiplier = 20
    elif state.name == "D":
        base = 90
        multiplier = 20
    else:
        base = 100
        multiplier = 100
    random_component = random.random()
    value = base + (random_component * 2 - 1) * multiplier
    return value
 
metric_type1 = Metric("type1", type_1_measure_metric)
env.add_metric(metric_type1)

def mock_measure_scalar_1(environment : Environment):
    metric_value = environment.measure_metric("type1")
    return metric_value

# Type 1 sensors measured states and labels thresholds
states_measured = [
    StateMeasured([states[0], states[1]], 0, "AvB"),
    StateMeasured([states[2]], 0, "C"),
    StateMeasured([states[3]], 0, "D")
]
labels = [
    LabelThreshold(states_measured[0], -256, 40),
    LabelThreshold(states_measured[1], 40, 80),
    LabelThreshold(states_measured[2], 80, 256)
]

# Instances of type 1 sensor
sensor_type1_unit1 = ScalarSensor("Type1_001", mock_measure_scalar_1, 0, 5, labels)
sensor_type1_unit2 = ScalarSensor("Type1_002", mock_measure_scalar_1, 0.2, 2, labels)
sensor_type1_unit3 = ScalarSensor("Type1_003", mock_measure_scalar_1, 0.1, 10, labels)
type1_sensors = [sensor_type1_unit1, sensor_type1_unit2, sensor_type1_unit3]

# Getting the measurements from type 1 sensors
probs = [x.get_measurements_dict(env) for x in type1_sensors]

# Print the results
print("State: ", env.get_current_state().name)
for prob in probs:
    print(prob["sensor_name"])
    for val in prob["results"]:
        print(val.friendly_name, val.mass)