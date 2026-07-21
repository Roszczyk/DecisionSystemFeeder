from sensor import ScalarSensor, LabelThreshold
from environment import Environment
from states import State, StateMeasured

import random

states = [
    State("A", 0.3),
    State("B", 0.4),
    State("C", 0.2),
    State("D", 0.1)
]

env = Environment(states, 25)   # environment with 4 states, changing state every 25 seconds

# SCALAR SENSOR TYPE 1 

def mock_measure_scalar_1(environment : Environment):
    random_val = random.random()
    return 100*random_val

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

sensor_type1_unit1 = ScalarSensor("Type1_001", mock_measure_scalar_1, 0, 5, labels)
sensor_type1_unit2 = ScalarSensor("Type1_002", mock_measure_scalar_1, 0.2, 2, labels)
sensor_type1_unit3 = ScalarSensor("Type1_003", mock_measure_scalar_1, 0.1, 10, labels)

probs = [
    sensor_type1_unit1.get_measurements_dict(env),
    sensor_type1_unit2.get_measurements_dict(env),
    sensor_type1_unit3.get_measurements_dict(env)
]

for prob in probs:
    print(prob["sensor_name"])
    for val in prob["results"]:
        print(val.friendly_name, val.mass)