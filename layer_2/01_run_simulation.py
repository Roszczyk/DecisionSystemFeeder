from src.sensor import ScalarSensor, LabelThreshold, conditional_probabilities_matrix_helper
from src.environment import Environment, Metric
from src.states import State, StateMeasured
from src.fusion import FusionLibrary
from src.simulation import Simulation

import random
from copy import deepcopy

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
def type_1_measure_metric(state : State, previous_value : float) -> float:
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

conditional_probabilites_matrix = conditional_probabilities_matrix_helper([
    [0.8, 0.1, 0.1],
    [0.6, 0.3, 0.1],
    [0.05, 0.9, 0.05],
    [0.1, 0.2, 0.7]
], states, states_measured)

# Instances of type 1 sensor
sensor_type1_unit1 = ScalarSensor("Type1_001", mock_measure_scalar_1, 0, 5, labels, conditional_probabilites_matrix)
sensor_type1_unit2 = ScalarSensor("Type1_002", mock_measure_scalar_1, 0.2, 20, labels, conditional_probabilites_matrix)
sensor_type1_unit3 = ScalarSensor("Type1_003", mock_measure_scalar_1, 0.1, 10, labels, conditional_probabilites_matrix)
type1_sensors = [sensor_type1_unit1, sensor_type1_unit2, sensor_type1_unit3]

# Getting the measurements from type 1 sensors
probs = [x.get_measurements_dict(env) for x in type1_sensors]
    
# Print the results
print("State: ", env.get_current_state().name)
for prob in probs:
    print(prob.sensor.name)
    for val in prob.results:
        print(val.friendly_name, "\t", val.mass)

fusion_library = FusionLibrary().methods
for method in fusion_library.keys():
    print(f"=== METHOD = {method:<20} ===")
    simulation = Simulation(method, fusion_library[method], env, type1_sensors)
    results = simulation.run_accuracy()
    total_cases = results["TOTAL"]["total cases"]
    results_copy = deepcopy(results)        # I don't know yet but maybe it will be necessary to save to file or sth
    print("Cases: ", total_cases)
    print("TOTAL RESULTS:")
    del results["TOTAL"]["total cases"]
    for key in results["TOTAL"].keys():
        print(f"{key:<20} {results['TOTAL'][key] * 100:>6.2f}%")
    print("STATE SPECIFIC")
    del results["TOTAL"]
    for key in results.keys():
        print(f"case {key} == {results[key]}")

    