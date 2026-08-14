from src.states import State, StateMeasured
from src.environment import Environment, Metric
from src.sensor import ScalarSensor, SimpleSensor, ComputerVision, \
    LabelThreshold, conditional_probabilities_matrix_helper
from src.fusion import FusionLibrary
from src.simulation import Simulation

import random

def main_feeder_simulation(fusion_methods : list[str] = None, iterations : int = 1000, return_all_logs : bool = False, verbose : bool = True):
    # validate the methods
    library = FusionLibrary().methods
    if fusion_methods is None:
        fusion_methods = library.keys()
    else:
        for m in fusion_methods:
            assert m in library, f"{m} is not in the library"

    # set up the environment
    bird_detected_prob = 0.2
    other_animal_prob = 0.1
    states = [
        State("bird_detected", bird_detected_prob),
        State("other_animal_detected", other_animal_prob),
        State("idle", 1.0 - bird_detected_prob - other_animal_prob)
    ]
    env = Environment(states)

    # initialize the sensors list
    all_sensors = []

    # movement sensors
    movement_metric_func = lambda s,p: ("detected" in s.name)
    env.add_metric(Metric("movement", movement_metric_func, False))

    movement_sensor_states_measured_possible = [
        StateMeasured([states[0], states[1]], friendly_name="movement_detected"),
        StateMeasured([states[2]], friendly_name="idle")
    ]

    def movement_func(env : Environment, correctness_probability : float) -> StateMeasured:
        is_moving = env.measure_metric("movement")
        measured_moving = is_moving and random.random() <= correctness_probability
        return movement_sensor_states_measured_possible[0] if measured_moving else movement_sensor_states_measured_possible[1]

    movement_sensors = [
        SimpleSensor("01_movement_sensor_good", movement_sensor_states_measured_possible, lambda e: movement_func(e, 0.95), 
                     0.95, None),
        SimpleSensor("02_movement_sensor", movement_sensor_states_measured_possible, lambda e: movement_func(e, 0.85), 
                     0.85, None),
        SimpleSensor("03_movement_sensor_poor", movement_sensor_states_measured_possible, lambda e: movement_func(e, 0,7),
                     0.7, None)     # TODO To run Bayes, I need to prepare conditional_probabilities
    ]
    all_sensors += movement_sensors

    # TODO Vibration sensors (ScalarSensor)
    # TODO Climate sensors (ScalarSensor)
    # TODO Camera sensors (ComputerVision)

    # TODO Run simulation