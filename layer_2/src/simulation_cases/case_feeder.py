from src.states import State, StateMeasured
from src.environment import Environment, Metric
from src.sensor import ScalarSensor, SimpleSensor, ComputerVision, \
    LabelThreshold, conditional_probabilities_matrix_helper
from src.fusion import FusionLibrary
from src.simulation import Simulation

import random
from copy import deepcopy

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

    simplest_sensor_states_measured_possible = [
        StateMeasured([states[0], states[1]], friendly_name="movement_detected"),
        StateMeasured([states[2]], friendly_name="idle")
    ]

    def movement_func(env : Environment, correctness_probability : float) -> StateMeasured:
        is_moving = env.measure_metric("movement")
        measured_moving = is_moving and random.random() <= correctness_probability
        return simplest_sensor_states_measured_possible[0] if measured_moving else simplest_sensor_states_measured_possible[1]

    movement_sensors = [
        SimpleSensor("01_movement_sensor_good", simplest_sensor_states_measured_possible, lambda e: movement_func(e, 0.95), 
                     0.95, None),
        SimpleSensor("02_movement_sensor", simplest_sensor_states_measured_possible, lambda e: movement_func(e, 0.85), 
                     0.85, None),
        SimpleSensor("03_movement_sensor_poor", simplest_sensor_states_measured_possible, lambda e: movement_func(e, 0.7),
                     0.7, None)     # TODO To run Bayes, I need to prepare conditional_probabilities
    ]
    all_sensors += movement_sensors

    # vibration sensors (ScalarSensor)
    def vibrations_metric_set(state : State, previous_value : float):      # values 0-20   
        if previous_value > 2.0 and state.name == "idle":
            random_diff = random.random() * 10
            value = max(0.0, previous_value - random_diff)
        elif previous_value <= 2.0 and state.name == "idle":
            random_diff = random.random() * 2 - 1.0
            value = previous_value + random_diff
        elif "detected" in state.name:
            random_diff = random.random() * 20
            value = min(previous_value + random_diff, 20.0)
        return value
    env.add_metric(Metric("vibrations", vibrations_metric_set, 1.0))

    def vibrations_sensor_func(environment : Environment):
        vibrations_val = environment.measure_metric("vibrations")
        return vibrations_val

    import math
    vibration_labels = [
        LabelThreshold(simplest_sensor_states_measured_possible[0], 7.0, math.inf),
        LabelThreshold(simplest_sensor_states_measured_possible[1], -math.inf, 7.0)
    ]
    vibrations_sensors = [
        ScalarSensor("04_vibrations_good", vibrations_sensor_func, 0.05, 0.1, vibration_labels),
        ScalarSensor("05_vibrations_poor", vibrations_sensor_func, 0.2, 0.5, vibration_labels)
    ]
    all_sensors += vibrations_sensors

    # Camera sensors (ComputerVision)
    cv_states_possible = [
        StateMeasured([states[0]], friendly_name="bird"),
        StateMeasured([states[1], states[2]], friendly_name="no bird")
    ]
    from src.sensor import BoundingBox
    def get_bounding_boxes(environment : Environment, frequency_of_mistakes : float, detection_threshold : float = 0.7):
        real_current_state = environment.get_current_state()
        output_bb = []
        if ("bird" in real_current_state.name and random.random() > frequency_of_mistakes) or \
            ("bird" not in real_current_state.name and random.random() < frequency_of_mistakes):
            if random.random() > 0.8:
                number_of_birds = 2
            else:
                number_of_birds = 1
        else:
            number_of_birds = 0
        # detect real birds
        for i in range(number_of_birds):
            x1 = random.random() / 2
            x2 = x1 + (random.random() / 2)
            y1 = random.random() / 2
            y2 = y1 + (random.random() / 2)
            confidence = random.random() * (1.0 - detection_threshold) + detection_threshold
            output_bb.append(BoundingBox(x1, y1, x2, y2, confidence, "bird"))
        # fake detections
        number_of_fake = round(random.random()*3)
        for i in range(number_of_fake):
            x1 = random.random() / 2
            x2 = x1 + (random.random() / 2)
            y1 = random.random() / 2
            y2 = y1 + (random.random() / 2)
            confidence = random.random() * detection_threshold
            output_bb.append(BoundingBox(x1, y1, x2, y2, confidence, "bird"))
        return output_bb
    def analyse_bounding_boxes(bb : list[BoundingBox]):
        detected_birds = []
        for box in bb:
            if box.cls == "bird" and box.confidence >= 0.7:
                detected_birds.append(box)
        bird_detected_state = deepcopy(cv_states_possible[0])
        bird_not_detected_state = deepcopy(cv_states_possible[1])
        confidence_values = [x.confidence for x in detected_birds]
        confidence_values.sort()
        mass = confidence_values[0] if len(confidence_values) > 0 else 0.0
        if len(confidence_values) > 1:
            for value in confidence_values[1:]:
                mass += (1.0 - mass) * value
        bird_detected_state.mass = mass
        bird_not_detected_state.mass = 1.0 - mass
        return [bird_detected_state, bird_not_detected_state]

    cv_sensors = [
        ComputerVision("06_cv_rgb", lambda env: get_bounding_boxes(env, 0.05), 
                       analyse_bounding_boxes, cv_states_possible),
        ComputerVision("07_cv_ir", lambda env: get_bounding_boxes(env, 0.3, 0.6), 
                       analyse_bounding_boxes, cv_states_possible)
    ]
    all_sensors += cv_sensors

    # Run simulation
    output = dict()

    for method in fusion_methods:
        print("### METHOD:", method)
        simulation = Simulation(method, library[method], env, all_sensors)
        sim_output = simulation.run_accuracy(iterations=iterations, return_all_logs=return_all_logs)
        results = sim_output["results"]

        output[method] = deepcopy(sim_output)

        total_cases = results["TOTAL"]["total cases"]
        output = deepcopy(sim_output)
        if verbose:
            print("Cases: ", total_cases)
            print("TOTAL RESULTS:")
            del results["TOTAL"]["total cases"]
            for key in results["TOTAL"].keys():
                print(f"{key:<20} {results['TOTAL'][key] * 100:>6.2f}%")
            print("STATE SPECIFIC")
            del results["TOTAL"]
            for key in results.keys():
                print(f"case {key} == {results[key]}")
    return output