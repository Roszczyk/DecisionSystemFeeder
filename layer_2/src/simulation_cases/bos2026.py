from copy import deepcopy
import random
import time

from src.states import State, StateMeasured
from src.environment import Environment, Metric
from src.sensor import ScalarSensor, LabelThreshold, conditional_probabilities_matrix_helper
from src.fusion import FusionLibrary
from src.simulation import Simulation


def get_states():
    return [
        State("A", 0.3),
        State("B", 0.4),
        State("C", 0.2),
        State("D", 0.1),
    ]


def create_environment():
    states = get_states()
    env = Environment(states)

    def measure_metric(state: State, previous_value: float) -> float:
        if state.name == "A":
            base = 20
            spread = 25
        elif state.name == "B":
            base = 42
            spread = 30
        elif state.name == "C":
            base = 65
            spread = 25
        elif state.name == "D":
            base = 85
            spread = 20
        else:
            base = 0
            spread = 100

        return base + (random.random() * 2 - 1) * spread

    metric = Metric("simulation_metric", measure_metric)
    env.add_metric(metric)

    return env


def create_labels(states):
    states_measured = [
        StateMeasured([states[0]], 0, "A"),
        StateMeasured([states[1]], 0, "B"),
        StateMeasured([states[2]], 0, "C"),
        StateMeasured([states[3]], 0, "D"),
    ]

    return [
        LabelThreshold(states_measured[0], -100, 30),
        LabelThreshold(states_measured[1], 30, 55),
        LabelThreshold(states_measured[2], 55, 75),
        LabelThreshold(states_measured[3], 75, 150),
    ]


def create_sensors(
    environment,
    uncertainty_parameters,
    conditional_probs = None
):
    states = environment.states
    labels = create_labels(states)

    def measure(environment: Environment):
        return environment.measure_metric("simulation_metric")

    sensors = []

    if conditional_probs == None:
        for name, relative, absolute in uncertainty_parameters:
            sensors.append(
                ScalarSensor(
                    name,
                    measure,
                    relative,
                    absolute,
                    labels
                )
            )
    else:
        assert len(uncertainty_parameters) == len(conditional_probs)
        for i in range(len(uncertainty_parameters)):
            sensors.append(
                ScalarSensor(
                    uncertainty_parameters[i][0],
                    measure,
                    uncertainty_parameters[i][1],
                    uncertainty_parameters[i][2],
                    labels,
                    conditional_probs[i]
                )
            )


    return sensors


def run_scenario(
    name,
    environment,
    sensors,
    iterations=1000,
    fusion_methods=None,
    return_all_logs=False,
    verbose=True,
):
    library = FusionLibrary().methods

    if fusion_methods is None:
        fusion_methods = list(library.keys())
    else:
        for method in fusion_methods:
            assert method in library, f"{method} is not in the library"

    output = {}

    print(f"\n{'=' * 60}")
    print(name)
    print(f"{'=' * 60}")

    for method in fusion_methods:
        print(f"\n--- {method} ---")

        simulation = Simulation(
            method,
            library[method],
            environment,
            sensors,
        )

        sim_output = simulation.run_accuracy(
            iterations=iterations,
            return_all_logs=return_all_logs,
        )

        output[method] = deepcopy(sim_output)

        if verbose:
            results = sim_output["results"]

            print("TOTAL RESULTS:")
            total = results["TOTAL"]

            for key, value in total.items():
                if key != "total cases":
                    print(f"{key:<20} {value * 100:>6.2f}%")

            print("STATE SPECIFIC:")

            for state, state_results in results.items():
                if state != "TOTAL":
                    print(f"{state}: {state_results}")

    return output


def scenario_01_low_uncertainty(
    fusion_methods=None,
    iterations=1000,
    return_all_logs=False,
    verbose=True,
):
    environment = create_environment()

    uncertainty_parameters = [
            ("LowUncertainty_001", 0.005, 0.2),
            ("LowUncertainty_002", 0.007, 0.3),
            ("LowUncertainty_003", 0.010, 0.2),
            ("LowUncertainty_004", 0.012, 0.4),
            ("LowUncertainty_005", 0.008, 0.5),
            ("LowUncertainty_006", 0.015, 0.3),
            ("LowUncertainty_007", 0.006, 0.6),
            ("LowUncertainty_008", 0.010, 0.5),
            ("LowUncertainty_009", 0.014, 0.2),
            ("LowUncertainty_010", 0.009, 0.4),
            ("LowUncertainty_011", 0.018, 0.3),
            ("LowUncertainty_012", 0.005, 0.7),
            ("LowUncertainty_013", 0.011, 0.4),
            ("LowUncertainty_014", 0.016, 0.2),
            ("LowUncertainty_015", 0.008, 0.6),
            ("LowUncertainty_016", 0.013, 0.3),
            ("HighUncertainty_017", 0.28, 5.0),
            ("HighUncertainty_018", 0.16, 7.0)
    ]

    sensors = create_sensors(
        environment,
        uncertainty_parameters,
    )

    return run_scenario(
        "Scenario 1: Low uncertainty",
        environment,
        sensors,
        iterations,
        fusion_methods,
        return_all_logs,
        verbose,
    )


def scenario_02_high_uncertainty(
    fusion_methods=None,
    iterations=1000,
    return_all_logs=False,
    verbose=True,
):
    environment = create_environment()

    uncertainty_parameters = [
            ("HighUncertainty_001", 0.10, 14.0),
            ("HighUncertainty_002", 0.12, 15.0),
            ("HighUncertainty_003", 0.15, 4.0),
            ("HighUncertainty_004", 0.18, 6.0),
            ("HighUncertainty_005", 0.14, 8.0),
            ("HighUncertainty_006", 0.20, 5.0),
            ("HighUncertainty_007", 0.11, 9.0),
            ("HighUncertainty_008", 0.16, 7.0),
            ("HighUncertainty_009", 0.22, 4.0),
            ("HighUncertainty_010", 0.13, 6.0),
            ("HighUncertainty_011", 0.25, 5.0),
            ("HighUncertainty_012", 0.10, 10.0),
            ("HighUncertainty_013", 0.17, 8.0),
            ("HighUncertainty_014", 0.23, 4.0),
            ("HighUncertainty_015", 0.14, 9.0),
            ("HighUncertainty_016", 0.19, 6.0),
            ("HighUncertainty_017", 0.28, 5.0),
            ("HighUncertainty_018", 0.16, 7.0)
    ]

    sensors = create_sensors(
        environment,
        uncertainty_parameters,
    )

    return run_scenario(
        "Scenario 2: High uncertainty",
        environment,
        sensors,
        iterations,
        fusion_methods,
        return_all_logs,
        verbose,
    )

def test_number_of_sensors(
        fusion_methods=None,
        iterations=1000,
        return_all_logs=False,
        verbose=True
        ):

    environment = create_environment()

    uncertainty_parameters = [
            ("HighUncertainty_001", 0.10, 14.0),
            ("HighUncertainty_002", 0.12, 15.0),
            ("HighUncertainty_003", 0.15, 4.0),
            ("HighUncertainty_004", 0.18, 6.0),
            ("HighUncertainty_005", 0.14, 8.0),
            ("HighUncertainty_006", 0.20, 5.0),
            ("HighUncertainty_007", 0.11, 9.0),
            ("HighUncertainty_008", 0.16, 7.0),
            ("HighUncertainty_009", 0.22, 4.0),
            ("HighUncertainty_010", 0.13, 6.0),
            ("HighUncertainty_011", 0.25, 5.0),
            ("HighUncertainty_012", 0.10, 10.0),
            ("HighUncertainty_013", 0.17, 8.0),
            ("HighUncertainty_014", 0.23, 4.0),
            ("HighUncertainty_015", 0.14, 9.0),
            ("HighUncertainty_016", 0.19, 6.0),
            ("HighUncertainty_017", 0.28, 5.0),
            ("HighUncertainty_018", 0.16, 7.0)
    ]

    for i in range(10):
        sensors = create_sensors(
            environment,
            uncertainty_parameters[:i+1],
        )
        run_scenario(
            f"Number of sensors: {i+1}",
            environment,
            sensors,
            iterations,
            fusion_methods,
            return_all_logs,
            verbose,
        )

def scenario_03_bayesian_calibrated(
        fusion_methods=None,
        iterations=1000,
        return_all_logs=False,
        verbose=True,
    ):

    states = get_states()

    # Same sensor parameters as in the previous scenarios.
    low_uncertainty_params = [
        ("Sensor_001", 0.005, 0.2),
        ("Sensor_002", 0.007, 0.3),
        ("Sensor_003", 0.010, 0.2),
        ("Sensor_004", 0.012, 0.4),
        ("Sensor_005", 0.008, 0.5),
        ("Sensor_006", 0.015, 0.3),
        ("Sensor_007", 0.006, 0.6),
        ("Sensor_008", 0.010, 0.5),
        ("Sensor_009", 0.014, 0.2),
    ]

    high_uncertainty_params = [
        ("Sensor_010", 0.10, 4.0),
        ("Sensor_011", 0.12, 5.0),
        ("Sensor_012", 0.15, 4.0),
        ("Sensor_013", 0.18, 6.0),
        ("Sensor_014", 0.14, 8.0),
        ("Sensor_015", 0.20, 5.0),
        ("Sensor_016", 0.11, 9.0),
        ("Sensor_017", 0.16, 7.0),
        ("Sensor_018", 0.22, 4.0),
    ]

    states_measured = [StateMeasured([x], 0, x.name) for x in states]

    conditional_probabilites_matrix_low = conditional_probabilities_matrix_helper([
            [0.90, 0.10, 0.00, 0.00],
            [0.10, 0.83, 0.07, 0.00],
            [0.00, 0.10, 0.80, 0.10],
            [0.00, 0.00, 0.00, 1.00]
        ], states, states_measured)

    conditional_probabilites_matrix_high = conditional_probabilities_matrix_helper([
            [0.80, 0.20, 0.00, 0.00],
            [0.16, 0.72, 0.12, 0.00],
            [0.00, 0.14, 0.68, 0.18],
            [0.00, 0.00, 0.20, 0.80]
        ], states, states_measured)

    environment = create_environment()

    sensors = create_sensors(
        environment,
        low_uncertainty_params + high_uncertainty_params,
        [conditional_probabilites_matrix_low] * len(low_uncertainty_params) 
            + [conditional_probabilites_matrix_high] * len(high_uncertainty_params)
    )

    return run_scenario(
        "Scenario 3: Probabilities defined",
        environment,
        sensors,
        iterations,
        fusion_methods,
        return_all_logs,
        verbose,
    )