from copy import deepcopy
import random

from src.states import State, StateMeasured
from src.environment import Environment, Metric
from src.sensor import ScalarSensor, LabelThreshold
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
            spread = 20
        elif state.name == "B":
            base = 45
            spread = 25
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
):
    states = environment.states
    labels = create_labels(states)

    def measure(environment: Environment):
        return environment.measure_metric("simulation_metric")

    sensors = []

    for name, relative, absolute in uncertainty_parameters:
        sensors.append(
            ScalarSensor(
                name,
                measure,
                relative,
                absolute,
                labels,
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
        ("LowUncertainty_001", 0.01, 1.5),
        ("LowUncertainty_002", 0.015, 1.25),
        ("LowUncertainty_003", 0.02, 1),
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
        ("HighUncertainty_001", 0.20, 8),
        ("HighUncertainty_002", 0.25, 10),
        ("HighUncertainty_003", 0.30, 12),
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