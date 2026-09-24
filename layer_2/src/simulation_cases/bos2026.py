# CASE 1 - scenario with 

from src.states import State, StateMeasured
from src.environment import Environment, Metric
from src.sensor import ScalarSensor, LabelThreshold, conditional_probabilities_matrix_helper
from src.fusion import FusionLibrary
from src.simulation import Simulation

import random
from copy import deepcopy

from src.states import State, StateMeasured
from src.environment import Environment, Metric
from src.sensor import (
    ScalarSensor,
    SimpleSensor,
    LabelThreshold,
    conditional_probabilities_matrix_helper,
)
from src.fusion import FusionLibrary
from src.simulation import Simulation

import random
from copy import deepcopy

def get_states():
    return [
        State("A", 0.3),
        State("B", 0.4),
        State("C", 0.2),
        State("D", 0.1),
    ]

def run_scenario(name, env, sensors, iterations=1000, fusion_methods=None, return_all_logs=False, verbose=True):

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
            env,
            sensors,
        )

        sim_output = simulation.run_accuracy(
            iterations=iterations,
            return_all_logs=return_all_logs
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

def scenario_01_high_uncertainty(
    fusion_methods=None,
    iterations=1000,
    return_all_logs=False,
    verbose=True,
):
    states = deepcopy(get_states())
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

    metric = Metric("high_uncertainty_metric", measure_metric)
    env.add_metric(metric)

    def measure(environment: Environment):
        return environment.measure_metric("high_uncertainty_metric")

    states_measured = [
        StateMeasured([states[0]], 0, "A"),
        StateMeasured([states[1]], 0, "B"),
        StateMeasured([states[2]], 0, "C"),
        StateMeasured([states[3]], 0, "D"),
    ]

    labels = [
        LabelThreshold(states_measured[0], -100, 30),
        LabelThreshold(states_measured[1], 30, 55),
        LabelThreshold(states_measured[2], 55, 75),
        LabelThreshold(states_measured[3], 75, 150),
    ]

    sensors = [
        ScalarSensor(
            "HighUncertainty_001",
            measure,
            0.20,
            8,
            labels,
        ),
        ScalarSensor(
            "HighUncertainty_002",
            measure,
            0.25,
            10,
            labels,
        ),
        ScalarSensor(
            "HighUncertainty_003",
            measure,
            0.30,
            12,
            labels,
        ),
    ]

    return run_scenario(
        "Scenario 1: High uncertainty",
        env,
        sensors,
        iterations,
        fusion_methods,
        return_all_logs,
        verbose,
    )