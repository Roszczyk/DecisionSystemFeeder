from src.environment import Environment
from src.sensor import Sensor
from src.states import State, StateMeasured

import logging
from copy import deepcopy

class AccuracyCounter:
    def __init__(self, name : str = "TOTAL", total_cases : int = None):
        self.name = name
        self.results_including_correct = 0   # counter for when the correct states in among the returned
        self.results_only_correct = 0        # counter for when fusion returned only the correct state
        self.results_correct_parts = 0       # adding the part of the answer that was correct
                                             # e.g. if fusion returns n answers, 1 is correct, 1/n is added
        # how many total cases happen (denominator):
        self.total_cases = total_cases if total_cases != None else 0
        self.total_cases_set_at_beginning = False if total_cases == None else True

    def add_to_results_including_correct(self):
        self.results_including_correct += 1
    
    def add_to_results_only_correct(self):
        self.results_only_correct += 1

    def add_to_results_correct_parts(self, part : float):
        assert part <= 1.0 and part >=0.0
        self.results_correct_parts += part

    def add_to_total_cases(self):
        if self.total_cases_set_at_beginning:
            print(f"Number of total cases for {self.name} was set at the beginning")
        else:
            self.total_cases += 1

    def return_metrics(self):
        if self.total_cases > 0:
            return dict({
                "including correct" : self.results_including_correct / self.total_cases,
                "only correct" : self.results_only_correct / self.total_cases,
                "correct parts" : self.results_correct_parts,
                "total cases" : self.total_cases
            })
        else:
            return dict({
                "including correct" : 0,
                "only correct" : 0,
                "correct parts" : 0,
                "total cases" : 0
            })



class Simulation:
    def __init__(self, 
                 simulation_name : str,
                 fusion_method : function, # func(sensor_outputs, environment_states)
                 environment : Environment,
                 sensors : list[Sensor]):
        self.name = simulation_name
        self.fusion_method = fusion_method
        if environment.state_changing_routine != None:
            print("For simulation to run accuratly, state changing routine needs to be manual.")
            environment = Environment(environment.states, None, environment.environmental_metrics)
        self.env = deepcopy(environment)
        self.sensors = deepcopy(sensors)

    def fusion(self, sensor_outputs) -> StateMeasured:
        states = self.env.get_env_states()
        return self.fusion_method(sensor_outputs, states)
    
    def run_accuracy(self, iterations : int = 100):
        main_counter = AccuracyCounter(total_cases=iterations)
        states_counters = dict()
        for state in self.env.states:
            states_counters[state.name] = AccuracyCounter(name=state.name)
        
        for i in range(iterations):
            current_state = self.env.get_current_state()
            states_counters[current_state.name].add_to_total_cases()
            results = [x.get_measurements_dict(self.env) for x in self.sensors]
            final_decision = self.fusion(results)
            if len(final_decision.states) == 1 and \
                final_decision.states[0].name == current_state.name:
                main_counter.add_to_results_correct_parts(1.0)
                main_counter.add_to_results_including_correct()
                main_counter.add_to_results_only_correct()
                states_counters[current_state.name].add_to_results_correct_parts(1)
                states_counters[current_state.name].add_to_results_including_correct()
                states_counters[current_state.name].add_to_results_only_correct()
                continue
            for state in final_decision.states:
                if state.name == current_state.name:
                    main_counter.add_to_results_including_correct()
                    main_counter.add_to_results_correct_parts(1 / len(final_decision.states))
                    states_counters[current_state.name].add_to_results_including_correct()
                    states_counters[current_state.name].add_to_results_correct_parts(1 / len(final_decision.states))
            self.env.change_state()
        
        results_dict = dict()
        results_dict["TOTAL"] = main_counter.return_metrics()
        for state_name in states_counters.keys():
            results_dict[state_name] = states_counters[state_name].return_metrics()
        return results_dict
