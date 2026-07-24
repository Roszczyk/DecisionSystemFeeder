from src.states import State

import numpy as np
import datetime 

class Environment:
    def __init__(self, states : list[State], state_changing_time_s : float = None, environmental_metrics : list[Metric] = []):
        check_sum = 0
        for state in states:
            check_sum = check_sum + state.probability
        check_sum = round(check_sum, 3) # guard agains numeric issues with float while checking
        assert check_sum == 1, f"States probabilities need to sum up to 1 not {check_sum}"
        self.states = states
        self.current_state = self.get_random_state()
        # if state_changing_time_s set to None, does not change until manually changed
        self.state_changing_routine = None if state_changing_time_s==None \
                        else datetime.timedelta(seconds = state_changing_time_s)
        self.last_state_change = datetime.datetime.now()
        # environmental metrics are metrics that can be measured from the env, e.g. temperature
        self.environmental_metrics = dict()
        for met in environmental_metrics:
            self.environmental_metrics[met.name] = met

    def get_random_state(self) -> State:
        s = [x for x in self.states]
        p = [x.probability for x in self.states]
        return np.random.choice(s, p=p)
    
    def get_current_state(self) -> State:
        if self.state_changing_routine != None and \
                datetime.datetime.now() - self.last_state_change > self.state_changing_routine:
            new_state = self.get_random_state()
            self.last_state_change = datetime.datetime.now()
            self.current_state = new_state
        return self.current_state
    
    def change_state(self, choose_state = None):
        if choose_state != None:
            for x in self.states:
                if x.name == choose_state:
                    self.current_state = x
                    return
            print(f"State {choose_state} not found, setting random state")
        self.current_state = self.get_current_state()

    def measure_metric(self, metric_name : str):
        assert metric_name in self.environmental_metrics.keys(), \
            f"There's no such metric defined as {metric_name}"
        metric = self.environmental_metrics[metric_name]
        value = metric.measure(self.get_current_state())
        return value
    
    def add_metric(self, new_metric : Metric, force : bool = False):
        if new_metric.name not in self.environmental_metrics.keys() or force:
            self.environmental_metrics[new_metric.name] = new_metric
        else:
            print(f"Metric {new_metric.name} is already existing")

class Metric:
    # Metrics are used to make scalar sensors measurement deterministic and related to the actual state
    def __init__(self, metic_name : str, measuring_function : function, starting_value = 0):
        self.name = metic_name
        self.measuring_function = measuring_function # must take 2 arguments State and Previous Value
        # example: def measuring_sensor_func(state : State, previous_value : T) -> T (where T is any type)
        self.previous_value = starting_value

    def measure(self, current_state : State):
        value_measured = self.measuring_function(current_state, self.previous_value)
        self.previous_value = value_measured
        return value_measured

###############################
###     TESTING SECTION     ###
###############################
if __name__ == "__main__":
    import time
    states = [State("X", 0.6), State("notX", 0.4)]
    env = Environment(states, 5)
    print(env.get_current_state().name)
    print(env.get_current_state().name)
    print(env.get_current_state().name)
    print(env.get_current_state().name)
    time.sleep(5)
    print(env.get_current_state().name)
    print(env.get_current_state().name)
    print(env.get_current_state().name)
    print(env.get_current_state().name)
    env = Environment(states)
    print(env.get_current_state().name)
    env.change_state()
    print(env.get_current_state().name)
    env.change_state()
    print(env.get_current_state().name)
    env.change_state("X")
    print(env.get_current_state().name)
    env.change_state("notX")
    print(env.get_current_state().name)