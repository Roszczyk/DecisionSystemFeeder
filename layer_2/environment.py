from states import State

import numpy as np
import datetime 

class Environment:
    def __init__(self, states : list[State], state_changing_time_s : float = None):
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