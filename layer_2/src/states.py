class State:            # static defined State of the Environment
    def __init__(self, name, probability):
        self.name = name
        self.probability = probability  # a priori probability of the event P(x_i)

class StateMeasured:    # object returned by the Sensor
    def __init__(self, fused_states : list[State], measured_mass : float = 0, friendly_name : str = None):
        self.states = fused_states
        self.mass = measured_mass                       # in Bayes - measured probability, DST - mass m(X)
        if friendly_name is not None:
            self.friendly_name = friendly_name
        else:
            # if not given, the friendly name is created by joining together the fused states
            self.friendly_name = "/".join([x.name for x in self.states])