class State:            # static defined State of the Environment
    def __init__(self, name, probability):
        self.name = name
        self.probability = probability  # a priori probability of the event P(x_i)

class StateMeasured:    # object returned by the Sensor
    def __init__(self, fused_states : list[State], measured_mass : float = 0, friendly_name : str = "unnamed"):
        self.friendly_name = friendly_name
        self.states = fused_states
        self.mass = measured_mass   # in Bayes - measured probability, DST - mass m(X)