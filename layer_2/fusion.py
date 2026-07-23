from states import StateMeasured, State
from sensor import SensorOutputDict

#################################
#####   BAYES INFERENCE     #####
#################################
def fusion_with_classical_bayes_inference(sensor_outputs : list[SensorOutputDict], # measurement dicts (Sensor.get_measurements_dict())
                                environment_states : list[State]):
    # classical Bayes inference = taking into consideration only the most likely option
    # sensor-determined probability is not taken into consideration, only a priori and a posteriori probability, 
    #   which are the parameters of the environment (of state) and sensors (conditional probs)
    # each sensor gives one decision, which is later descibed using P(y_i|x)

    # prepare decisions vector y->
    decisions_vector = []
    for s in sensor_outputs:
        sensor_decision = None
        for r in s.results:
            if sensor_decision == None or r.mass > sensor_decision.mass:
                sensor_decision = r
        sensor_decision_tuple = (s.sensor, sensor_decision)
        decisions_vector.append(sensor_decision_tuple)
    # acquire every state a priori probability P(x)
    a_priori_probs = dict()
    for state in environment_states:
        a_priori_probs[state] = state.probability
    # calculate conditional probability for vector depending on each state P(y->|x)
    y_vector_state_conditional_prob = dict()
    for x in environment_states:
        accumulator = 1
        for y in decisions_vector:
            conditional_prob_y_x = y[0].conditional_probabilities.get_value_by_friendly_name(x.name, y[1].friendly_name).value
            accumulator = accumulator * conditional_prob_y_x
        y_vector_state_conditional_prob[x] = accumulator
    # calculate total probability for vector y->
    y_total_prob = 0
    for x in environment_states:
        y_total_prob = y_total_prob + y_vector_state_conditional_prob[x] * x.probability
    # Bayes Theorem to calculate the probabilities of states depending on vector y->
    conditional_probs_x_y_vector = dict()
    for x in environment_states:
        conditional_prob_x_y = y_vector_state_conditional_prob[x] * x.probability / y_total_prob
        conditional_probs_x_y_vector[x] = conditional_prob_x_y
    # final decision: argmax_x(P(x|y->))
    final_decision = None
    for x in environment_states:
        if final_decision == None or \
                conditional_probs_x_y_vector[x] > conditional_probs_x_y_vector[final_decision]:
            final_decision = x
    # the final decision was infered
    return final_decision

#################################
#####   DEMPSTER-SHAFER     #####
#################################
# TODO


#################################
#####     SIMPLE VOTING     #####
#################################
# TODO