from src.states import StateMeasured, State
from src.sensor import SensorOutputDict

from copy import deepcopy

class FusionLibrary:
    def __init__(self):
        self.methods = dict({
            "classical Bayes" : fusion_with_classical_bayes_inference,
            "simple voting" : fusion_simple_voting,
            "cumulative voting" : fusion_cumulative_voting,
            "approval voting" : fusion_approval_voting
        })

#################################
#####   BAYES INFERENCE     #####
#################################
def fusion_with_classical_bayes_inference(sensor_outputs : list[SensorOutputDict], # measurement dicts (Sensor.get_measurements_dict())
                                environment_states : list[State])   -> StateMeasured:
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
    # the final decision was 
    final_decision = StateMeasured([final_decision], conditional_probs_x_y_vector[final_decision], final_decision.name)
    return final_decision


def fusion_with_bayes_inference_sensor_prob(sensor_outputs : list[SensorOutputDict], # measurement dicts (Sensor.get_measurements_dict())
                                environment_states : list[State])   -> StateMeasured:
    pass

#################################
#####   DEMPSTER-SHAFER     #####
#################################
def dempster_shafer_fusing_full_outputs(sensor_outputs : list[SensorOutputDict], 
                                environment_states : list[State],
                                decision_config : str = "fused_mass")   -> list[StateMeasured]:
    import src.dst_utils as dst
    assert decision_config in ["belief_interval", "belief", "plausibility", "fused_mass"]
    # apply Dempster combination rule to fuse sensors
    combined = sensor_outputs[0].results
    for i in range(len(sensor_outputs)-1):
        combined = dst.dempster_combination_rule(combined, sensor_outputs[i+1].results)
    # decision-making - preparations
    calculate_plausibility = True if decision_config in ["belief_interval", "highest_plausibility"] else False
    calculate_belief = True if decision_config in ["belief_interval", "highest_belief"] else False
    if calculate_plausibility:
        plausibility = {}
        for state in environment_states:
            plausibility[state] = dst.plausibility_function([state], combined)
    if calculate_belief:
        belief = {}
        for state in environment_states:
            belief[state] = dst.belief_function([state], combined)
    # decision-making
    decision_output_list = []
    if decision_config == "belief":
        for state in environment_states:
            decision_output_list.append(StateMeasured([state], belief[state], state.name))
    elif decision_config == "plausibility":
        for state in environment_states:
            decision_output_list.append(StateMeasured([state], plausibility[state], state.name))
    elif decision_config == "fused_mass":
        decision_output_list = combined
    elif decision_config == "belief_interval":
        for state in environment_states:
            decision_output_list.append(dst.StateMeasured_with_BeliefInterval([state], belief[state], plausibility[state], state.name))
    return decision_output_list

def fusion_with_dempster_shafer(sensor_outputs : list[SensorOutputDict], 
                                environment_states : list[State],
                                decision_config : str = "highest_mass",
                                returned_mass_is_a_fusion : bool = False        # if True and if a few events have the same mass, the result
                                                                                # object will have a mass that is a sum of its elements
    ) -> StateMeasured:
    assert decision_config in ["belief_interval_belief_focus", "belief_interval_plausibility_focus",
                                "highest_belief", "highest_plausibility", "highest_mass"]
    # get full outputs in the specific config for DST fusion
    org_config = "belief_interval" if decision_config in ["belief_interval_belief_focus", "belief_interval_plausibility_focus"] else \
                    "belief" if decision_config == "highest_belief" else \
                    "plausibility" if decision_config == "highest_plausibility" else \
                    "fused_mass" if decision_config == "highest_mass" else \
                    None
    assert org_config is not None
    states_list = dempster_shafer_fusing_full_outputs(sensor_outputs, environment_states, org_config)
    # choose the best option according to the chosen method (config)
    if decision_config in ["belief_interval_belief_focus", "belief_interval_plausibility_focus"]:
        focus = decision_config.split("_")[-2]
        undominated = []
        dominated = []
        for opt1 in states_list:
            for opt2 in states_list:
                if opt1.friendly_name == opt2.friendly_name:
                    continue
                if opt1.belief >= opt2.belief and opt1.plausibility >= opt2.plausibility and \
                    (opt1.belief > opt2.belief or opt1.plausibility > opt2.plausibility):
                    dominated.append(opt2)
        for state in states_list:
            if state not in dominated:
                undominated.append(state)
        highest = []
        highest_val = -1.0
        for state in states_list:
            state_val = state.belief if focus == "belief" else state.plausibiltiy
            if len(highest) == 0 or highest_val == state_val:
                highest.append(state)
                highest_val = state_val
            elif highest_val < state_val:
                highest = [state]
                highest_val = state_val
        fused_states = []
        for node in highest:
            fused_states += node.states
        fused_mass = highest_val * len(highest) if returned_mass_is_a_fusion else highest_val
        fused_decisions = StateMeasured(fused_states, fused_mass, "/".join([x.friendly_name for x in highest]))
        return fused_decisions
    if decision_config == "highest_mass":
        highest = []
        for state in states_list:
            if len(highest) == 0 or highest[0].mass == state.mass:
                highest.append(state)
            elif highest[0].mass < state.mass:
                highest = [state]
        # TODO  Think about something more creative in accurate in this scenario than just
        #       returning the first element
        return highest[0]
    if decision_config == "highest_belief" or decision_config == "highest_plausibility":
        highest = []
        for state in states_list:
            if len(highest) == 0 or highest[0].mass == state.mass:
                highest.append(state)
            elif highest[0].mass < state.mass:
                highest = [state]
        fused_states = []
        for node in highest:
            fused_states += node.states
        fused_mass = highest[0].mass * len(highest) if returned_mass_is_a_fusion else highest[0].mass
        fused_decisions = StateMeasured(fused_states, fused_mass, "/".join([x.friendly_name for x in highest]))
        return fused_decisions






#################################
#####         VOTING        #####
#################################
def fusion_simple_voting(sensor_outputs : list[SensorOutputDict],
                         environment_states : list[State]) -> StateMeasured:
    # prepare the counter
    count_results = dict()
    for state in environment_states:
        count_results[state.name] = 0
    for s in sensor_outputs:
    # find the sensor's choice
        if len(s.results) == 0: 
            print(f"Sensor {s.sensor.name} gave no results")
            continue
        sensor_choice = None
        for r in s.results:
            if sensor_choice == None or r.mass > sensor_choice.mass:
                sensor_choice = r
    # add the choice to the counter
        for state in sensor_choice.states:
            count_results[state.name] = count_results[state.name] + 1
    # as a final decision choose the state with highest score
    final_decision = []
    for state in count_results.keys():
        if len(final_decision) == 0 or count_results[final_decision[0]] == count_results[state]:
            final_decision.append(state)
        elif count_results[final_decision[0]] < count_results[state]:
            final_decision = [state]
    # prepare final StateMeasured to return
    mass = count_results[final_decision[0]] / len(sensor_outputs)
    name = ""
    temp = []
    for state_name in final_decision:
        for state in environment_states:
            if state_name == state.name:
                temp.append(state)
    for s in final_decision:
        name = name + s + "/"
    name = name.rstrip("/")
    final_decision = StateMeasured(temp, mass, name)
    return final_decision


def fusion_cumulative_voting(sensor_outputs : list[SensorOutputDict],
                         environment_states : list[State]) -> StateMeasured:
    # prepare the counter
    count_results = dict()
    for state in environment_states:
        count_results[state.name] = 0
    # add sensor's choices to the counter
    # cumulative vote sum = 1
    for s in sensor_outputs:
        if len(s.results) == 0: 
            print(f"Sensor {s.sensor.name} gave no results")
            continue
        for r in s.results:
            for state in r.states:
                count_results[state.name] += r.mass
    # as a final decision choose the state with highest score
    final_decision = []
    for state in count_results.keys():
        if len(final_decision) == 0 or count_results[final_decision[0]] == count_results[state]:
            final_decision.append(state)
        elif count_results[final_decision[0]] < count_results[state]:
            final_decision = [state]
    # prepare final StateMeasured to return
    mass = count_results[final_decision[0]] / len(sensor_outputs)
    name = ""
    temp = []
    for state_name in final_decision:
        for state in environment_states:
            if state_name == state.name:
                temp.append(state)
    for s in final_decision:
        name = name + s + "/"
    name = name.rstrip("/")
    final_decision = StateMeasured(temp, mass, name)
    return final_decision


def fusion_approval_voting(sensor_outputs : list[SensorOutputDict],
                         environment_states : list[State],
                         minimum_approval_mass : float = 0.3,
                         config : str = "highest",  # config:
                                                    #  - highest = returns the states with the highest value
                                                    #  - minimum_approval = returns the states with the mass over minimum_approval_mass
                         output_minimum_approval_mass : float = None     # if None: output_minimum_approval_mass == minimum_approval_mass
                                                                    # relevant only for "minimum_approval" config
                         ) -> StateMeasured:
    assert config in ["highest", "minimum_approval"]
    if output_minimum_approval_mass == None:
        output_minimum_approval_mass = minimum_approval_mass
    elif config != "minimum_approval":
        print("Output minimum approval mass will be ignored, because of the set mode ", config)
    # prepare the counter
    count_results = dict()
    for state in environment_states:
        count_results[state.name] = 0
    # each sensor votes for or against the state
    for sensor in sensor_outputs:
        for r in sensor.results:
            if r.mass > minimum_approval_mass:
                for state in r.states:
                    count_results[state.name] += 1
    # choose final decision
    final_decision = []
    if config == "highest":
    # as a final decision choose the state with highest score
        for state in count_results.keys():
            if len(final_decision) == 0 or count_results[final_decision[0]] == count_results[state]:
                final_decision.append(state)
            elif count_results[final_decision[0]] < count_results[state]:
                final_decision = [state]
    elif config == "minimum_approval":
    # as a final decision choose all the states over output_minimum_approval_mass
        for state in count_results.keys():
            if count_results[state] / len(sensor_outputs) > output_minimum_approval_mass:
                final_decision.append(state)
    # prepare final StateMeasured to return
    if config == "highest":
        mass = count_results[final_decision[0]] / len(sensor_outputs)
    elif config == "minimum_approval":
        # be aware that in minimum_approval mode, the returned mass is a minimum approval mass
        mass = output_minimum_approval_mass
    name = ""
    temp = []
    for state_name in final_decision:
        for state in environment_states:
            if state_name == state.name:
                temp.append(state)
    for s in final_decision:
        name = name + s + "/"
    name = name.rstrip("/")
    final_decision = StateMeasured(temp, mass, name)
    return final_decision
