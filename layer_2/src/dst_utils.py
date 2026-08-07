if __name__ == "__main__":
    from states import StateMeasured, State
else:
    from src.states import StateMeasured, State

def dempster_combination_rule(
        context_states : list[State],
        sensor_A_results : list[StateMeasured],
        sensor_B_results : list[StateMeasured]):
    pass
    # TODO


def belief_function(
        context_states : list[State],
        results : list[StateMeasured]):
    mass_sum = 0.0
    ctx_list = [x.name for x in context_states]
    for res in results:
        add = False
        for state in res.states:
            if state.name not in ctx_list:
                add = False
                break
            else:
                add = True
        if add:
            mass_sum += res.mass
    return mass_sum

def plausibility_function(
        context_states : list[State],
        results : list[StateMeasured]):
    mass_sum = 0.0
    for res in results:
        add = False
        for state in res.states:
            for ctx in context_states:
                if state.name == ctx.name:
                    add = True
        if add:
            mass_sum += res.mass
    return mass_sum


### TESTING:
if __name__ == "__main__":
    states = [
        State("A", 0.4),
        State("B", 0.4),
        State("C", 0.2)
    ]

    # 2^\Theta = 
    # - A           0.2
    # - B           0.2
    # - C           0.2
    # - A,B         0.1
    # - B,C         0.1
    # - C,A         0.1
    # - A,B,C       0.1

    states_measured = []
    for st in states:
        states_measured.append(StateMeasured([st], 0.2))
    for st_1 in states:
        for st_2 in states:
            if st_1.name != st_2.name and st_1.name > st_2.name:
                states_measured.append(StateMeasured([st_1, st_2], 0.1))
    states_measured.append(StateMeasured(states, 0.1))

    for sm in states_measured:
        print([x.name for x in sm.states])

    print(plausibility_function([states[0]], states_measured), "== 0.5")
    print(plausibility_function([states[0], states[1]], states_measured), "== 0.8")
    
    print(belief_function([states[0]], states_measured), "== 0.2")
    print(belief_function([states[0], states[1]], states_measured), "== 0.5")