from src.simulation_cases.case_01 import main_case_01
from src.simulation_cases.case_feeder import main_feeder_simulation
from src.fusion import FusionLibrary


if __name__ == "__main__":
    print("Simulation FEEDER")
    run_methods = list(FusionLibrary().methods.keys())
    run_methods.remove("classical Bayes")
    main_feeder_simulation(run_methods)