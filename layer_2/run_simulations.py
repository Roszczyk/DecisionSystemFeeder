from src.simulation_cases.case_01 import main_case_01
from src.simulation_cases.bos2026 import scenario_01_high_uncertainty
from src.fusion import FusionLibrary


if __name__ == "__main__":
    print("Simulation FEEDER")
    run_methods = list(FusionLibrary().methods.keys())
    run_methods.remove("classical Bayes")
    scenario_01_high_uncertainty(run_methods)
