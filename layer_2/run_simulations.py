from src.simulation_cases.case_01 import main_case_01
from src.simulation_cases.bos2026 import scenario_01_low_uncertainty, scenario_02_high_uncertainty, test_number_of_sensors
from src.fusion import FusionLibrary


if __name__ == "__main__":
    run_methods = list(FusionLibrary().methods.keys())
    run_methods.remove("classical Bayes")
    scenario_01_low_uncertainty(run_methods)
    scenario_02_high_uncertainty(run_methods)
    # test_number_of_sensors(["DST belief", "nanson voting"])