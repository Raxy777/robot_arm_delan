"""Acceptance checks for pre-registered Phase 1.4 decision gates."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.phase1_4_gates import evaluate_calibration, evaluate_closed_loop


def main():
    good = evaluate_calibration([1] * 10, [0.70] * 8 + [0.80] * 2, 3, 5)
    
    assert good["passed"] and good["case_win_fraction"] == 1.0
    assert not evaluate_calibration([1] * 10, [0.9] * 10, 3, 5)["passed"]
    assert not evaluate_calibration([1] * 10, [0.5] * 10, 2, 5)["passed"]
    
    assert evaluate_closed_loop([1] * 15, [0.85] * 15, 3, 5)["passed"]
    assert not evaluate_closed_loop([1] * 15, [0.95] * 15, 3, 5)["passed"]
    
    print("PASS: calibration and closed-loop gates are quantitative and coverage-enforced")


if __name__ == "__main__":
    main()