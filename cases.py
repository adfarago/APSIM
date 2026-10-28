"""The two built-in examples from getInputs in the MATLAB live script.

All portfolios are measured in units of the consumption good at time 0 and
in units of each asset. Each row is an investor; each column after the first
is an asset. The payoff matrix has one row per state and one column per asset.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Case:
    probabilities: np.ndarray
    payoffs: np.ndarray
    holdings: np.ndarray
    discount: np.ndarray
    utility_parameters: np.ndarray
    state_labels: list[str]
    investor_labels: list[str]
    asset_labels: list[str]


def get_case(case_number: int) -> Case:
    """Return a fresh copy of Case 1 or Case 2, as defined in MATLAB."""
    probabilities = np.array([0.15, 0.25, 0.25, 0.35])
    payoffs = np.array([[1, 5, 3], [1, 3, 5], [1, 8, 4], [1, 4, 8]], dtype=float)
    if case_number == 1:
        holdings = [[49, 0, 10, 0], [49, 0, 0, 10]]
        discount = [0.96, 0.96]
        utility_parameters = [[2, 1.5, 0], [2, 2.5, 0]]
        investors = ["Mario", "Hue"]
    elif case_number == 2:
        holdings = [[49, 0, 10, 0], [98, 0, 20, 0],
                    [49, 0, 0, 10], [98, 0, 0, 20]]
        discount = [0.96] * 4
        utility_parameters = [[2, 1.5, 0], [2, 1.5, 0],
                              [2, 2.5, 0], [2, 2.5, 0]]
        investors = ["Mario", "Marie", "Hue", "Hugo"]
    else:
        raise ValueError("Choose case 1 or case 2.")

    return Case(probabilities, payoffs, np.array(holdings, dtype=float),
                np.array(discount), np.array(utility_parameters),
                ["BadS", "BadN", "GoodS", "GoodN"], investors,
                ["Bond", "MFC", "HFC"])
