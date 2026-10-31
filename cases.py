"""Built-in examples for the financial market simulator.

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
    """Return a fresh copy of a built-in case."""
    probabilities = np.array([0.15, 0.25, 0.25, 0.35])
    payoffs = np.array([[1, 5, 3], [1, 3, 5], [1, 8, 4], [1, 4, 8]], dtype=float)
    states = ["BadS", "BadN", "GoodS", "GoodN"]
    assets = ["Bond", "MFC", "HFC"]
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
    elif case_number == 3:
        # Values transcribed from preset_cases/Case3.xlsx, Sheet1. Keeping the
        # case in Python means the browser app needs no Excel-reading package.
        states = ["Depression", "Recession", "Normality", "Prosperity", "Boom"]
        probabilities = np.array([0.10, 0.25, 0.30, 0.25, 0.10])
        payoffs = np.array([[1, 4, 2], [1, 2, 5], [1, 5, 3],
                            [1, 7, 6], [1, 5, 16]], dtype=float)
        holdings = [[41, 0, 10, 0], [41, 0, 0, 10]]
        discount = [0.98, 0.95]
        utility_parameters = [[2, 1.5, 0], [4, 4, 10]]
        investors = ["Mario", "Dagmar"]
        assets = ["Bond", "MFC", "DFC"]
    elif case_number == 6:
        # Transcribed from preset_cases/Case6.xlsx. Parameter 2 is unused for
        # quadratic utility and is blank in the workbook, so it is set to zero.
        states = ["Depression1", "Depression2", "Recession1", "Recession2",
                  "Normality1", "Normality2", "Prosperity1", "Prosperity2",
                  "Boom1", "Boom2"]
        probabilities = np.array([0.05, 0.05, 0.10, 0.10, 0.20,
                                  0.20, 0.10, 0.10, 0.05, 0.05])
        assets = ["STBond", "GovBds", "NonGvBds", "ValueStx", "GthStx", "SmlStx"]
        payoffs = np.array([
            [1, 0.9594, 0.8897, 0.8772, 0.7789, 0.8560],
            [1, 1.0672, 1.0470, 0.7436, 0.7164, 0.8062],
            [1, 0.8968, 0.9038, 0.9229, 0.9605, 0.8989],
            [1, 0.9135, 0.9289, 0.9297, 0.9579, 0.8207],
            [1, 1.0692, 1.0543, 1.1506, 0.9814, 1.1135],
            [1, 0.9636, 0.9940, 1.1849, 1.0584, 1.1672],
            [1, 1.1931, 1.3131, 1.1086, 1.1186, 1.1186],
            [1, 1.0434, 1.0377, 1.2397, 1.3053, 1.1915],
            [1, 1.1063, 1.1220, 1.1793, 1.3374, 1.4193],
            [1, 1.1085, 1.1141, 1.2947, 1.3060, 1.2611],
        ], dtype=float)
        holdings = [[515, 0, 100, 100, 120, 120, 60],
                    [515, 0, 100, 100, 120, 120, 60]]
        discount = [0.96, 0.96]
        utility_parameters = [[1, 600, 0], [4, 2, 100]]
        investors = ["Quade", "Dagmar"]
    else:
        raise ValueError("Choose case 1, 2, 3 or 6.")

    return Case(probabilities, payoffs, np.array(holdings, dtype=float),
                np.array(discount), np.array(utility_parameters),
                states, investors, assets)
