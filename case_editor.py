"""Convert case assumptions to and from editable Shiny tables.

These helpers do not depend on Shiny, so edited inputs can be checked before
they are passed to the numerical simulation.
"""

import numpy as np
import pandas as pd

from cases import Case
from simulation import reservation_prices


def case_to_tables(case):
    """Create editable tables; market totals are calculated, not edited."""
    payoffs = pd.DataFrame(case.payoffs, columns=case.asset_labels)
    payoffs.insert(0, "Probability", case.probabilities)
    payoffs.insert(0, "State", case.state_labels)
    portfolios = pd.DataFrame(case.holdings, columns=["Consume"] + case.asset_labels)
    portfolios.insert(0, "Investor", case.investor_labels)
    preferences = pd.DataFrame(case.utility_parameters, columns=["Utility type", "Utility param1", "Utility param2"])
    preferences.insert(0, "Discount", case.discount)
    # Investor names are edited in the portfolio table and shared by both tables.
    preferences.insert(0, "Investor", case.investor_labels)
    return payoffs, portfolios, preferences


def _numbers(table, columns, name):
    try:
        values = table.loc[:, columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: enter a number in every numerical cell.") from None
    if not np.isfinite(values).all():
        raise ValueError(f"{name}: values must be finite numbers.")
    return values


def _labels(values, name):
    labels = []
    for value in values:
        if pd.isna(value) or not str(value).strip():
            raise ValueError(f"Every {name} must have a name.")
        labels.append(str(value).strip())
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name.capitalize()} names must be unique.")
    if any(label.upper() == "MARKET" for label in labels):
        raise ValueError(f"{name.capitalize()} names cannot be MARKET, which is reserved for totals.")
    return labels


def case_from_tables(base_case, payoff_table, portfolio_table, preference_table):
    """Build a new Case from the grids, validating the inputs before trading."""
    assets = base_case.asset_labels
    if (len(payoff_table) != len(base_case.state_labels)
            or len(portfolio_table) != len(base_case.investor_labels)
            or len(preference_table) != len(base_case.investor_labels)):
        raise ValueError("The number of states and investors must stay the same as in the selected case.")
    states = _labels(payoff_table["State"], "state")
    investors = _labels(portfolio_table["Investor"], "investor")
    preference_investors = _labels(preference_table["Investor"], "investor")
    if preference_investors != investors:
        raise ValueError("Investor names and order must match in the portfolios and preferences tables.")

    payoff_values = _numbers(payoff_table, ["Probability"] + assets, "Probabilities and payoffs")
    holdings = _numbers(portfolio_table, ["Consume"] + assets, "Initial portfolios")
    preference_values = _numbers(preference_table, ["Discount", "Utility type", "Utility param1", "Utility param2"], "Investor preferences")
    probabilities = payoff_values[:, 0]
    if np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-8):
        raise ValueError("State probabilities must all be positive and add up to 1.")
    if np.any(payoff_values[:, 1:] < 0):
        raise ValueError("Security payoffs must be non-negative.")
    if np.any(preference_values[:, 0] <= 0):
        raise ValueError("Discount factors must be positive.")
    if not np.isin(preference_values[:, 1], [1, 2, 4]).all():
        raise ValueError("Utility type must be 1 (quadratic), 2 (CRRA), or 4 (DRRA).")
    # These are the denominators in the MATLAB utility formulas.
    utility_params = preference_values[:, 1:]
    if np.any((utility_params[:, 0] == 1) & (utility_params[:, 1] == 0)):
        raise ValueError("Quadratic utility parameter 1 cannot be zero.")
    crra_or_drra = np.isin(utility_params[:, 0], [2, 4])
    if np.any(crra_or_drra & ((utility_params[:, 1] <= 0) | (utility_params[:, 1] > 10))):
        raise ValueError("CRRA and DRRA utility parameter 1 must be greater than 0 and at most 10.")
    if np.any(crra_or_drra & (utility_params[:, 1] == 1)):
        raise ValueError("CRRA utility parameter 1 cannot be 1 in this model.")
    case = Case(probabilities.copy(), payoff_values[:, 1:].copy(), holdings.copy(),
                preference_values[:, 0].copy(), utility_params.copy(), states,
                investors, list(assets))
    # This also checks consumption domains and nonzero current marginal utility.
    reservation_prices(case.holdings, case.payoffs, case.probabilities, case.discount, case.utility_parameters)
    return case
