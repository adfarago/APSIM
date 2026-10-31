"""Convert case assumptions to and from editable Shiny tables.

These helpers do not depend on Shiny, so edited inputs can be checked before
they are passed to the numerical simulation.
"""

import numpy as np
import pandas as pd
from dataclasses import replace

from cases import Case
from simulation import reservation_prices

MAX_ASSETS = 6
MAX_STATES = 10
MAX_INVESTORS = 5
MIN_ASSETS = 1
MIN_STATES = 2
MIN_INVESTORS = 2


def case_to_draft(case):
    """Make an independent editable copy of all case inputs."""
    return {
        "assets": list(case.asset_labels), "states": list(case.state_labels),
        "investors": list(case.investor_labels),
        "preference_investors": list(case.investor_labels),
        "probabilities": case.probabilities.copy(), "payoffs": case.payoffs.copy(),
        "holdings": case.holdings.copy(), "discount": case.discount.copy(),
        "utility_parameters": case.utility_parameters.copy(),
    }


def add_asset_to_draft(draft):
    """Return a draft with a zero-payoff, zero-holding asset appended."""
    count = len(draft["assets"])
    if count >= MAX_ASSETS:
        raise ValueError("The maximum number of assets is six.")
    new_name = f"Asset {count + 1}"
    if new_name.casefold() in {str(name).strip().casefold() for name in draft["assets"]}:
        raise ValueError(f"An asset is already named {new_name}. Please rename it before adding another asset.")
    updated = dict(draft)
    updated["assets"] = [*draft["assets"], new_name]
    updated["payoffs"] = np.column_stack((draft["payoffs"], np.zeros(len(draft["states"]))))
    updated["holdings"] = np.column_stack((draft["holdings"], np.zeros(len(draft["investors"]))))
    return updated


def add_state_to_draft(draft):
    """Append a state with zero probability and zero payoffs for every asset."""
    count = len(draft["states"])
    if count >= MAX_STATES:
        raise ValueError("The maximum number of states is ten.")
    new_name = f"State {count + 1}"
    if new_name.casefold() in {str(name).strip().casefold() for name in draft["states"]}:
        raise ValueError(f"A state is already named {new_name}. Please rename it before adding another state.")
    updated = dict(draft)
    updated["states"] = [*draft["states"], new_name]
    updated["probabilities"] = np.append(draft["probabilities"], 0.0)
    updated["payoffs"] = np.vstack((draft["payoffs"], np.zeros(len(draft["assets"]))))
    return updated


def add_investor_to_draft(draft):
    """Append an investor with the requested holdings and preferences."""
    if len(draft["investors"]) >= MAX_INVESTORS:
        raise ValueError("The maximum number of investors is five.")
    new_name = f"Investor {len(draft['investors']) + 1}"
    if new_name.casefold() in {str(name).strip().casefold() for name in draft["investors"]}:
        raise ValueError(f"An investor is already named {new_name}. Please rename them before adding another investor.")
    updated = dict(draft)
    updated["investors"] = [*draft["investors"], new_name]
    updated["preference_investors"] = list(updated["investors"])
    # Columns are consumption, then the first asset, then the remaining assets.
    holdings = [50.0, 0.0, *([1.0] * (len(draft["assets"]) - 1))]
    updated["holdings"] = np.vstack((draft["holdings"], holdings))
    updated["discount"] = np.append(draft["discount"], 0.96)
    updated["utility_parameters"] = np.vstack((draft["utility_parameters"], [2.0, 2.0, 0.0]))
    return updated


def remove_last_asset_from_draft(draft):
    """Remove the rightmost asset and its payoff and portfolio columns."""
    if len(draft["assets"]) <= MIN_ASSETS:
        raise ValueError("There must be at least one asset.")
    updated = dict(draft)
    updated["assets"] = draft["assets"][:-1]
    updated["payoffs"] = draft["payoffs"][:, :-1].copy()
    updated["holdings"] = draft["holdings"][:, :-1].copy()
    return updated


def remove_last_state_from_draft(draft):
    """Remove the bottom state, its probability and its payoff row."""
    if len(draft["states"]) <= MIN_STATES:
        raise ValueError("There must be at least two states.")
    updated = dict(draft)
    updated["states"] = draft["states"][:-1]
    updated["probabilities"] = draft["probabilities"][:-1].copy()
    updated["payoffs"] = draft["payoffs"][:-1, :].copy()
    return updated


def remove_last_investor_from_draft(draft):
    """Remove the bottom investor from both portfolios and preferences."""
    if len(draft["investors"]) <= MIN_INVESTORS:
        raise ValueError("There must be at least two investors.")
    updated = dict(draft)
    updated["investors"] = draft["investors"][:-1]
    updated["preference_investors"] = draft["preference_investors"][:-1]
    updated["holdings"] = draft["holdings"][:-1, :].copy()
    updated["discount"] = draft["discount"][:-1].copy()
    updated["utility_parameters"] = draft["utility_parameters"][:-1, :].copy()
    return updated


def draft_to_case(base_case, draft):
    """Validate edited asset names and reuse the existing case validation."""
    assets = _labels(draft["assets"], "asset")
    if len(assets) < MIN_ASSETS:
        raise ValueError("There must be at least one asset.")
    if len(assets) > MAX_ASSETS:
        raise ValueError("The maximum number of assets is six.")
    if len({name.casefold() for name in assets}) != len(assets):
        raise ValueError("Asset names must be unique, ignoring capitalisation.")
    if any(name.casefold() in {"state", "probability", "investor", "consume"} for name in assets):
        raise ValueError("Asset names cannot be State, Probability, Investor or Consume.")
    payoffs = pd.DataFrame(draft["payoffs"], columns=assets)
    payoffs.insert(0, "Probability", draft["probabilities"])
    payoffs.insert(0, "State", draft["states"])
    portfolios = pd.DataFrame(draft["holdings"], columns=["Consume"] + assets)
    portfolios.insert(0, "Investor", draft["investors"])
    preferences = pd.DataFrame(draft["utility_parameters"], columns=["Utility type", "Utility param1", "Utility param2"])
    preferences.insert(0, "Discount", draft["discount"])
    preferences.insert(0, "Investor", draft["preference_investors"])
    return case_from_tables(replace(base_case, asset_labels=assets), payoffs, portfolios, preferences)


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
    if not MIN_ASSETS <= len(assets) <= MAX_ASSETS:
        raise ValueError("A case must have between one and six assets.")
    if not MIN_STATES <= len(payoff_table) <= MAX_STATES:
        raise ValueError("A case must have between two and ten states.")
    if not MIN_INVESTORS <= len(portfolio_table) <= MAX_INVESTORS:
        raise ValueError("A case must have between two and five investors.")
    if len(preference_table) != len(portfolio_table):
        raise ValueError("Each investor must have a portfolio and a preference row.")
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
