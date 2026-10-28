"""Numerical translation of apsim_matlab_simulate.mlx.

NumPy uses zero-based indexing: MATLAB's first column H(:,1) becomes H[:, 0].
We retain the original pairwise trading procedure, grid sizes, tolerances and
maximum number of trades. No numerical optimisation package is used.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

from cases import Case


@dataclass
class TradingResult:
    holdings: np.ndarray
    reservation_prices: np.ndarray
    converged: bool
    iterations: int
    max_price_deviation: float


def utility(consumption, gamma):
    """CRRA utility, matching u(x,param) in MATLAB for these two cases."""
    # Both built-in cases have gamma different from 1, so the MATLAB formula
    # applies directly. Invalid candidate consumptions are handled below.
    return (consumption ** (1 - gamma) - 1) / (1 - gamma)


def marginal_utility(consumption, gamma):
    return consumption ** (-gamma)


def reservation_prices(holdings, payoffs, probabilities, discount, gamma):
    """Investor-by-asset reservation prices from discounted marginal utility."""
    now = holdings[:, 0]
    future = holdings[:, 1:] @ payoffs.T
    if np.any(now <= 0) or np.any(future <= 0):
        raise ValueError("Consumption must be positive in every state.")
    pricing_kernel = (discount[:, None] * probabilities[None, :]
                      * marginal_utility(future, gamma[:, None])
                      / marginal_utility(now, gamma)[:, None])
    return pricing_kernel @ payoffs


def _trade_amount(now, future, payoff, price, direction, discount, gamma,
                  probabilities, n_grid):
    """Replicate the shrinking-grid search used by both MATLAB trade loops.

    The candidates begin at 0 and end at 1, then at 0.1, 0.01, and so on
    if the maximum combined utility gain occurs at the first candidate.
    """
    pick = 0  # MATLAB's h_pick=1 is Python index 0.
    max_dh = 10.0
    chosen = 0.0
    while pick == 0 and max_dh > 1e-6:
        max_dh /= 10
        dh = np.linspace(0, max_dh, n_grid)
        now1 = now[0] - direction * price * dh
        now2 = now[1] + direction * price * dh
        future1 = future[0][None, :] + direction * dh[:, None] * payoff
        future2 = future[1][None, :] - direction * dh[:, None] * payoff

        # CRRA utility is real-valued only for positive consumption. Exclude
        # infeasible candidate trades; zero trade remains a valid candidate.
        feasible = ((now1 > 0) & (now2 > 0) & (future1 > 0).all(axis=1)
                    & (future2 > 0).all(axis=1))
        gains = np.full((n_grid, 2), -np.inf)
        for i, (c0, c) in enumerate(((now1, future1), (now2, future2))):
            values = (utility(c0[feasible], gamma[i])
                      + discount[i] * (utility(c[feasible], gamma[i]) @ probabilities))
            gains[feasible, i] = values - values[0]
        acceptable = np.flatnonzero(feasible & (gains[:, 0] >= 0) & (gains[:, 1] >= 0))
        # MATLAB max returns the first occurrence of a tie; np.argmax does too.
        pick = int(np.argmax(gains[acceptable].sum(axis=1)))
        chosen = float(dh[acceptable[pick]])
    return chosen


def simulate_equilibrium(case: Case, holdings=None, trade_state_claims=False,
                         n_grid=100, tolerance=None, max_iterations=10000):
    """Simulate asset trades, or subsequent state-claim trades.

    For state claims, holdings should be the asset-equilibrium holdings.
    Existing assets remain in the portfolio but only state claims are traded.
    """
    h = np.array(case.holdings if holdings is None else holdings, dtype=float, copy=True)
    payoffs = case.payoffs.copy()
    n_assets = payoffs.shape[1]
    n_states = len(case.probabilities)
    gamma = case.utility_parameters[:, 1]
    if trade_state_claims:
        h = np.column_stack((h, np.zeros((len(h), n_states))))
        payoffs = np.column_stack((payoffs, np.eye(n_states)))
    if tolerance is None:
        tolerance = 5e-5 if trade_state_claims else 5e-6

    converged = False
    for iteration in range(1, max_iterations + 1):
        all_prices = reservation_prices(h, payoffs, case.probabilities, case.discount, gamma)
        # The second loop compares only the reservation prices of state claims.
        prices = all_prices[:, n_assets:] if trade_state_claims else all_prices
        deviations = prices - prices.mean(axis=0)
        max_deviation = float(np.max(np.abs(deviations)))
        if max_deviation < tolerance:
            converged = True
            break

        # Choose the asset/state with the greatest range across investors.
        chosen_column = int(np.argmax(np.ptp(deviations, axis=0)))
        order = np.argsort(deviations[:, chosen_column], kind="stable")
        buyer, seller = int(order[-1]), int(order[0])
        direction = np.sign(prices[buyer, chosen_column] - prices[seller, chosen_column])
        trade_price = (prices[buyer, chosen_column] + prices[seller, chosen_column]) / 2
        asset_column = n_assets + chosen_column if trade_state_claims else chosen_column
        pair = np.array([buyer, seller])
        current = h[pair, 0]
        future = h[pair, 1:] @ payoffs.T
        trade_payoff = payoffs[:, asset_column]
        trade_probabilities = case.probabilities
        if trade_state_claims:
            # MATLAB evaluates only the affected state's utility in this stage.
            future = future[:, chosen_column:chosen_column + 1]
            trade_payoff = trade_payoff[chosen_column:chosen_column + 1]
            trade_probabilities = case.probabilities[chosen_column:chosen_column + 1]
        amount = _trade_amount(current, future, trade_payoff, trade_price,
                               direction, case.discount[pair], gamma[pair],
                               trade_probabilities, n_grid)
        h[buyer, 0] -= direction * trade_price * amount
        h[seller, 0] += direction * trade_price * amount
        h[buyer, asset_column + 1] += direction * amount
        h[seller, asset_column + 1] -= direction * amount

    # Recalculate after the final trade, including when the iteration limit is
    # reached. This avoids reporting stale prices in the non-converged case.
    all_prices = reservation_prices(h, payoffs, case.probabilities, case.discount, gamma)
    compared = all_prices[:, n_assets:] if trade_state_claims else all_prices
    max_deviation = float(np.max(np.abs(compared - compared.mean(axis=0))))
    return TradingResult(h, all_prices, converged, iteration, max_deviation)


def build_results(case: Case, result: TradingResult):
    """Produce the tables and plot data in the MATLAB script, before rounding."""
    h = result.holdings
    p = result.reservation_prices
    prob, x = case.probabilities, case.payoffs
    investors, assets, states = case.investor_labels, case.asset_labels, case.state_labels
    market_rows = investors + ["MARKET"]
    # Rows of C correspond to investors, columns to states.
    c0, c = h[:, 0], h[:, 1:] @ x.T
    market_prices = p.mean(axis=0)
    returns = (x / market_prices).T
    market_holdings = case.holdings[:, 1:].sum(axis=0)
    market_return = (x @ market_holdings) / (market_holdings @ market_prices)
    all_returns = np.vstack((returns, market_return))

    gamma = case.utility_parameters[:, 1]
    state_prices = (case.discount[:, None] * prob
                    * marginal_utility(c, gamma[:, None])
                    / marginal_utility(c0, gamma)[:, None])
    state_prices = np.vstack((state_prices, state_prices.mean(axis=0)))
    ppc = state_prices / prob
    risk_free_return = 1 / (ppc[-1] @ prob)
    expected = all_returns @ prob
    excess = expected - risk_free_return
    sd = np.sqrt(((all_returns - expected[:, None]) ** 2) @ prob)
    # Follow MATLAB's convention: a zero-volatility asset has Sharpe ratio 0.
    sharpe = np.divide(excess, sd, out=np.zeros_like(excess), where=sd != 0)
    market_variance = sd[-1] ** 2
    beta = ((all_returns * market_return) @ prob - expected * (market_return @ prob)) / market_variance
    alpha = excess - beta * excess[-1]
    position_values = h[:, 1:] * market_prices
    weights = position_values / position_values.sum(axis=1, keepdims=True)

    def with_market(values):
        return np.vstack((values, values.sum(axis=0)))

    tables = {
        "State probabilities and asset payoffs": pd.DataFrame(np.column_stack((prob, x)), index=states, columns=["Prob"] + assets),
        "Initial portfolios": pd.DataFrame(with_market(case.holdings), index=market_rows, columns=["Consume"] + assets),
        "Investor preferences": pd.DataFrame(np.column_stack((case.discount, case.utility_parameters)), index=investors, columns=["Discount", "utility type", "utility param1", "utility param2"]),
        "Equilibrium portfolios": pd.DataFrame(with_market(h), index=market_rows, columns=["Consume"] + assets),
        "Equilibrium consumption levels": pd.DataFrame(with_market(np.column_stack((c0, c))), index=market_rows, columns=["Now"] + states),
        "Equilibrium reservation prices of assets": pd.DataFrame(np.vstack((p, market_prices)), index=market_rows, columns=assets),
        "Equilibrium asset returns": pd.DataFrame(all_returns, index=assets + ["MARKET"], columns=states),
        "Equilibrium asset characteristics": pd.DataFrame(np.column_stack((expected, excess, sd, sharpe, beta, alpha)), index=assets + ["MARKET"], columns=["Exp_Ret", "Exp_ER", "SD_Ret", "SR", "Beta", "Alpha"]),
        "Portfolio weights": pd.DataFrame(weights, index=investors, columns=assets),
        "Equilibrium reservation prices of state claims": pd.DataFrame(state_prices, index=market_rows, columns=states),
        "Equilibrium price per chance (PPC)": pd.DataFrame(ppc, index=market_rows, columns=states),
    }
    return tables, {"beta": beta, "expected": expected, "risk_free": risk_free_return,
                    "market_excess": excess[-1], "market_return": market_return, "ppc": ppc[-1]}


def state_claim_tables(case: Case, result: TradingResult):
    """Tables reported after the optional second stage of trading."""
    h, p = result.holdings, result.reservation_prices
    x = np.column_stack((case.payoffs, np.eye(len(case.probabilities))))
    labels = case.asset_labels + ["SC_" + state for state in case.state_labels]
    rows = case.investor_labels + ["MARKET"]
    consumption = np.column_stack((h[:, 0], h[:, 1:] @ x.T))
    return {
        "Portfolios after trading state claims": pd.DataFrame(np.vstack((h, h.sum(axis=0))), index=rows, columns=["Consume"] + labels),
        "Consumption after trading state claims": pd.DataFrame(np.vstack((consumption, consumption.sum(axis=0))), index=rows, columns=["Now"] + case.state_labels),
        "Reservation prices after trading state claims": pd.DataFrame(np.vstack((p, p.mean(axis=0))), index=rows, columns=labels),
    }
