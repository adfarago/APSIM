"""Numerical translation of apsim_matlab_simulate.mlx.

NumPy uses zero-based indexing: MATLAB's first column H(:,1) becomes H[:, 0].
We retain the original pairwise trading procedure, grid sizes and tolerances.
Runs use a time limit rather than the MATLAB script's maximum number of trades.
No numerical optimisation package is used.
"""

from dataclasses import dataclass
import asyncio
import time
import numpy as np
import pandas as pd

from cases import Case

MAX_RUNTIME_SECONDS = 120
RELAX_TOLERANCE_AFTER_SECONDS = 30
RELAX_TOLERANCE_FACTOR = 1.5


class SimulationTimeout(Exception):
    """The time allowed for the simulation has elapsed before completion."""


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise SimulationTimeout("No convergence: the maximum simulation time was reached.")


@dataclass
class TradingResult:
    holdings: np.ndarray
    reservation_prices: np.ndarray
    converged: bool
    iterations: int
    max_price_deviation: float
    tolerance_used: float
    tolerance_relaxed: bool


def _utility_parameters(parameters):
    """Read one investor's MATLAB parameter row: [type, parameter 1, parameter 2]."""
    params = np.asarray(parameters, dtype=float)
    if params.shape != (3,) or not np.isfinite(params).all():
        raise ValueError("Utility parameters must be three finite numbers per investor.")
    utility_type, parameter1, parameter2 = params
    if utility_type not in (1, 2, 4):
        raise ValueError(f"Unsupported utility type {utility_type:g}; choose 1, 2 or 4.")
    if utility_type == 1 and parameter1 == 0:
        raise ValueError("Quadratic utility parameter 1 cannot be zero.")
    return utility_type, parameter1, parameter2


def _valid_consumption(consumption, parameters):
    """Real, finite utility domain for an investor's selected utility type.

    Quadratic utility is defined for any finite consumption. For CRRA we keep
    the previous positive-consumption restriction; DRRA similarly needs
    consumption above its shift, since the power may be fractional or negative.
    """
    utility_type, _, shift = _utility_parameters(parameters)
    consumption = np.asarray(consumption)
    valid = np.isfinite(consumption)
    if utility_type == 2:
        valid = valid & (consumption > 0)
    elif utility_type == 4:
        valid = valid & (consumption > shift)
    return valid


def utility(consumption, parameters):
    """MATLAB u(x, param): quadratic (1), CRRA (2), or DRRA (4)."""
    kind, parameter1, shift = _utility_parameters(parameters)
    x = np.asarray(consumption, dtype=float)
    if not np.all(_valid_consumption(x, parameters)):
        raise ValueError(f"Consumption is outside the domain of utility type {kind:g}.")
    if kind == 1:
        return 100 * x - (100 / (2 * parameter1)) * x ** 2
    if parameter1 == 1:
        raise ValueError("The MATLAB CRRA utility formula is undefined when parameter 1 is 1.")
    if kind == 2:
        return (x ** (1 - parameter1) - 1) / (1 - parameter1)
    return ((x - shift) ** (1 - parameter1) - 1) / (1 - parameter1)


def marginal_utility(consumption, parameters):
    """MATLAB u1(x, param), for the same three utility types."""
    kind, parameter1, shift = _utility_parameters(parameters)
    x = np.asarray(consumption, dtype=float)
    if not np.all(_valid_consumption(x, parameters)):
        raise ValueError(f"Consumption is outside the domain of utility type {kind:g}.")
    if kind == 1:
        return 100 - (100 / parameter1) * x
    if kind == 2:
        return x ** (-parameter1)
    return (x - shift) ** (-parameter1)


def utility_change(consumption, change, parameters):
    """Compute u(c + change) - u(c) accurately, without subtracting utilities.

    Close to equilibrium the gain may be much smaller than either utility value.
    For CRRA/DRRA, log1p and expm1 preserve precision for small changes. The
    quadratic expression is an exact algebraic expansion of the same difference.
    """
    kind, parameter1, shift = _utility_parameters(parameters)
    c = np.asarray(consumption, dtype=float)
    dc = np.asarray(change, dtype=float)
    if not np.all(_valid_consumption(c, parameters)) or not np.all(_valid_consumption(c + dc, parameters)):
        raise ValueError(f"Consumption is outside the domain of utility type {kind:g}.")
    if kind == 1:
        return (100 - (100 / parameter1) * c) * dc - (50 / parameter1) * dc ** 2
    if parameter1 == 1:
        raise ValueError("The MATLAB CRRA utility formula is undefined when parameter 1 is 1.")
    base = c if kind == 2 else c - shift
    power = 1 - parameter1
    return base ** power * np.expm1(power * np.log1p(dc / base)) / power


def reservation_prices(holdings, payoffs, probabilities, discount, utility_parameters):
    """Investor-by-asset reservation prices from discounted marginal utility."""
    now = holdings[:, 0]
    future = holdings[:, 1:] @ payoffs.T
    pricing_kernel = np.empty_like(future, dtype=float)
    for i, parameters in enumerate(utility_parameters):
        current_marginal = marginal_utility(now[i], parameters)
        if current_marginal == 0:
            raise ValueError(f"Investor {i + 1} has zero marginal utility of current consumption; reservation prices are undefined.")
        pricing_kernel[i] = (discount[i] * probabilities
                             * marginal_utility(future[i], parameters) / current_marginal)
    return pricing_kernel @ payoffs


def _trade_amount(now, future, payoff, price, direction, discount, utility_parameters,
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
        change_now = (-direction * price * dh, direction * price * dh)
        change_future = (direction * dh[:, None] * payoff, -direction * dh[:, None] * payoff)
        now1, now2 = now[0] + change_now[0], now[1] + change_now[1]
        future1 = future[0][None, :] + change_future[0]
        future2 = future[1][None, :] + change_future[1]

        # Check each investor's domain: quadratic accepts negative consumption,
        # CRRA needs x > 0, and DRRA needs x above its shift.
        feasible = (_valid_consumption(now1, utility_parameters[0])
                    & _valid_consumption(now2, utility_parameters[1])
                    & _valid_consumption(future1, utility_parameters[0]).all(axis=1)
                    & _valid_consumption(future2, utility_parameters[1]).all(axis=1))
        gains = np.full((n_grid, 2), -np.inf)
        for i in range(2):
            gains[feasible, i] = (
                utility_change(now[i], change_now[i][feasible], utility_parameters[i])
                + discount[i] * (utility_change(future[i], change_future[i][feasible],
                                                utility_parameters[i]) @ probabilities)
            )
        acceptable = np.flatnonzero(feasible & (gains[:, 0] >= 0) & (gains[:, 1] >= 0))
        # MATLAB max returns the first occurrence of a tie; np.argmax does too.
        pick = int(np.argmax(gains[acceptable].sum(axis=1)))
        chosen = float(dh[acceptable[pick]])
    return chosen


def simulate_equilibrium(case: Case, holdings=None, trade_state_claims=False,
                         n_grid=100, tolerance=None, max_iterations=None, deadline=None,
                         started_at=None):
    """Simulate asset trades, or subsequent state-claim trades.

    For state claims, holdings should be the asset-equilibrium holdings.
    Existing assets remain in the portfolio but only state claims are traded.
    """
    if started_at is None:
        started_at = time.monotonic()
    if deadline is None:
        deadline = started_at + MAX_RUNTIME_SECONDS
    steps = _equilibrium_steps(case, holdings, trade_state_claims, n_grid,
                               tolerance, max_iterations, deadline, started_at)
    while True:
        try:
            next(steps)
        except StopIteration as finished:
            return finished.value


async def simulate_equilibrium_async(case: Case, holdings=None, trade_state_claims=False,
                                     n_grid=100, tolerance=None, max_iterations=None,
                                     batch_size=20, deadline=None, started_at=None):
    """Run the same trades cooperatively so a browser can process Stop clicks.

    The simulation yields to the event loop between batches. Cancellation then
    raises asyncio.CancelledError, and no partial portfolio is returned.
    """
    if batch_size < 1:
        raise ValueError("Batch size must be positive.")
    if started_at is None:
        started_at = time.monotonic()
    if deadline is None:
        deadline = started_at + MAX_RUNTIME_SECONDS
    steps = _equilibrium_steps(case, holdings, trade_state_claims, n_grid,
                               tolerance, max_iterations, deadline, started_at)
    try:
        while True:
            # Keep NumPy's error policy scoped to a batch, never across an await.
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                for _ in range(batch_size):
                    try:
                        next(steps)
                    except StopIteration as finished:
                        return finished.value
            await asyncio.sleep(0.001)
    finally:
        steps.close()


def _equilibrium_steps(case, holdings, trade_state_claims, n_grid, tolerance, max_iterations, deadline, started_at):
    """Yield after each trade; the deadline is the normal stopping limit."""
    if max_iterations is not None and max_iterations < 1:
        raise ValueError("An explicit trade limit must be positive.")
    h = np.array(case.holdings if holdings is None else holdings, dtype=float, copy=True)
    payoffs = case.payoffs.copy()
    n_assets = payoffs.shape[1]
    n_states = len(case.probabilities)
    utility_parameters = case.utility_parameters
    if trade_state_claims:
        h = np.column_stack((h, np.zeros((len(h), n_states))))
        payoffs = np.column_stack((payoffs, np.eye(n_states)))
    if tolerance is None:
        tolerance = 5e-5 if trade_state_claims else 5e-6

    converged = False
    tolerance_used = tolerance
    tolerance_relaxed = False
    iteration = 0
    while max_iterations is None or iteration < max_iterations:
        check_deadline(deadline)
        iteration += 1
        all_prices = reservation_prices(h, payoffs, case.probabilities, case.discount, utility_parameters)
        # The second loop compares only the reservation prices of state claims.
        prices = all_prices[:, n_assets:] if trade_state_claims else all_prices
        deviations = prices - prices.mean(axis=0)
        max_deviation = float(np.max(np.abs(deviations)))
        # Prefer the original tolerance. After 30 seconds of the whole run,
        # permit a gap at most 50% larger, but never larger than that.
        relaxed_tolerance = tolerance * RELAX_TOLERANCE_FACTOR
        long_running = time.monotonic() - started_at >= RELAX_TOLERANCE_AFTER_SECONDS
        if max_deviation < tolerance or (long_running and max_deviation < relaxed_tolerance):
            converged = True
            tolerance_relaxed = max_deviation >= tolerance
            tolerance_used = relaxed_tolerance if tolerance_relaxed else tolerance
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
                               direction, case.discount[pair], utility_parameters[pair],
                               trade_probabilities, n_grid)
        h[buyer, 0] -= direction * trade_price * amount
        h[seller, 0] += direction * trade_price * amount
        h[buyer, asset_column + 1] += direction * amount
        h[seller, asset_column + 1] -= direction * amount
        yield

    # Recalculate after the final trade, including if an explicit trade limit
    # was supplied by a caller. Normal app runs do not set such a limit.
    check_deadline(deadline)
    all_prices = reservation_prices(h, payoffs, case.probabilities, case.discount, utility_parameters)
    compared = all_prices[:, n_assets:] if trade_state_claims else all_prices
    max_deviation = float(np.max(np.abs(compared - compared.mean(axis=0))))
    return TradingResult(h, all_prices, converged, iteration, max_deviation,
                         tolerance_used, tolerance_relaxed)


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

    # Compute state prices with the same investor-specific marginal utility
    # used by the trading simulation, including quadratic and DRRA.
    state_prices = np.empty_like(c, dtype=float)
    for i, parameters in enumerate(case.utility_parameters):
        current_marginal = marginal_utility(c0[i], parameters)
        if current_marginal == 0:
            raise ValueError(f"Investor {i + 1} has zero marginal utility of current consumption; state prices are undefined.")
        state_prices[i] = (case.discount[i] * prob
                           * marginal_utility(c[i], parameters) / current_marginal)
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
