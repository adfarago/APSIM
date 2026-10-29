"""Run the Python translation from a terminal.

This is a development runner, not yet the student-facing web application.
The numerical functions are kept separate so Shiny can use them later.
"""

import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cases import get_case
from simulation import (MAX_RUNTIME_SECONDS, SimulationTimeout, build_results,
                        simulate_equilibrium, state_claim_tables)


def show_tables(tables, digits):
    """Round only for display; calculations always use full precision."""
    for title, table in tables.items():
        print(f"\n{title}:")
        print(table.round(digits).to_string())


def status(result):
    description = "converged" if result.converged else "maximum number of trades reached"
    print(f"{description}; iterations: {result.iterations}; "
          f"maximum reservation-price deviation: {result.max_price_deviation:.8g}")
    if result.tolerance_relaxed:
        print(f"Slightly relaxed convergence tolerance used: {result.tolerance_used:g}")


def show_figures(case, data):
    """Recreate the two figures in the MATLAB live script."""
    fig, ax = plt.subplots()
    markers = ["o", "s", "D"]
    for i, label in enumerate(case.asset_labels):
        ax.plot(data["beta"][i], data["expected"][i], marker=markers[i],
                linestyle="none", markersize=7, label=label)
    # Include zero, like the original code, then draw the security market line.
    left, right = ax.get_xlim()
    left = min(left, 0)
    line_x = np.array([left, right])
    ax.plot(line_x, data["risk_free"] + line_x * data["market_excess"],
            color="black", linewidth=2, label="SML")
    ax.set(xlim=(left, right), xlabel="Beta", ylabel="Expected Return",
           title="Security Market Line")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-1:] + handles[:-1], labels[-1:] + labels[:-1])

    fig2, ax2 = plt.subplots()
    ax2.plot(data["market_return"], data["ppc"], "o-", linewidth=2)
    ax2.set(xlabel="Market Return", ylabel="PPC", title="Pricing Kernel & Market Return")
    ax2.grid(True)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Simulate MATLAB Case 1 or Case 2.")
    parser.add_argument("--case", type=int, choices=[1, 2], help="Built-in case (1 or 2)")
    parser.add_argument("--digits", type=int, choices=range(5), default=2,
                        help="Decimal places in tables (0 to 4; default 2)")
    parser.add_argument("--state-claims", action="store_true",
                        help="Simulate further trading in state claims")
    parser.add_argument("--no-plots", action="store_true", help="Do not open plot windows")
    args = parser.parse_args()
    if args.case is None:
        while True:
            choice = input("Choose a case (1 or 2): ").strip()
            if choice in ("1", "2"):
                args.case = int(choice)
                break
            print("Please enter 1 or 2.")

    case = get_case(args.case)
    print(f"\nCase {args.case}: simulating equilibrium...")
    start = time.perf_counter()
    started_at = time.monotonic()
    deadline = started_at + MAX_RUNTIME_SECONDS
    result = simulate_equilibrium(case, deadline=deadline, started_at=started_at)
    print(f"Elapsed time: {time.perf_counter() - start:.2f} seconds")
    status(result)
    tables, figure_data = build_results(case, result)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        show_tables(tables, args.digits)
        if args.state_claims:
            print("\nSimulating further trading with state claims...")
            start = time.perf_counter()
            second = simulate_equilibrium(case, result.holdings, trade_state_claims=True, deadline=deadline, started_at=started_at)
            print(f"Elapsed time: {time.perf_counter() - start:.2f} seconds")
            status(second)
            show_tables(state_claim_tables(case, second), args.digits)
    if not args.no_plots:
        show_figures(case, figure_data)


if __name__ == "__main__":
    try:
        main()
    except SimulationTimeout:
        print("No convergence: the maximum run time of 2 minutes was reached.")
