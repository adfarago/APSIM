"""Student-facing Shiny interface for the financial market simulator.

The financial model remains in cases.py and simulation.py, which can also be
used by main.py. This file only handles user input and presentation. It can be
run locally with Shiny and, in a later step, exported for Shinylive.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from shiny import App, reactive, render, ui

from cases import get_case
from simulation import build_results, simulate_equilibrium, state_claim_tables


# Organise the original MATLAB output into small, readable sections. The strings
# are the keys returned by build_results() and state_claim_tables().
INPUT_TABLES = [
    "State probabilities and asset payoffs", "Initial portfolios",
    "Investor preferences",
]
EQUILIBRIUM_TABLES = [
    "Equilibrium portfolios", "Equilibrium consumption levels",
    "Equilibrium reservation prices of assets", "Equilibrium asset returns",
    "Equilibrium asset characteristics", "Portfolio weights",
]
STATE_TABLES = [
    "Equilibrium reservation prices of state claims",
    "Equilibrium price per chance (PPC)",
]

# Keeping the styling here makes the first app easy to move and customise.
# The numerical calculations are unaffected by the choice of colours or layout.
app_ui = ui.page_fluid(
    ui.tags.style("""
        body { background: #f5f7fb; color: #192a40; }
        .container-fluid { max-width: 1000px; margin-left: auto; margin-right: auto; }
        .app-heading { padding: 1.5rem 0 0.6rem; }
        .app-heading h1 { font-weight: 700; font-size: 2rem; }
        .app-heading p { color: #52647a; max-width: 68rem; }
        .control-panel { background: white; padding: 1.25rem; border: 1px solid #dce4ee;
                         border-radius: 0.75rem; margin: 0.75rem 0 1.25rem; }
        .result-table { background: white; border: 1px solid #dce4ee;
                        border-radius: 0.65rem; padding: 1rem; margin: 1rem 0; }
        .result-table h3 { font-size: 1.05rem; margin-bottom: 0.8rem; }
        .table-scroll { overflow-x: auto; }
        .table-scroll table { margin-bottom: 0; white-space: nowrap; }
        .table-scroll table th, .table-scroll table td { text-align: center; }
        .table-scroll table tr > :first-child { text-align: left; }
        .result-status { margin: 1rem 0; }
        .hint { color: #52647a; }
    """),
    ui.div(
        ui.h1("APSIM (Asset Pricing Simulator)"),
        ui.p("Explore how investors trade assets and how equilibrium prices, "
             "consumption and returns emerge."),
        class_="app-heading",
    ),
    ui.div(
        ui.layout_columns(
            ui.input_select("case", "Choose a case", {"1": "Case 1", "2": "Case 2"}, selected="1"),
            ui.input_slider("digits", "Decimal places", min=0, max=4, value=2, step=1),
            ui.input_checkbox("state_claims", "Trade state claims after equilibrium", value=False),
            col_widths=(4, 4, 4),
        ),
        ui.input_action_button("run", "Run simulation", class_="btn-primary"),
        ui.p("Choose a case and trading option, then select Run simulation. "
             "Decimal places can be changed afterwards.", class_="hint mt-2 mb-0"),
        class_="control-panel",
    ),
    ui.output_ui("status"),
    ui.navset_tab(
        ui.nav_panel("Case inputs", ui.output_ui("inputs_tables")),
        ui.nav_panel("Equilibrium", ui.output_ui("equilibrium_tables")),
        ui.nav_panel("State prices", ui.output_ui("state_tables")),
        ui.nav_panel("Figures",
                     ui.h3("Security Market Line", class_="mt-4"),
                     ui.output_plot("security_market_line", height="380px"),
                     ui.h3("Pricing Kernel & Market Return", class_="mt-4"),
                     ui.output_plot("pricing_kernel", height="380px")),
        ui.nav_panel("State-claim trading", ui.output_ui("claims_tables")),
    ),
    ui.p("Results are calculated from the assumptions of the selected case.",
         class_="hint mt-4"),
    title="APSIM (Asset Pricing Simulator)",
)


def table_cards(tables, titles, digits):
    """Format tables for display without rounding the underlying calculations."""
    cards = []
    for title in titles:
        # Preserve row names, which identify investors, assets or states.
        # escape=True ensures that table labels are treated as text, not HTML.
        html = tables[title].to_html(
            float_format=lambda number: f"{number:.{digits}f}",
            classes="table table-striped table-sm", border=0, escape=True,
        )
        cards.append(ui.div(ui.h3(title), ui.div(ui.HTML(html), class_="table-scroll"),
                            class_="result-table"))
    return ui.TagList(*cards)


def server(input, output, session):
    # The event decorator makes a snapshot when Run is clicked. Selecting a new
    # case or changing the checkbox alone does not silently replace old results.
    # Changing decimal places only affects presentation, so it does not rerun
    # the simulation. An unclicked button has value zero and produces no result.
    @reactive.calc
    @reactive.event(input.run)
    def calculation():
        case_number = int(input.case())
        wants_claims = bool(input.state_claims())
        case = get_case(case_number)
        started = time.perf_counter()
        equilibrium = simulate_equilibrium(case)
        tables, figures = build_results(case, equilibrium)
        claims = None
        claims_results = None
        if wants_claims:
            claims = simulate_equilibrium(case, equilibrium.holdings, trade_state_claims=True)
            claims_results = state_claim_tables(case, claims)
        return {
            "case_number": case_number, "case": case, "equilibrium": equilibrium,
            "tables": tables, "figures": figures, "claims": claims,
            "claims_tables": claims_results, "elapsed": time.perf_counter() - started,
        }

    @render.ui
    def status():
        # Before the first click reactive.event keeps all result outputs empty.
        result = calculation()
        eq = result["equilibrium"]
        label = "Converged" if eq.converged else "Trade limit reached before convergence"
        colour = "alert-success" if eq.converged else "alert-warning"
        parts = [f"Case {result['case_number']}: {label}. ",
                 f"Asset iterations: {eq.iterations}; calculation time: {result['elapsed']:.2f} seconds."]
        if result["claims"] is not None:
            claims = result["claims"]
            claim_status = "converged" if claims.converged else "trade limit reached before convergence"
            parts.append(f" State-claim trading: {claim_status} ({claims.iterations} iterations).")
            if not claims.converged:
                colour = "alert-warning"
        return ui.div(*parts, class_=f"alert {colour} result-status", role="status")

    @render.ui
    def inputs_tables():
        return table_cards(calculation()["tables"], INPUT_TABLES, input.digits())

    @render.ui
    def equilibrium_tables():
        return table_cards(calculation()["tables"], EQUILIBRIUM_TABLES, input.digits())

    @render.ui
    def state_tables():
        return table_cards(calculation()["tables"], STATE_TABLES, input.digits())

    @render.ui
    def claims_tables():
        result = calculation()
        if result["claims_tables"] is None:
            return ui.p("State-claim trading was not selected for this run. "
                        "Select it above and run the simulation to see these results.",
                        class_="hint mt-4")
        return table_cards(result["claims_tables"], list(result["claims_tables"]), input.digits())

    @render.plot(alt="Expected asset returns against beta, with the security market line")
    def security_market_line():
        result = calculation()
        data = result["figures"]
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, label in enumerate(result["case"].asset_labels):
            ax.plot(data["beta"][i], data["expected"][i], marker=["o", "s", "D"][i],
                    linestyle="none", markersize=7, label=label)
        left, right = ax.get_xlim()
        left = min(left, 0)
        x = np.array([left, right])
        ax.plot(x, data["risk_free"] + x * data["market_excess"], color="black", label="SML")
        ax.set(xlim=(left, right), xlabel="Beta", ylabel="Expected Return")
        ax.grid(True, alpha=0.4)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-1:] + handles[:-1], labels[-1:] + labels[:-1])
        fig.tight_layout()
        return fig

    @render.plot(alt="Price per chance against market return across states")
    def pricing_kernel():
        data = calculation()["figures"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(data["market_return"], data["ppc"], "o-", linewidth=2)
        ax.set(xlabel="Market Return", ylabel="PPC")
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        return fig


app = App(app_ui, server)
