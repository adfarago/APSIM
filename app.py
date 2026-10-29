"""Student-facing Shiny interface for the financial market simulator.

The financial model remains in cases.py and simulation.py, which can also be
used by main.py. This file only handles user input and presentation. It can be
run locally with Shiny and, in a later step, exported for Shinylive.
"""

import asyncio
import time

import matplotlib.pyplot as plt
import numpy as np
from shiny import App, reactive, render, req, ui

from cases import get_case
from case_editor import case_from_tables, case_to_tables
from simulation import (MAX_RUNTIME_SECONDS, SimulationTimeout, build_results,
                        check_deadline, simulate_equilibrium_async, state_claim_tables)


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
        shiny-data-frame .shiny-data-grid > table > thead > tr > th,
        shiny-data-frame .shiny-data-grid > table > tbody > tr > td { text-align: center; }
        shiny-data-frame .shiny-data-grid > table > thead > tr > th:first-child,
        shiny-data-frame .shiny-data-grid > table > tbody > tr > td:first-child { text-align: left; }
        .result-status { margin: 1rem 0; }
        .hint { color: #52647a; }
        .case-editor-grid[inert] { opacity: 0.6; }
    """),
    # Keep the existing grids mounted: rebuilding them on each status change
    # would discard cell edits. The Stop button exists exactly while running.
    # `inert` blocks both mouse and keyboard interaction without changing data.
    ui.tags.script("""
        document.addEventListener('DOMContentLoaded', () => {
            const syncCaseEditorLock = () => {
                const running = document.getElementById('stop') !== null;
                document.querySelectorAll('.case-editor-grid').forEach((grid) => {
                    if (grid.inert !== running) grid.inert = running;
                });
            };
            new MutationObserver(syncCaseEditorLock).observe(document.body, {
                childList: true, subtree: true
            });
            syncCaseEditorLock();
        });
    """),
    ui.div(
        ui.h1("APSIM (Asset Pricing Simulator)"),
        ui.p('This is an implementation of the APSIM from the book "Investors and Markets" by William F. Sharpe'),
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
        ui.input_action_button("modify", "Modify Case", class_="btn-outline-primary ms-2"),
        ui.div(ui.output_ui("stop_control"), style="display: inline-block;"),
        ui.p("Select Modify Case to load editable inputs, or run the selected built-in case directly. "
             "After editing a cell, press Enter or click outside it before running.", class_="hint mt-2 mb-0"),
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
        id="results_tabs",
    ),
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
    # A separate copy of the selected built-in case is loaded into the grids.
    # Edits and results belong to this browser session only.
    editable_case = reactive.value(None)
    editing_number = reactive.value(None)
    result_value = reactive.value(None)
    stopped = reactive.value(False)
    timed_out = reactive.value(False)
    started_at = reactive.value(None)

    @reactive.extended_task
    async def simulate_task(case_number, case, wants_claims, edited, started):
        # Yield before beginning so the running controls reach the browser.
        await asyncio.sleep(0.01)
        deadline = started + MAX_RUNTIME_SECONDS
        equilibrium = await simulate_equilibrium_async(case, deadline=deadline, started_at=started)
        check_deadline(deadline)
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            tables, figures = build_results(case, equilibrium)
        claims = None
        claims_results = None
        if wants_claims:
            claims = await simulate_equilibrium_async(case, equilibrium.holdings, trade_state_claims=True, deadline=deadline, started_at=started)
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                claims_results = state_claim_tables(case, claims)
        check_deadline(deadline)
        return {
            "case_number": case_number, "case": case, "equilibrium": equilibrium,
            "tables": tables, "figures": figures, "claims": claims,
            "claims_tables": claims_results, "elapsed": time.monotonic() - started,
            "edited": edited,
        }

    session.on_ended(simulate_task.cancel)

    @reactive.effect
    def update_run_controls():
        running = simulate_task.status() == "running"
        ui.update_action_button("run", disabled=running)
        ui.update_action_button("modify", disabled=running)

    @render.ui
    def stop_control():
        if simulate_task.status() == "running":
            return ui.input_action_button("stop", "Stop Simulation", class_="btn-outline-danger ms-2")
        return None

    @reactive.effect
    @reactive.event(input.stop)
    def stop_simulation():
        if simulate_task.status() == "running":
            if time.monotonic() - started_at.get() >= MAX_RUNTIME_SECONDS:
                timed_out.set(True)
            else:
                stopped.set(True)
            result_value.set(None)
            simulate_task.cancel()

    @reactive.effect
    def collect_simulation():
        state = simulate_task.status()
        if state == "success":
            result_value.set(simulate_task.result())
        elif state == "error":
            result_value.set(None)
            error = simulate_task.error()
            if isinstance(error, SimulationTimeout):
                timed_out.set(True)
            else:
                message = str(error) if isinstance(error, ValueError) else "The inputs produced an undefined numerical result. Please check the case assumptions."
                ui.notification_show(message, type="error", duration=None)
                ui.update_navs("results_tabs", selected="Case inputs")

    @reactive.effect
    @reactive.event(input.modify)
    def load_inputs():
        if simulate_task.status() == "running":
            return
        editable_case.set(get_case(int(input.case())))
        editing_number.set(int(input.case()))
        result_value.set(None)
        stopped.set(False)
        timed_out.set(False)
        ui.update_navs("results_tabs", selected="Case inputs")

    @reactive.calc
    def edit_tables():
        case = editable_case.get()
        if case is None:
            return None
        return case_to_tables(case)

    @render.data_frame
    def edit_payoffs():
        tables = edit_tables()
        if tables is None:
            return None
        return render.DataGrid(tables[0], editable=True, width="100%", summary=False)

    @render.data_frame
    def edit_portfolios():
        tables = edit_tables()
        if tables is None:
            return None
        return render.DataGrid(tables[1], editable=True, width="100%", summary=False)

    @render.data_frame
    def edit_preferences():
        tables = edit_tables()
        if tables is None:
            return None
        return render.DataGrid(tables[2], editable=True, width="100%", summary=False)

    def parse_edit(*, patch):
        # Editable-grid values arrive as text. Convert numeric columns before
        # applying a patch so they remain compatible with pandas numeric types.
        # Also reject edits server-side in case a request arrives during a run.
        with reactive.isolate():
            if simulate_task.status() == "running":
                raise ValueError("Case inputs cannot be edited while a simulation is running.")
        if patch["column_index"] == 0:
            return str(patch["value"])
        try:
            value = float(patch["value"])
            if not np.isfinite(value):
                raise ValueError
            return value
        except (TypeError, ValueError):
            ui.notification_show("Enter a finite number in this cell.", type="error", duration=8)
            raise ValueError("Enter a finite number in this cell.") from None

    edit_payoffs.set_patch_fn(parse_edit)
    edit_portfolios.set_patch_fn(parse_edit)
    edit_preferences.set_patch_fn(parse_edit)

    # Track which case the editor contains. Changing the dropdown alone must
    # never apply a Case 1 edit to Case 2 or silently discard those edits.
    @reactive.effect
    @reactive.event(input.run)
    def run_simulation():
        if simulate_task.status() == "running":
            return
        case_number = int(input.case())
        base = editable_case.get()
        if base is not None and editing_number.get() != case_number:
            ui.notification_show("Select Modify Case to load the newly selected case before running it.",
                                 type="warning", duration=None)
            return
        try:
            if base is None:
                case = get_case(case_number)
            else:
                case = case_from_tables(base, edit_payoffs.data_patched(),
                                        edit_portfolios.data_patched(), edit_preferences.data_patched())
            stopped.set(False)
            timed_out.set(False)
            result_value.set(None)
            started = time.monotonic()
            started_at.set(started)
            simulate_task.invoke(case_number, case, bool(input.state_claims()), base is not None, started)
        except (ValueError, FloatingPointError, OverflowError) as error:
            result_value.set(None)
            message = str(error) if isinstance(error, ValueError) else "The inputs produced an undefined numerical result. Please check the case assumptions."
            ui.notification_show(message, type="error", duration=None)
            ui.update_navs("results_tabs", selected="Case inputs")

    def calculation():
        return req(result_value.get())

    @render.ui
    def status():
        if simulate_task.status() == "running":
            reactive.invalidate_later(1)
            elapsed = min(int(time.monotonic() - started_at.get()), MAX_RUNTIME_SECONDS)
            return ui.div(f"Simulating… {elapsed} seconds (maximum run time: 2 minutes).",
                          class_="alert alert-info result-status", role="status")
        if timed_out.get():
            return ui.div("No convergence: the maximum run time of 2 minutes was reached.",
                          class_="alert alert-warning result-status", role="status")
        if stopped.get():
            return ui.div("Simulation stopped.", class_="alert alert-warning result-status", role="status")
        result = calculation()
        eq = result["equilibrium"]
        label = "Converged" if eq.converged else "Trade limit reached before convergence"
        colour = "alert-success" if eq.converged else "alert-warning"
        description = " (modified inputs)" if result["edited"] else ""
        parts = [f"Case {result['case_number']}{description}: {label}. ",
                 f"Calculation time: {result['elapsed']:.2f} seconds."]
        if eq.tolerance_relaxed:
            parts.append(f" Asset trading used a slightly relaxed convergence tolerance ({eq.tolerance_used:g}).")
        if result["claims"] is not None:
            claims = result["claims"]
            claim_status = "converged" if claims.converged else "trade limit reached before convergence"
            parts.append(f" State-claim trading: {claim_status}.")
            if claims.tolerance_relaxed:
                parts.append(f" State-claim trading used a slightly relaxed convergence tolerance ({claims.tolerance_used:g}).")
            if not claims.converged:
                colour = "alert-warning"
        return ui.div(*parts, class_=f"alert {colour} result-status", role="status")

    @render.ui
    def inputs_tables():
        base = editable_case.get()
        if base is not None:
            return ui.TagList(
                ui.p(f"Editing Case {editing_number.get()}. Double-click a cell to change it. "
                     "The tables use full precision; market totals are calculated automatically. "
                     "Click Modify Case again to reload the selected case.", class_="hint mt-3"),
                ui.div(ui.h3("State probabilities and asset payoffs"), ui.div(ui.output_data_frame("edit_payoffs"), class_="case-editor-grid"), class_="result-table"),
                ui.div(ui.h3("Initial portfolios"), ui.div(ui.output_data_frame("edit_portfolios"), class_="case-editor-grid"), class_="result-table"),
                ui.div(ui.h3("Investor preferences"),
                       ui.p("Investor names and order must match the initial portfolios table. "
                            "Utility types: 1 = quadratic, 2 = CRRA, 4 = DRRA.", class_="hint"),
                       ui.div(ui.output_data_frame("edit_preferences"), class_="case-editor-grid"), class_="result-table"),
            )
        result = result_value.get()
        if result is None:
            return ui.p("Click Modify Case to load and edit the selected case, "
                        "or run it directly to view its inputs.", class_="hint mt-4")
        return table_cards(result["tables"], INPUT_TABLES, input.digits())

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
