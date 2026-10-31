"""Student-facing Shiny interface for the financial market simulator.

The financial model remains in cases.py and simulation.py, which can also be
used by main.py. This file only handles user input and presentation. It can be
run locally with Shiny and, in a later step, exported for Shinylive.
"""

import asyncio
import time

from matplotlib.figure import Figure
import numpy as np
from shiny import App, reactive, render, req, ui

from cases import get_case
from case_editor import (add_asset_to_draft, add_investor_to_draft,
                         add_state_to_draft, case_to_draft, draft_to_case,
                         remove_last_asset_from_draft, remove_last_investor_from_draft,
                         remove_last_state_from_draft)
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
        /* Give the editable inputs the compact look of the previous data grid. */
        .input-table { width: 100%; border-collapse: separate; border-spacing: 0;
                       border-top: 1px solid #dce1e7; border-left: 1px solid #dce1e7;
                       font-size: 0.9em; }
        .input-table th, .input-table td { padding: 0; text-align: center;
                       border-right: 1px solid #dce1e7; border-bottom: 1px solid #dce1e7; }
        .input-table th { background: #f2f3f5; font-weight: 600; padding: 0.3rem 0.5rem; }
        .input-table th:has(input) { padding: 0; }
        .input-table tbody tr:hover { background: #f6f8fa; }
        .input-table th:first-child, .input-table td:first-child { text-align: left; }
        .input-table input.form-control { min-width: 5rem; padding: 0.3rem 0.5rem;
                       height: auto; border: 1px solid transparent; border-radius: 0;
                       background: transparent; box-shadow: none; text-align: center;
                       font-size: inherit; line-height: 1.5; }
        .input-table th input.form-control { font-weight: 600; }
        .input-table input.form-control:focus { background: #fff; border-color: #86b7fe;
                       box-shadow: inset 0 0 0 1px #86b7fe; }
        .input-table label.control-label { position: absolute; width: 1px; height: 1px;
                       padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0);
                       white-space: nowrap; border: 0; }
        .input-table input[type=number] { appearance: textfield; -moz-appearance: textfield; }
        .input-table input[type=number]::-webkit-inner-spin-button,
        .input-table input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
        .input-table td:first-child input { text-align: left; min-width: 7rem; }
        .input-table .form-group, .input-table .shiny-input-container { margin-bottom: 0; width: 100%; }
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
            // Asset names are edited only in the payoff header. Mirror the
            // name immediately in the portfolio header without rebuilding data.
            document.addEventListener('input', (event) => {
                const match = /^asset_name_(\\d+)$/.exec(event.target.id || '');
                if (match) {
                    document.querySelectorAll('[data-asset-index="' + match[1] + '"]')
                        .forEach((header) => { header.textContent = event.target.value; });
                }
                const investorMatch = /^investor_name_(\\d+)$/.exec(event.target.id || '');
                if (investorMatch) {
                    document.querySelectorAll('[data-investor-index="' + investorMatch[1] + '"]')
                        .forEach((cell) => { cell.textContent = event.target.value; });
                }
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
            ui.input_select("case", "Choose a case", {"1": "Case 1", "2": "Case 2", "3": "Case 3", "6": "Case 6"}, selected="1"),
            ui.input_slider("digits", "Decimal places", min=0, max=4, value=2, step=1),
            ui.input_checkbox("state_claims", "Trade state claims after equilibrium", value=False),
            col_widths=(4, 4, 4),
        ),
        ui.input_action_button("run", "Run simulation", class_="btn-primary"),
        ui.input_action_button("modify", "Load Case", class_="btn-outline-primary ms-2"),
        ui.div(ui.output_ui("stop_control"), style="display: inline-block;"),
        ui.p("Select Load Case to load editable inputs, or run the selected built-in case directly. "
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
            classes="table table-striped table-sm", border=0, escape=True, na_rep="—",
        )
        cards.append(ui.div(ui.h3(title), ui.div(ui.HTML(html), class_="table-scroll"),
                            class_="result-table"))
    return ui.TagList(*cards)


def editor_input(id, value, description, numeric=True):
    """Create a compact table input with an accessible label."""
    label = ui.span(description, class_="visually-hidden")
    if numeric:
        # Browser number controls follow the operating system's locale and can
        # show commas. A text control displays the Python decimal point exactly;
        # read_editor() checks and converts the value before it is used.
        field = ui.input_text(id, label, value=str(float(value)))
        field.children[-1].attrs["inputmode"] = "decimal"
        return field
    return ui.input_text(id, label, value=str(value))


def editable_tables(draft):
    """Create the editable case tables, with asset names in the payoff header."""
    assets, states, investors = draft["assets"], draft["states"], draft["investors"]
    payoff_header = ui.tags.tr(
        ui.tags.th("State"), ui.tags.th("Probability"),
        *[ui.tags.th(editor_input(f"asset_name_{j}", name, f"Asset {j + 1} name", False))
          for j, name in enumerate(assets)],
    )
    payoff_rows = [ui.tags.tr(
        ui.tags.td(editor_input(f"state_name_{i}", state, f"State {i + 1} name", False)),
        ui.tags.td(editor_input(f"probability_{i}", draft["probabilities"][i], f"Probability for {state}")),
        *[ui.tags.td(editor_input(f"payoff_{i}_{j}", draft["payoffs"][i, j], f"Payoff for {name} in {state}"))
          for j, name in enumerate(assets)],
    ) for i, state in enumerate(states)]
    portfolio_header = ui.tags.tr(
        ui.tags.th("Investor"), ui.tags.th("Consume"),
        *[ui.tags.th(ui.tags.span(name, **{"data-asset-index": str(j)})) for j, name in enumerate(assets)],
    )
    portfolio_rows = [ui.tags.tr(
        ui.tags.td(editor_input(f"investor_name_{i}", investor, f"Investor {i + 1} name", False)),
        *[ui.tags.td(editor_input(f"holding_{i}_{j}", draft["holdings"][i, j],
                                  f"{investor} initial {'consumption' if j == 0 else assets[j - 1]}"))
          for j in range(len(assets) + 1)],
    ) for i, investor in enumerate(investors)]
    preference_header = ui.tags.tr(*[ui.tags.th(label) for label in
        ("Investor", "Discount", "Utility type", "Utility param1", "Utility param2")])
    preference_rows = [ui.tags.tr(
        ui.tags.td(ui.tags.span(investor, **{"data-investor-index": str(i)})),
        ui.tags.td(editor_input(f"discount_{i}", draft["discount"][i], f"{investor} discount")),
        *[ui.tags.td(editor_input(f"utility_{i}_{j}", draft["utility_parameters"][i, j], f"{investor} utility {j + 1}"))
          for j in range(3)],
    ) for i, investor in enumerate(investors)]
    def table(header, rows):
        return ui.div(ui.tags.table(ui.tags.thead(header), ui.tags.tbody(*rows), class_="input-table"), class_="table-scroll")
    return ui.TagList(
        ui.div(ui.h3("State probabilities and asset payoffs"),
               ui.div(table(payoff_header, payoff_rows),
                      ui.input_action_button("add_asset", "Add Asset", class_="btn-outline-primary mt-3"),
                      ui.input_action_button("remove_asset", "Remove Last Asset", class_="btn-outline-primary mt-3 ms-2"),
                      ui.input_action_button("add_state", "Add State", class_="btn-outline-primary mt-3 ms-2"),
                      ui.input_action_button("remove_state", "Remove Last State", class_="btn-outline-primary mt-3 ms-2"),
                      class_="case-editor-grid"), class_="result-table"),
        ui.div(ui.h3("Initial portfolios"),
               ui.div(table(portfolio_header, portfolio_rows),
                      ui.input_action_button("add_investor", "Add Investor", class_="btn-outline-primary mt-3"),
                      ui.input_action_button("remove_investor", "Remove Last Investor", class_="btn-outline-primary mt-3 ms-2"),
                      class_="case-editor-grid"), class_="result-table"),
        ui.div(ui.h3("Investor preferences"),
               ui.p("Investor names are edited in the initial portfolios table. "
                    "Utility types: 1 = quadratic, 2 = CRRA, 4 = DRRA.", class_="hint"),
               ui.div(table(preference_header, preference_rows), class_="case-editor-grid"), class_="result-table"),
    )


def server(input, output, session):
    # A separate copy of the selected built-in case is loaded into the grids.
    # Edits and results belong to this browser session only.
    editable_case = reactive.value(None)
    editor_draft = reactive.value(None)
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
        ui.update_action_button("add_asset", disabled=running)
        ui.update_action_button("add_state", disabled=running)
        ui.update_action_button("add_investor", disabled=running)
        ui.update_action_button("remove_asset", disabled=running)
        ui.update_action_button("remove_state", disabled=running)
        ui.update_action_button("remove_investor", disabled=running)

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
        selected_case = get_case(int(input.case()))
        editable_case.set(selected_case)
        editor_draft.set(case_to_draft(selected_case))
        editing_number.set(int(input.case()))
        result_value.set(None)
        stopped.set(False)
        timed_out.set(False)
        ui.update_navs("results_tabs", selected="Case inputs")

    def read_editor():
        """Take a snapshot of all currently entered values without rounding."""
        draft = editor_draft.get()
        n_assets, n_states, n_investors = len(draft["assets"]), len(draft["states"]), len(draft["investors"])
        def number(id):
            value = input[id]()
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ValueError("Enter a finite number using a decimal point in every numerical cell.") from None
            if not np.isfinite(value):
                raise ValueError("Enter a finite number in every numerical cell.")
            return value
        return {
            "assets": [input[f"asset_name_{j}"]() for j in range(n_assets)],
            "states": [input[f"state_name_{i}"]() for i in range(n_states)],
            "investors": [input[f"investor_name_{i}"]() for i in range(n_investors)],
            "preference_investors": [input[f"investor_name_{i}"]() for i in range(n_investors)],
            "probabilities": np.array([number(f"probability_{i}") for i in range(n_states)]),
            "payoffs": np.array([[number(f"payoff_{i}_{j}") for j in range(n_assets)] for i in range(n_states)]),
            "holdings": np.array([[number(f"holding_{i}_{j}") for j in range(n_assets + 1)] for i in range(n_investors)]),
            "discount": np.array([number(f"discount_{i}") for i in range(n_investors)]),
            "utility_parameters": np.array([[number(f"utility_{i}_{j}") for j in range(3)] for i in range(n_investors)]),
        }

    @reactive.effect
    @reactive.event(input.add_asset)
    def add_asset():
        if simulate_task.status() == "running" or editable_case.get() is None:
            return
        try:
            # Read current controls first: rebuilding the tables must not lose edits.
            editor_draft.set(add_asset_to_draft(read_editor()))
        except ValueError as error:
            ui.notification_show(str(error), type="error", duration=8)

    @reactive.effect
    @reactive.event(input.add_state)
    def add_state():
        if simulate_task.status() == "running" or editable_case.get() is None:
            return
        try:
            editor_draft.set(add_state_to_draft(read_editor()))
        except ValueError as error:
            ui.notification_show(str(error), type="error", duration=8)

    @reactive.effect
    @reactive.event(input.add_investor)
    def add_investor():
        if simulate_task.status() == "running" or editable_case.get() is None:
            return
        try:
            editor_draft.set(add_investor_to_draft(read_editor()))
        except ValueError as error:
            ui.notification_show(str(error), type="error", duration=8)

    @reactive.effect
    @reactive.event(input.remove_asset)
    def remove_asset():
        if simulate_task.status() == "running" or editable_case.get() is None:
            return
        try:
            editor_draft.set(remove_last_asset_from_draft(read_editor()))
        except ValueError as error:
            ui.notification_show(str(error), type="error", duration=8)

    @reactive.effect
    @reactive.event(input.remove_state)
    def remove_state():
        if simulate_task.status() == "running" or editable_case.get() is None:
            return
        try:
            editor_draft.set(remove_last_state_from_draft(read_editor()))
        except ValueError as error:
            ui.notification_show(str(error), type="error", duration=8)

    @reactive.effect
    @reactive.event(input.remove_investor)
    def remove_investor():
        if simulate_task.status() == "running" or editable_case.get() is None:
            return
        try:
            editor_draft.set(remove_last_investor_from_draft(read_editor()))
        except ValueError as error:
            ui.notification_show(str(error), type="error", duration=8)

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
            ui.notification_show("Select Load Case to load the newly selected case before running it.",
                                 type="warning", duration=None)
            return
        try:
            if base is None:
                case = get_case(case_number)
            else:
                case = draft_to_case(base, read_editor())
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
                ui.p(f"Editing Case {editing_number.get()}. Edit the values and asset names in the tables. "
                     "The tables use full precision; market totals are calculated automatically. "
                     "Click Load Case again to reload the selected case.", class_="hint mt-3"),
                ui.p("New states start with zero probability and payoffs. Before running, "
                     "set positive probabilities that add to 1 and payoffs that give valid consumption.",
                     class_="hint"),
                editable_tables(editor_draft.get()),
            )
        result = result_value.get()
        if result is None:
            return ui.p("Click Load Case to load and edit the selected case, "
                        "or run it directly to view its inputs.", class_="hint mt-4")
        return table_cards(result["tables"], INPUT_TABLES, input.digits())

    @render.ui
    def equilibrium_tables():
        return ui.TagList(
            table_cards(calculation()["tables"], EQUILIBRIUM_TABLES, input.digits()),
            ui.p("A dash indicates an undefined value, such as the return on a zero-price asset.", class_="hint"),
        )

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
        fig = Figure(figsize=(7, 4))
        ax = fig.subplots()
        for i, label in enumerate(result["case"].asset_labels):
            if not np.isfinite(data["beta"][i]) or not np.isfinite(data["expected"][i]):
                continue  # A zero-price asset has no defined return or beta.
            ax.plot(data["beta"][i], data["expected"][i], marker=["o", "s", "D", "^", "v"][i % 5],
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
        fig = Figure(figsize=(7, 4))
        ax = fig.subplots()
        ax.plot(data["market_return"], data["ppc"], "o-", linewidth=2)
        ax.set(xlabel="Market Return", ylabel="PPC")
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        return fig


app = App(app_ui, server)
