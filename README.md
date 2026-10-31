# Financial market simulator

This project translates built-in cases from the MATLAB live script.
The Shiny app runs locally in a browser. The calculation code is separate from
the interface; publishing the app with Shinylive and GitHub Pages is a later step.

## Files

- `cases.py`: assumptions for Cases 1, 2, 3 and 6.
- `case_editor.py`: conversion and validation for editable case inputs.
- `simulation.py`: trading procedures, financial calculations and tables.
- `app.py`: the Shiny interface, including controls, tables and charts.
- `main.py`: case selection, printing and plots.
- `requirements.txt`: Python packages needed for local execution.

## Run the Shiny app on this computer

The existing environment is in the parent `Test` folder. Open PowerShell in
the `market_simulator` folder and run:

```powershell
..\.venv\Scripts\python.exe -m shiny run --host 127.0.0.1 --port 8000 app.py
```

Open http://127.0.0.1:8000/ in a browser. Choose a case and click **Run
simulation**. To compare both cases, change the selector and click Run again.
While a simulation is running, **Stop Simulation** cancels it without showing
partial results. The Run and Modify buttons are unavailable until it finishes
or stops, and the case input tables are locked without losing existing edits.
The app shows elapsed time and automatically stops without results
if the complete run exceeds two minutes, including state-claim trading.
To change assumptions, choose a case and click **Load Case**. Edit the input
tables and then click **Run simulation**. Investor names are edited in the
portfolio table and mirrored in the preferences table. Asset names can be edited in the payoff table header and are
mirrored in the portfolio table. Click **Add Asset** to add up to six assets;
the new asset starts with zero payoffs and holdings. **Add State** adds up to
ten states, with a zero probability and zero payoffs initially. Before running,
give every state a positive probability, make probabilities sum to one, and
ensure investors have valid consumption in each state. **Add Investor** adds a
new investor with the displayed default holdings and preferences, up to five
investors. The corresponding **Remove Last** buttons remove the last asset,
state or investor. At least one asset, two states and two investors must remain.
After removing a state, adjust probabilities to sum to one before running.
A dash in the results
indicates an undefined value, for example the return on a zero-price asset.
Clicking **Load Case** again reloads the selected case and discards edits.
The edits are temporary and are not saved when the page is reloaded.
Select **Trade state claims after equilibrium** and run again to see the
additional tables in the State-claim trading tab. Changing decimal places
updates tables without rerunning the model. Press Ctrl+C in the terminal to stop
the app. If port 8000 is busy, change `--port 8000` to `--port 8001` and open
http://127.0.0.1:8001/ instead.

This local Shiny run uses a Python server on your own computer. It is not yet the
browser-only Shinylive export. Keep the terminal running while using the app.

## Install in a new environment

Use a current stable Python installation (Python 3.12 or 3.13 recommended).
Open PowerShell in this folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

If Python 3.12 is not installed, replace `-3.12` with `-3.13`, if available.
The last command asks you to enter 1 or 2. Alternatively:

```powershell
.\.venv\Scripts\python.exe main.py --case 1
.\.venv\Scripts\python.exe main.py --case 2 --state-claims --digits 4
```

Use `--no-plots` to suppress figure windows. Close the figure windows to return
to the terminal. Rounding affects the displayed values, not the calculations.

The algorithm follows the live script's grid search and stopping criteria.
It excludes grid candidates outside the selected utility function's domain.
The default stopping limit is two minutes rather than a fixed number of trades.
Utility gains are calculated directly to preserve accuracy for small trades.
After 30 seconds of total runtime, the convergence tolerance may increase by
at most 50%; the app reports when this relaxed tolerance was needed.
The project owner has compared the translated results with MATLAB. Further
changes to the model should be checked against the original results again.

Each investor's utility parameter row is `[type, parameter 1, parameter 2]`.
The model supports the revised MATLAB utility types: 1 (quadratic), 2 (CRRA),
and 4 (DRRA). Types 2 and 4 require positive consumption relative to
their respective zero or shift. Their utility parameter 1 must be greater
than 0 and at most 10, excluding 1 because the utility formula is undefined
there. Security payoffs must be non-negative. Both built-in cases use type 2.

Run the utility and simulation checks from this folder with:

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
