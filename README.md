# Financial market simulator

This project translates the two built-in cases in the MATLAB live script.
The Shiny app runs locally in a browser. The calculation code is separate from
the interface; publishing the app with Shinylive and GitHub Pages is a later step.

## Files

- `cases.py`: Case 1 and Case 2 assumptions.
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
It excludes grid candidates that would produce non-positive consumption, and
recomputes prices after the final trade if the iteration limit is reached.
The project owner has compared the translated results with MATLAB. Further
changes to the model should be checked against the original results again.
