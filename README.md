# APSIM (Asset Pricing Simulator)

This project is an implementation of the APSIM program corresponding to the book
*Investors and Markets* by William F. Sharpe. It simulates trading between
investors and displays equilibrium prices, consumption and returns.

## Files

Keep these six files together in the same project folder as this README:

- `cases.py`: assumptions for Cases 1, 2, 3 and 6.
- `case_editor.py`: conversion and validation for editable case inputs.
- `simulation.py`: trading procedures, financial calculations and tables.
- `app.py`: the Shiny interface, including controls, tables and charts.
- `main.py`: command-line case selection, output and plots.
- `requirements.txt`: Python packages needed to run the program.

The preset cases are included in `cases.py`; no spreadsheets or other project
files are required to install or run the app.

## Install in a new environment

The following instructions use Windows PowerShell. You need an internet
connection to download Python and the packages during setup.

1. Install Python 3.12 or 3.13 from [python.org](https://www.python.org/downloads/)
   if it is not already installed. On Windows, select **Add Python to PATH** in
   the installer. Then open a new PowerShell window.
2. Create a project folder and put the six files listed above and this README
   in it. In PowerShell, change to that folder, replacing the example path with
   your own:

   ```powershell
   cd "C:\path\to\market_simulator"
   ```

3. Create a virtual environment in the project folder:

   ```powershell
   py -3.12 -m venv .venv
   ```

   If you installed Python 3.13 instead, use `py -3.13 -m venv .venv`. You can
   use `py --list` to see the Python versions available to the Windows launcher.
4. Install the required packages into that environment:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

The commands below use the environment's Python directly, so you do not need
to activate it. The `.venv` folder is created by the installation and does not
need to be copied from another computer. Shiny for Python is installed by the
command above; a separate R installation is not needed.

## Run the Shiny app on the computer

In PowerShell, change to the project folder if you are not already there, then
run:

```powershell
.\.venv\Scripts\python.exe -m shiny run --host 127.0.0.1 --port 8000 app.py
```

Open <http://127.0.0.1:8000/> in a browser. Keep PowerShell open while using the
app; press **Ctrl+C** in the terminal to stop it. If port 8000 is already in use,
replace `--port 8000` with `--port 8001` and open <http://127.0.0.1:8001/>.
This command runs a local Python server; it does not publish a website.

Choose a case and click **Run simulation**. To edit its assumptions, click
**Load Case**, change the inputs, and then run the simulation. Loading a case
again discards its edits. Edits are temporary and are lost when the page is
reloaded.

In the case editor, asset names are edited in the payoff table and investor
names in the initial portfolios table; the corresponding names are updated in
the other tables. You can add or remove assets, states and investors using the
buttons below the tables. The limits are one to six assets, two to ten states,
and two to five investors. New states start with zero probability and payoffs:
before running, give each state a positive probability, make the probabilities
sum to one, and ensure consumption is valid for each investor in every state.
After removing a state, adjust the probabilities to sum to one.

Select **Trade state claims after equilibrium** before running to see additional
results in the State-claim trading tab. Changing the number of decimal places
updates tables without rerunning the simulation. A dash indicates an undefined
value, such as the return on a zero-price asset.

While a simulation runs, the case inputs and the Run simulation and Load Case
buttons are disabled. **Stop Simulation** stops the run without displaying
partial results. The app also stops a run that exceeds two minutes, including
any state-claim trading. After 30 seconds, the convergence tolerance may be
relaxed by at most 50%; the app reports when this was needed.

## Run from the command line

From the same project folder, run:

```powershell
.\.venv\Scripts\python.exe main.py
```

Enter 1, 2, 3 or 6 when prompted. Alternatively, specify the case and options
directly, for example:

```powershell
.\.venv\Scripts\python.exe main.py --case 6 --state-claims --digits 4
```

Use `--no-plots` to suppress figure windows. Close the figure windows to return
to the terminal. Rounding affects displayed values, not calculations.

## Utility functions and input checks

An investor's utility parameters are `[type, parameter 1, parameter 2]`.
The supported types are 1 (quadratic), 2 (CRRA) and 4 (DRRA). For CRRA and
DRRA, parameter 1 must be greater than zero and at most 10, excluding 1.
Consumption must be positive for CRRA and above the shift parameter for DRRA.
Security payoffs must be non-negative. The case editor checks inputs before a
simulation is run.
