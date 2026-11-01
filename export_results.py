"""Write the tables from a completed simulation to a single CSV file."""

import csv
import io
from pathlib import Path
import sys


# These lists also define the order of the tables in the Shiny interface.
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

EXPORT_PATH = Path(__file__).resolve().parent / "apsim_results.csv"


def results_csv(result, digits):
    """Return titled tables as CSV text formatted to the selected precision.

    Each table starts with its title and is followed by its column headings and
    values. A blank row separates tables. State-claim results are included only
    if the optional second stage was run. The input tables come from the actual
    case used in the simulation, rather than from the current editor contents.
    """
    tables = result["tables"]
    sections = [(title, tables[title]) for title in
                INPUT_TABLES + EQUILIBRIUM_TABLES + STATE_TABLES]
    if result["claims_tables"] is not None:
        sections.extend(result["claims_tables"].items())

    # Use the same text for both the local file and browser download. Pandas
    # formats floats like the on-screen tables and quotes labels with commas.
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    for title, table in sections:
        writer.writerow([title])
        table.to_csv(output, index=True, lineterminator="\n", na_rep="",
                     float_format=lambda number: f"{number:.{digits}f}")
        writer.writerow([])
    return output.getvalue()


def export_results(result, digits, path=EXPORT_PATH):
    """Overwrite a local file with the results, using Excel-friendly UTF-8."""
    if sys.platform == "emscripten":
        raise OSError("A browser-only app cannot write to the project folder.")
    text = results_csv(result, digits)
    path = Path(path)
    path.write_text(text, encoding="utf-8-sig", newline="")
    return path
