from __future__ import annotations

from pathlib import Path
import pandas as pd

# This script extracts each resource row's data from the EDI Hub+ Excel spreadsheet 
# and converts it into a clean CSV file for downstream processing.

# CONFIGURATION (file paths and schema)

# Excel sheet file path containing all row data.
EXCEL_PATH = Path("data/raw/Resource centre taxonomy and resources_wd_macros.xlsm")

# Full resource sheet.
SHEET_NAME = "5_Resources_full_data"

# Output path to store final cleaned CSV file.
OUT_CSV = Path("data/raw/resources_full_data_full.csv")

# Defined document headers for correct parsing and extraction of relevant columns.
HEADERS = [
    "ID",
    "Example",
    "Title",
    "Author",
    "Institution",
    "Link",
    "LinkIsAccesible",
    "Included_Excluded",
    "Reason for Exclusion (if applicable)",
    "Date Accessed",
    "Date Created",
    "Accountable Person",
    "Summary (will be generated using AI)",
    "Subject of Resource (Archived)",
    "Type of Resource (Archived)",
    "Resource Subject (Modified)",
    "Resource Type (Modified)",
    "Resource Format (Modified)",
    "Sector",
    "Level",
    "Discipline (Archived)",
    "Individual Characteristics",
    "Career Pathway",
    "Research Funding Process",
    "Organisational Culture",
    "Miscellaneous Themes",
    "Evaluated in Resource",
    "Evaluation Link",
    "Evalution Methods",
    "Outcomes Measured",
    "EDI Intervention Readiness Level",
    "EDI Intervention Readiness (Archive)",
]

# Detect the row index of the header row and scan for defined column names.
# This avoids relying on fixed row positions, enabling robust extraction.
def find_header_row(df_raw: pd.DataFrame) -> int:
    # Go through each row to detect its header title.
    for i in range(len(df_raw)):
        # Convert row to string values and remove extra whitespace
        row = df_raw.iloc[i].astype(str).str.strip()
        row = row.replace("nan", "")
        # Check if this row matches defined header structure
        if row.iloc[0] == "ID" and ("Title" in row.values) and ("Link" in row.values):
            return i
    raise ValueError(
        "Could not detect the header row containing 'ID', 'Title', and 'Link'. "
        "The sheet may be formatted unexpectedly."
    )

def main() -> None:
    # Ensure the input Excel file exists before processing
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Excel file not found: {EXCEL_PATH}")

    # Create output directory if it does not already exist
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Load the Excel sheet without assuming a header row
    df_raw = pd.read_excel(
        EXCEL_PATH,
        sheet_name=SHEET_NAME,
        header=None,
        engine="openpyxl",
        dtype=object,
    )

    # Remove completely empty rows to clean the raw input
    df_raw = df_raw.dropna(how="all")

    # Dynamically locate the header row in the sheet
    header_idx = find_header_row(df_raw)

    # Extract all rows after the header (actual data)
    df_data = df_raw.iloc[header_idx + 1 :].copy()

    # Trim columns to match the expected schema length
    df_data = df_data.iloc[:, : len(HEADERS)]

    # Assign standardised column names
    df_data.columns = HEADERS

    # Remove rows where ID is missing (invalid or empty entries)
    df_data = df_data.dropna(subset=["ID"], how="all")

    # Save the cleaned dataset to CSV for downstream use
    df_data.to_csv(OUT_CSV, index=False)

    # Print summary information for verification/logging
    print(" Export complete")
    print(f"Sheet: {SHEET_NAME}")
    print(f"Rows exported: {len(df_data)}")
    print(f"CSV: {OUT_CSV}")

# Run the extraction when the script is executed directly
if __name__ == "__main__":
    main()