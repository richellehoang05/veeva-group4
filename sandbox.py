import pandas as pd
import hashlib
from colorama import Fore, Style
from pyfiglet import Figlet

"""
Signal Detection in Pharmacovigilance

This script performs disproportionality analysis on a dataset of adverse event reports to identify potential signals.

The main steps are:
1.  Load and preprocess the adverse event data.
2.  For each pair, construct a 2x2 contingency table.
3.  Calculate disproportionality measures:
    - Proportional Reporting Ratio (PRR)
    - Chi-Squared (χ²) statistic
4.  Identify statistically significant signals based on predefined thresholds.
5.  Cross-reference signals with known drug safety profiles to distinguish
    between new potential risks and known indications and side effects.
6.  Export the results to a CSV file.
7.  Submit the top 3 Signals requiring review to the Health Authority.
"""

PRR_THRESHOLD = 2.0
CHI_SQUARED_THRESHOLD = 4.0


def load_data(file_path: str) -> pd.DataFrame:
    """Loads data from a CSV file into a pandas DataFrame."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    return df


def clean_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs basic cleaning and validation of the input DataFrame.
    
    TODO: Validate and clean/normalize the data
         - Validation ex: are there impossible patient ages, incorrect dates, etc.
         - Normalization ex: standardize text to lower case, remove leading/trailing white space, etc.
    """
    cleaned_df = df.copy()  # Start with a copy of the original DataFrame
    # --- TODO: YOUR CODE HERE ---
    
    # remove white space
    cleaned_df = cleaned_df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # convert drug names to title case
    cleaned_df["drug_name"] = cleaned_df["drug_name"].str.title()

    # event_term: 
    # first term should be captilizaed, the rest lower case
    cleaned_df["event_term"] = cleaned_df["event_term"].str.lower().str.capitalize()

    # safety_report_case_number
    # convert to ints
    cleaned_df["safety_report_case_number"] = pd.to_numeric(cleaned_df["safety_report_case_number"], errors="coerce").astype("Int64")

    # patient_age
    cleaned_df["patient_age"] = pd.to_numeric(cleaned_df["patient_age"], errors="coerce").astype("Int64")

    # drop invalid values from dataset
    cleaned_df = cleaned_df.dropna(subset=["patient_age", "safety_report_case_number"])

    # drop invalid ages
    cleaned_df = cleaned_df[(cleaned_df["patient_age"] >= 0) & (cleaned_df["patient_age"] <= 120)]

    # patient_gender 
    # captalize
    cleaned_df["patient_gender"] = cleaned_df["patient_gender"].str.upper()

    # gender hould be either M U or F, drop values outside of this
    cleaned_df = cleaned_df[cleaned_df["patient_gender"].isin(["F", "M", "U"])]

    # report_receipt_date
    # convert to dates format
    cleaned_df["report_receipt_date"] = pd.to_datetime(cleaned_df["report_receipt_date"], errors="coerce")
    # drop invalid date values
    cleaned_df = cleaned_df.dropna(subset=["report_receipt_date"])

    # drop future dates
    cleaned_df = cleaned_df[cleaned_df["report_receipt_date"] <= pd.Timestamp.today()]
    
    # fix indexing 
    cleaned_df = cleaned_df.reset_index(drop=True)
    # ----------------------------
    return cleaned_df

if __name__ == "__main__":
    dataset = pd.read_csv("group_4_data/raw_ae_reports.csv", dtype=str, keep_default_na=False)
    cleaned_df = clean_input_data(dataset)
    cleaned_df.to_csv("output.csv")

