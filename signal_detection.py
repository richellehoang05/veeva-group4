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


def calculate_prr(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the Proportional Reporting Ratio (PRR).

    PRR is a measure of disproportionality used in pharmacovigilance. It
    compares the proportion of reports for a given event with a specific drug
    to the proportion of reports for the same event with all other drugs.

    Formula:
        PRR = (A / (A + B)) / (C / (C + D))

    Where:
        A, B, C, D are the values from the 2x2 contingency table.
        
    Example:
        A = 10 (Drug X, Event Y)
        B = 190 (Drug X, Other Events)
        C = 50 (Other Drugs, Event Y)
        D = 9750 (Other Drugs, Other Events)
        
        Proportion for Drug X = 10 / (10 + 190) = 0.05
        Proportion for Other Drugs = 50 / (50 + 9750) = 0.0051
        PRR = 0.05 / 0.0051 ≈ 9.8
    """
    validate_required_cols(df, {'A', 'B', 'C', 'D'})
    # --- Implementation: vectorized PRR calculation ---
    A = df['A']
    B = df['B']
    C = df['C']
    D = df['D']

    # Proportions for the drug and for all other drugs
    drug_prop = A / (A + B)
    others_prop = C / (C + D)

    # PRR = (A / (A + B)) / (C / (C + D))
    prr = drug_prop / others_prop

    return prr


def calculate_chi_squared(df: pd.DataFrame) -> pd.Series:
    """
    Calculates the Chi-Squared (χ²) statistic.

    The Chi-Squared test is used to assess the statistical significance of the
    association between a drug and an adverse event. A higher Chi-Squared
    value indicates a stronger association that is less likely to be due to
    random chance.

    Formula:
        χ² = N * (|AD - BC| - N/2)² / ((A+B)(C+D)(A+C)(B+D))

    Where:
        N = Total number of reports (A + B + C + D)
        A, B, C, D are the values from the 2x2 contingency table.
        
    Example (using the same numbers as PRR):
        A=10, B=190, C=50, D=9750
        N = 10 + 190 + 50 + 9750 = 10000
        
        Numerator = 10000 * (|10*9750 - 190*50| - 10000/2)²
        Denominator = (10+190) * (50+9750) * (10+50) * (190+9750)
        χ² = 58.93
    """
    validate_required_cols(df, {'A', 'B', 'C', 'D'})
    # --- TODO: YOUR CODE HERE ---
    A = df['A']
    B = df['B']
    C = df['C']
    D = df['D']
    N = df[['A', 'B', 'C', 'D']].sum(axis=1)
    AD = A * D
    BC = B * C
    Numerator = N * ((AD - BC).abs() - N/2).pow(2)
    A_plus_B = df[['A', 'B']].sum(axis=1)
    C_plus_D = df[['C', 'D']].sum(axis=1)
    A_plus_C = df[['A', 'C']].sum(axis=1)
    B_plus_D = df[['B', 'D']].sum(axis=1)
    Denom = (A_plus_B) * (C_plus_D) * (A_plus_C) * (B_plus_D)
    stat = Numerator / Denom
    # ----------------------------
    return stat


def calculate_statistics(df: pd.DataFrame, background_incidence_path: str) -> pd.DataFrame:
    """
    Calculates PRR and Chi-Squared statistics for all drug-event pairs.

    This function takes a DataFrame of adverse event reports and performs
    vectorized calculations to determine the PRR and Chi-Squared values for
    every unique drug-event combination found in the data.

    Contingency Table for a given drug-event pair:
                   | Event      | No Event
    -----------------------------------------
    Drug           |     A      |     B
    -----------------------------------------
    Other Drugs    |     C      |     D

    A = Count of reports with the specific drug and specific event.
    B = Count of reports with the specific drug and any other event.
    C = Count of reports with any other drug and the specific event.
    D = Count of reports with any other drug and any other event.
    """
    results_df = df.copy()  # Start with a copy of the original DataFrame
    background_incidence = load_data(background_incidence_path)

    # --- TODO: YOUR CODE HERE ---
    # 1. Count occurrences for each drug-event pair (A)
    # Assuming the raw data has 'drug_name' and 'event_term' columns
    results_df = df.groupby(['drug_name', 'event_term']).size().reset_index(name='A')

    # 2. Calculate marginal totals to derive B, C, and D
    # Total reports for each drug (A + B)
    drug_totals = df.groupby('drug_name').size().reset_index(name='drug_total')

    # Total reports for each event (A + C)
    event_totals = df.groupby('event_term').size().reset_index(name='event_total')

    # Total number of all reports (N = A + B + C + D)
    n_total = len(df)

    # 3. Merge totals back into our results table
    results_df = results_df.merge(drug_totals, on='drug_name')
    results_df = results_df.merge(event_totals, on='event_term')

    # 4. Derive B, C, and D based on the marginals
    # B = (Total reports for drug) - (Reports with this event)
    results_df['B'] = results_df['drug_total'] - results_df['A']

    # C = (Total reports for event) - (Reports with this drug)
    results_df['C'] = results_df['event_total'] - results_df['A']

    # D = (Total population) - (A + B + C)
    results_df['D'] = n_total - (results_df['A'] + results_df['B'] + results_df['C'])
    # ----------------------------
    results_df['prr'] = calculate_prr(results_df)
    results_df['chi_squared'] = calculate_chi_squared(results_df)
    results_df['case_count'] = results_df['A']
    return results_df


def analyze_signals(df: pd.DataFrame, drug_profiles_path: str) -> pd.DataFrame:
    """
    Identifies statistically significant signals (PRR >= 2, X2 >= 4) and cross-references them
    with known drug safety information to determine whether these signals require further investigation.
    The drug safety information lists what the drug is used to treat and it's known side effects.

    This function should modify output a copy of the input dataframe
    with an appended "requires_investigation" column with boolean values
    """
    validate_required_cols(df, {'prr', 'chi_squared', 'drug_name', 'event_term'})
    results_df = df.copy()  # Start with a copy of the original DataFrame
    drug_profiles = pd.read_json(drug_profiles_path)

    # --- TODO: YOUR CODE HERE ---
    def is_investigation_required(row):
        # 1. Check statistical significance thresholds
        # PRR must be >= 2.0 and Chi-Squared >= 4.0
        statistically_sig = (row['prr'] >= PRR_THRESHOLD) and (row['chi_squared'] >= CHI_SQUARED_THRESHOLD)

        if not statistically_sig:
            return False

        # 2. Get the profile for the specific drug
        # We match the drug name from our results to the 'drug_name' in the JSON
        drug_info = drug_profiles[drug_profiles['drug_name'].str.upper() == str(row['drug_name']).upper()]

        if not drug_info.empty:
            profile = drug_info.iloc[0]
            # Standardize terms to lower case for accurate comparison
            current_event = str(row['event_term']).lower()
            known_side_effects = [str(se).lower() for se in profile.get('side_effects', [])]
            known_indications = [str(ind).lower() for ind in profile.get('indications', [])]

            # 3. If the event is already a known side effect or an indication,
            # it is not a "new" signal requiring investigation
            if current_event in known_side_effects or current_event in known_indications:
                return False

        return True

    # Create the column required by the main() function's final steps
    results_df['requires_review'] = results_df.apply(is_investigation_required, axis=1)
    # ----------------------------

    return results_df


def validate_required_cols(df: pd.DataFrame, required_cols: set):
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame is missing required columns. Must contain: {list(required_cols)}")


def report_to_health_authority(suspect_pairs: list):
    """
    Submits your final list of top outlier signals to the Health Authority.
    The Authority only has resources to investigate the most severe, undeniable statistical anomalies.

    Expected format: suspect_pairs = [("Drug A", "Event X"), ("Drug B", "Event Y"), ("Drug C", "Event Z")]
    """
    print("\n" + "=" * 60)
    print("📡 TRANSMITTING FINDINGS TO HEALTH AUTHORITY...")
    print("=" * 60)
    clean_pairs = []
    for drug, event in suspect_pairs:
        clean_pairs.append(f"{str(drug).strip().upper()}:{str(event).strip().upper()}")

    clean_pairs = sorted(list(set(clean_pairs)))

    submission_string = "|".join(clean_pairs)
    submission_hash = hashlib.sha256(submission_string.encode('utf-8')).hexdigest()
    TARGET_HASH = "YOUR_GENERATED_HASH_GOES_HERE"

    fig = Figlet(font='standard')
    if submission_hash == TARGET_HASH:
        print(f"{Fore.GREEN}SUCCESS: The Health Authority has verified your statistical anomalies.{Style.RESET_ALL}")
        print(f"Based on your compelling data, emergency safety reviews have been triggered for:")
        for pair in clean_pairs:
            d, e = pair.split(':')
            print(f" 🚨 {Fore.CYAN}{d.title()}{Style.RESET_ALL} -> {Fore.RED}{e.capitalize()}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}Incredible work, investigator.{Style.RESET_ALL}\n")
    else:
        print(f"\n{Fore.YELLOW}{fig.renderText('SUBMISSION REJECTED')}{Style.RESET_ALL}")
        print(f"{Fore.RED}The Health Authority reviewed your list and found it lacking.{Style.RESET_ALL}")
        print("They stated: 'Your list either contains background noise, or you missed a major threat.'")


def main(
    ae_reports_path: str,
    background_incidence_path: str,
    drug_profiles_path: str,
    output_path: str,
):
    # 1. Load Data
    raw_df = load_data(ae_reports_path)

    # 2. Clean Data
    clean_df = clean_input_data(raw_df)

    # 3. Disproportionality calculations
    results_df = calculate_statistics(clean_df, background_incidence_path)

    # 4. Analyze signals
    final_df = analyze_signals(results_df, drug_profiles_path)

    # 5. Export statistically significant signals to a CSV
    final_df = final_df.sort_values(by=['prr', 'chi_squared'], ascending=False)
    final_df.to_csv(output_path, index=False)
    print(f"Results exported to {output_path}")

    # 6. Submit the top 3 outliers to health authorities
    validate_required_cols(final_df, {'prr', 'requires_review'})
    outliers_df = final_df[final_df['requires_review'] == True].head(3)
    suspect_pairs = list(zip(outliers_df['drug_name'], outliers_df['event_term']))
    report_to_health_authority(suspect_pairs)


if __name__ == "__main__":
    main(
        ae_reports_path="data/raw_ae_reports.csv",
        background_incidence_path="data/background_incidence.csv",
        drug_profiles_path="data/drug_safety_profiles.json",
        output_path="signal_detection_results.csv",
    )
