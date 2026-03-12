import pandas as pd
from signal_detection import calculate_prr

def main():
    test_df = pd.DataFrame(
        {
            "A": [10, 20],
            "B": [190, 380],
            "C": [50, 100],
            "D": [9750, 19500],
        }
    )

    prr_series = calculate_prr(test_df)
    print(prr_series)

if __name__ == "__main__":
    main()

