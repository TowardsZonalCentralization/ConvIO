# convert_to_positive.py
import pandas as pd
import os

def convert_negative_to_positive(input_file: str, output_file: str) -> None:
    """
    Convert all negative values in a CSV to positive and save to a new file.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")
    
    df = pd.read_csv(input_file)
    df = df.abs()  # Convert negative to positive
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved cleaned data to {output_file}")

if __name__ == "__main__":
    input_file = 'k-means/data/cluster_data.csv'
    output_file = 'k-means/data/cluster_data_positive.csv'
    convert_negative_to_positive(input_file, output_file)
