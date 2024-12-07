import os
import pandas as pd

input_file_path = r"C:\Users\Deshan\Documents\IIT LECS\Year 2 Sem 1\DSGP\dataset creation\ebd_LK-33_revbul_202201_202301_unv_smp_relOct-2024.txt"
output_file_path = r"C:\Users\Deshan\Documents\IIT LECS\Year 2 Sem 1\DSGP\dataset creation\raw_dataset.csv"

try:
    df = pd.read_csv(input_file_path, sep="\t", header=0, low_memory=False)
    df.to_csv(output_file_path, index=False)

    print(f"Raw dataset saved successfully at {output_file_path}")
    
except Exception as e:
    print(f"An error occurred: {e}")
