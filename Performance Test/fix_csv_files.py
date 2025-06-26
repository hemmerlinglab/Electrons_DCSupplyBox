import pandas as pd
import glob

def fix_ch_file(filename):
    df = pd.read_csv(filename)

    input_col = "Artiq Input (V)"
    input_values = df[input_col].tolist()

    # Fix second -5.0
    indices = [i for i, v in enumerate(input_values) if v == -5.0]
    if len(indices) >= 2:
        df.at[indices[1], input_col] = -4.0

    # Fix second 9.0
    indices = [i for i, v in enumerate(input_values) if v == 9.0]
    if len(indices) >= 2:
        df.at[indices[1], input_col] = 9.9

    df.to_csv(filename, index=False)
    print(f"Fixed: {filename}")

# 修复所有 ch*.csv 文件
for file in glob.glob("ch*.csv"):
    fix_ch_file(file)
