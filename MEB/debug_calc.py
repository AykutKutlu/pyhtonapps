import pandas as pd
from io import BytesIO
from meb_calculator import calculate_meb

# Sample prices file yükle
excel_file = pd.ExcelFile('sample_prices.xlsx')
print("Excel sheet names:", excel_file.sheet_names)

df = pd.read_excel('sample_prices.xlsx')
print("\nDataFrame info:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst row:")
print(df.iloc[0])

# Hesapla
with open('sample_prices.xlsx', 'rb') as f:
    result = calculate_meb(f, 'ESSN')

print("\n" + "="*70)
print("CALCULATION RESULT")
print("="*70)
print("\nColumns:", list(result.columns))
print("\nIndex:", list(result.index))
print("\nFirst few rows:")
print(result.iloc[:3, :])

print("\n\nDetailed results:")
for idx in result.index:
    print(f"\n{idx}:")
    for col in result.columns:
        val = result.loc[idx, col]
        print(f"  {col}: {val}")
