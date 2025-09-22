import streamlit as st
import pandas as pd
import os
import re

# Set the page title and layout
st.set_page_config(page_title="Fiyat Hesaplama Dashboard", layout="wide")

# --- Data Loading and Wrangling ---
# Ensure Excel files exist
excel_file1_path = "harcama gruplarina gore endeks sonuclari (5).xlsx"
excel_file2_path = "pivot.xlsx"

if not os.path.exists(excel_file1_path):
    st.error(f"Error: The file '{excel_file1_path}' was not found.")
    st.stop()

if not os.path.exists(excel_file2_path):
    st.error(f"Error: The file '{excel_file2_path}' was not found.")
    st.stop()

try:
    tuik_grup = pd.read_excel(excel_file1_path, engine='openpyxl')
    tuik_urun = pd.read_excel(excel_file2_path, engine='openpyxl')
except Exception as e:
    st.error(f"An error occurred while reading the Excel files: {e}")
    st.stop()

# Data Cleaning (tuik_urun)
# Corrected: Drop the first column to match the R code's behavior
# Slicing the last 3 columns to handle any extra empty columns in the excel file
tuik_urun = tuik_urun.iloc[3:412, -3:]
tuik_urun.columns = ["X.1", "X.2", "X.3"]
tuik_urun = tuik_urun.drop(columns="X.2")
tuik_urun[['kod', 'item']] = tuik_urun['X.1'].str.split(r'\.\s*', n=1, expand=True)
tuik_urun['item'] = tuik_urun['item'].str.replace(r'[()]', '', regex=True)
tuik_urun = tuik_urun.rename(columns={"X.3": "price"})
tuik_urun['price'] = pd.to_numeric(tuik_urun['price'], errors='coerce')
tuik_urun = tuik_urun.drop(columns="X.1")
tuik_urun = tuik_urun.dropna(subset=['price'])

# Data Cleaning (tuik_grup)
date_column = pd.DataFrame(pd.date_range(start="2005-01-01", end="2025-08-01", freq="MS"), columns=['Date'])
tuik_grup1 = tuik_grup.copy()
tuik_grup1.columns = tuik_grup1.iloc[1]
tuik_grup1 = tuik_grup1.iloc[4:].reset_index(drop=True)
tuik_grup1 = tuik_grup1.iloc[:, 3:]

cols_to_keep = [col for col in tuik_grup1.columns if re.match(r'^\d{5}$', str(col))]
tuik_grup2 = tuik_grup1[cols_to_keep]

tuik_grup2 = pd.concat([date_column, tuik_grup2], axis=1)
tuik_grup2 = tuik_grup2[tuik_grup2['Date'] > pd.Timestamp("2022-03-01")]
tuik_grup2.iloc[:, 1:] = tuik_grup2.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

tuik_grup2_new = tuik_grup2.copy()
tuik_grup2_new.iloc[1:, 1:] = tuik_grup2_new.iloc[1:, 1:].values / tuik_grup2_new.iloc[:-1, 1:].values
tuik_grup2_new = tuik_grup2_new.iloc[1:]

tuik_grup2_new_transposed = tuik_grup2_new.set_index('Date').T.reset_index()
tuik_grup2_new_transposed = tuik_grup2_new_transposed.rename(columns={'index': 'kod'})

# Correcting specific codes based on the R script
tuik_urun['kod_5hane'] = tuik_urun['kod'].str[:5]
tuik_urun = tuik_urun.dropna()
tuik_urun.loc[tuik_urun['item'] == "Kiracı Tarafından Ödenen Gerçek Kira", 'kod_5hane'] = "04111"
tuik_urun.loc[tuik_urun['item'] == "Pamuklu Kumaş", 'kod_5hane'] = "03130"

# --- Corrected Calculation Logic ---
merged_data = pd.merge(tuik_urun, tuik_grup2_new_transposed, left_on="kod_5hane", right_on="kod", how="left")
merged_data['price'] = pd.to_numeric(merged_data['price'], errors='coerce')
merged_data = merged_data.dropna(subset=['price'])

cols_to_multiply = [col for col in tuik_grup2_new_transposed.columns if col != 'kod']

current_price = merged_data['price'].copy()
for col_name in cols_to_multiply:
    current_price = current_price * merged_data[col_name]

final_df = pd.DataFrame({
    'item': merged_data['item'],
    'price': current_price
})

# --- Streamlit App ---
st.title("Fiyat Hesaplama Dashboard")
st.markdown("---")

# Initialize session state for the selected items table
if 'selected_data' not in st.session_state:
    st.session_state.selected_data = pd.DataFrame(columns=["Goods and Services", "Price", "Total Price"])

# Sidebar for user input
with st.sidebar:
    st.header("Seçim Paneli")
    selected_item = st.selectbox("Mal ve Hizmetler:", final_df['item'].tolist())
    multiplier = st.number_input("Adet:", value=1.0, min_value=0.0)

    col1, col2 = st.columns(2)
    with col1:
        add_button = st.button("Ekle")
    with col2:
        calculate_button = st.button("Hesapla")

    st.markdown("---")
    st.subheader("Seçilen Mal ve Hizmetler")
    st.table(st.session_state.selected_data)

# Logic for "Ekle" (Add) button
if add_button:
    new_row = final_df[final_df['item'] == selected_item].copy()
    new_row.columns = ["Goods and Services", "Price"]
    new_row["Total Price"] = new_row["Price"] * multiplier
    
    # Check if item is already in the table to avoid duplicates
    if selected_item not in st.session_state.selected_data["Goods and Services"].values:
        st.session_state.selected_data = pd.concat([st.session_state.selected_data, new_row], ignore_index=True)
    else:
        st.warning(f"'{selected_item}' zaten listede.")

# Logic for "Hesapla" (Calculate) button
if calculate_button:
    if not st.session_state.selected_data.empty:
        updated_df = st.session_state.selected_data.copy()
        updated_df['Total Price'] = updated_df['Price'] * multiplier

        total_sum = updated_df['Total Price'].sum()
        total_row = pd.DataFrame([["Toplam", None, total_sum]], columns=["Goods and Services", "Price", "Total Price"])
        
        st.session_state.selected_data = pd.concat([updated_df, total_row], ignore_index=True)
        st.success("Hesaplama tamamlandı!")

# Main panel for output
st.header("Hesaplanan Fiyat")
st.table(st.session_state.selected_data)
