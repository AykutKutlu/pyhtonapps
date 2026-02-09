"""
MEB (Minimum Expenditure Basket) Calculation Tool
Streamlit web application for calculating Minimum Expenditure Basket based on TUIK indices
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows
import io

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="MEB Calculation Tool", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "MEB Calculation Tool • Built with Streamlit"
    }
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    h1 {
        color: #1f77b4;
    }
    h2 {
        color: #1f77b4;
        margin-top: 1.5rem;
    }
    .stTabs [role="tab"] {
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'merge_df' not in st.session_state:
    st.session_state.merge_df = None
if 'merge_df1' not in st.session_state:
    st.session_state.merge_df1 = None
if 'calculation_complete' not in st.session_state:
    st.session_state.calculation_complete = False
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'active_results_tab' not in st.session_state:
    st.session_state.active_results_tab = 0
if 'active_graphs_tab' not in st.session_state:
    st.session_state.active_graphs_tab = 0
if 'generate_graphs' not in st.session_state:
    st.session_state.generate_graphs = False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_excel_export(meb_results, all_components=None):
    """Create a formatted Excel file with multiple sheets matching R code logic"""
    excel_buffer = io.BytesIO()
    
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        # Sheet 1: Summary (Veri1 in R)
        meb_results.to_excel(writer, sheet_name="Summary")
        
        # Sheet 2: All Components (Veri2 in R)
        if all_components is not None:
            all_components.to_excel(writer, sheet_name="All_Components")
        
        # Sheet 3: Statistics
        stats_data = {
            "Metric": ["Mean", "Median", "Min", "Max", "Std Dev", "Growth %"],
            "Value": [
                meb_results["Total_MEB"].mean(),
                meb_results["Total_MEB"].median(),
                meb_results["Total_MEB"].min(),
                meb_results["Total_MEB"].max(),
                meb_results["Total_MEB"].std(),
                ((meb_results["Total_MEB"].iloc[-1] / meb_results["Total_MEB"].iloc[0]) - 1) * 100
            ]
        }
        pd.DataFrame(stats_data).to_excel(writer, sheet_name="Statistics", index=False)
        
        # Format workbook
        for worksheet in writer.book.worksheets:
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    excel_buffer.seek(0)
    return excel_buffer


def calculate_meb(tuik_data, selected_option):
    """
    MEB hesaplamasını baz fiyatlar (Aralık 2025) ve endeksler üzerinden gerçekleştirir.
    R kodundaki mantığa göre geliştirilmiştir.
    """
    try:
        # 1. TUIK Verisini Hazırlama
        tuik_data = tuik_data.loc[:, ~tuik_data.columns.str.contains('^Unnamed')]
        tuik_data.columns = ["Year", "Month", "Months"] + list(tuik_data.columns[3:])
        tuik_data = tuik_data[:-5]
        enflasyon2 = tuik_data.iloc[2:].reset_index(drop=True)
        
        start_date = datetime(2005, 1, 1)
        date_range = pd.date_range(start=start_date, periods=len(enflasyon2), freq='MS')
        enflasyon2['Date'] = date_range

        # Enflasyon Oranları Hesaplama (YoY - Yıllık)
        enf_sutun = ["0", "011", "04", "06", "07", "08", "10", "13"]
        available_enf_sutun = [col for col in enf_sutun if col in enflasyon2.columns]
        enf_index = enflasyon2[available_enf_sutun].apply(pd.to_numeric, errors='coerce')
        
        enf_rate = enf_index.pct_change(periods=12) # 12 months for YoY
        enf_rate.index = date_range
        enf_rate.columns = [f"Inf_{col}" for col in enf_rate.columns]

        # 2. Baz Fiyat Dosyalarını Yükleme
        if selected_option == "Turkish":
            base_prices_df = pd.read_excel("MEB_TR.xlsx")
        else:
            base_prices_df = pd.read_excel("MEB_SSN.xlsx")
            
        base_prices_df.columns = [str(c).strip() for c in base_prices_df.columns]
        name_col = next((col for col in ['Product_Name', 'Product Name', 'Ürün Adı', 'Urun Adi', 'Ürün'] if col in base_prices_df.columns), base_prices_df.columns[0])
        price_col = 'Base_Price' if 'Base_Price' in base_prices_df.columns else base_prices_df.columns[1]

        base_date = datetime(2025, 12, 1)
        
        # 3. Ürün Eşleştirme
        product_name_to_code = {
            "Rice": "01111", "Bulgur": "01111", "Bread": "01113", "Yoghurt": "01146",
            "White Cheese": "01145", "Egg": "01148", "Sun-Flower Oil": "01151",
            "Tomato": "01172", "Cucumber": "01172", "Dry Bean": "01176",
            "Granulated Sugar": "01181", "Salt": "01193", "Tea": "01230",
            "Notebook": "09740", "Pencil": "09740", "Other stationery": "09740",
            "Medicines": "06111", "Fees Paid To Specialist Doctor": "06231",
            "Actual Rent": "04110", "Water": "04411", "Electricity": "04510", "Tube Gas (12 L)": "04522",
            "Detergents (For Laundry)": "05611", "Dishwasher Detergents": "05611",
            "Disinfectants And Insecticidies": "05611", "Shaving Articles": "13120",
            "Articles For Dental Hygiene": "13120", "Bath Soap": "13120",
            "Hair Care Products": "13120", "Toilet Paper": "13120", "Baby Napkin": "13120",
            "Hygiene Pad For Women": "13120",
            "Transport": "07321", "Transport (Total)": "07321", "Communication": "08320"
        }

        category_map = {
            "Gıda": ["Rice", "Bulgur", "Bread", "Yoghurt", "White Cheese", "Egg", "Sun-Flower Oil", "Tomato", "Cucumber", "Dry Bean", "Granulated Sugar", "Salt", "Tea"],
            "Eğitim": ["Notebook", "Pencil", "Other stationery"],
            "Sağlık": ["Medicines", "Fees Paid To Specialist Doctor"],
            "Barınma": ["Actual Rent", "Water", "Electricity", "Tube Gas (12 L)"],
            "NFI": ["Detergents (For Laundry)", "Dishwasher Detergents", "Disinfectants And Insecticidies", "Shaving Articles", "Articles For Dental Hygiene", "Bath Soap", "Hair Care Products", "Toilet Paper", "Baby Napkin", "Hygiene Pad For Women"],
            "Koruma": ["Transport", "Transport (Total)", "Communication"]
        }

        # 4. Miktar Çarpanları
        qty_configs = {
            "ESSN": {"Rice": 18.9, "Bulgur": 9.45, "Bread": 47.25, "Yoghurt": 9.45, "White Cheese": 9.45, "Egg": 189, "Sun-Flower Oil": 4.725, "Tomato": 5.67, "Cucumber": 5.67, "Dry Bean": 9.45, "Granulated Sugar": 9.45, "Salt": 0.945, "Tea": 0.945, "Notebook": 2, "Pencil": 2, "Other stationery": 2, "Medicines": 3, "Fees Paid To Specialist Doctor": 3, "Actual Rent": 1, "Water": 15, "Electricity": 208.333, "Tube Gas (12 L)": 1, "Detergents (For Laundry)": 1.5, "Dishwasher Detergents": 0.75, "Disinfectants And Insecticidies": 0.5, "Shaving Articles": 2, "Articles For Dental Hygiene": 1, "Bath Soap": 1.5, "Hair Care Products": 0.65, "Toilet Paper": 12, "Baby Napkin": 150, "Hygiene Pad For Women": 30, "Communication": 1, "Transport": 32},
            "CESSN": {"Rice": 12.9, "Bulgur": 6.45, "Bread": 32.25, "Yoghurt": 6.45, "White Cheese": 6.45, "Egg": 129, "Sun-Flower Oil": 3.225, "Tomato": 3.87, "Cucumber": 3.87, "Dry Bean": 6.45, "Granulated Sugar": 6.45, "Salt": 0.645, "Tea": 0.645, "Notebook": 2, "Pencil": 2, "Other stationery": 2, "Medicines": 3, "Fees Paid To Specialist Doctor": 3, "Actual Rent": 1, "Water": 15, "Electricity": 208.333, "Tube Gas (12 L)": 1, "Detergents (For Laundry)": 1.5, "Dishwasher Detergents": 0.75, "Disinfectants And Insecticidies": 0.5, "Shaving Articles": 2, "Articles For Dental Hygiene": 1, "Bath Soap": 1.5, "Hair Care Products": 0.65, "Toilet Paper": 12, "Baby Napkin": 150, "Hygiene Pad For Women": 30, "Communication": 1, "Transport": 32},
            "Ineligible": {"Rice": 15, "Bulgur": 7.5, "Bread": 37.5, "Yoghurt": 7.5, "White Cheese": 7.5, "Egg": 150, "Sun-Flower Oil": 3.75, "Tomato": 4.5, "Cucumber": 4.5, "Dry Bean": 7.5, "Granulated Sugar": 7.5, "Salt": 0.75, "Tea": 0.75, "Notebook": 2, "Pencil": 2, "Other stationery": 2, "Medicines": 3, "Fees Paid To Specialist Doctor": 3, "Actual Rent": 1, "Water": 15, "Electricity": 208.333, "Tube Gas (12 L)": 1, "Detergents (For Laundry)": 1.5, "Dishwasher Detergents": 0.75, "Disinfectants And Insecticidies": 0.5, "Shaving Articles": 2, "Articles For Dental Hygiene": 1, "Bath Soap": 1.5, "Hair Care Products": 0.65, "Toilet Paper": 12, "Baby Napkin": 150, "Hygiene Pad For Women": 30, "Communication": 1, "Transport": 32},
            "Turkish": {"Rice": 15, "Bulgur": 7.5, "Bread": 37.5, "Yoghurt": 7.5, "White Cheese": 7.5, "Egg": 150, "Sun-Flower Oil": 3.75, "Tomato": 4.5, "Cucumber": 4.5, "Dry Bean": 7.5, "Granulated Sugar": 7.5, "Salt": 0.75, "Tea": 0.75, "Notebook": 2, "Pencil": 2, "Other stationery": 2, "Medicines": 3, "Fees Paid To Specialist Doctor": 3, "Actual Rent": 1, "Water": 18, "Electricity": 208.333, "Tube Gas (12 L)": 1, "Detergents (For Laundry)": 1, "Dishwasher Detergents": 0.65, "Disinfectants And Insecticidies": 0.6, "Shaving Articles": 2, "Articles For Dental Hygiene": 2, "Bath Soap": 1.2, "Hair Care Products": 0.65, "Toilet Paper": 10, "Baby Napkin": 90, "Hygiene Pad For Women": 30, "Communication": 1, "Transport": 32}
        }
        
        selected_quantities = qty_configs[selected_option]

        # 5. Hesaplama Motoru
        results = pd.DataFrame(index=enflasyon2.index)
        results['Date'] = date_range

        for product_name, tuik_code in product_name_to_code.items():
            if tuik_code in enflasyon2.columns:
                try:
                    base_index_val = enflasyon2.loc[enflasyon2['Date'] == base_date, tuik_code].values[0]
                    match = base_prices_df.loc[base_prices_df[name_col] == product_name, price_col]
                    if not match.empty:
                        base_price_val = match.values[0]
                        monthly_prices = base_price_val * (enflasyon2[tuik_code].astype(float) / base_index_val)
                        results[product_name] = monthly_prices * selected_quantities.get(product_name, 0)
                    else:
                        results[product_name] = 0.0
                except:
                    results[product_name] = 0.0
            else:
                results[product_name] = 0.0

        # 6. Kategorik Toplamlar
        for cat_name, items in category_map.items():
            existing_items = [item for item in items if item in results.columns]
            results[cat_name] = results[existing_items].sum(axis=1) if existing_items else 0.0

        # 7. Genel MEB Toplamı
        results['Total_MEB'] = results[list(category_map.keys())].sum(axis=1)
        
        # Enflasyon verilerini ekle
        results = pd.concat([results.set_index('Date'), enf_rate], axis=1).reset_index()
        results = results.loc[:, ~results.columns.str.contains('^Unnamed')]
        
        return results

    except Exception as e:
        st.error(f"Hesaplama hatası: {str(e)}")
        return None


# ============================================================================
# MAIN TITLE AND NAVIGATION
# ============================================================================
st.title("🧮 MEB Calculation Tool")
st.markdown("*Minimum Expenditure Basket Calculator based on TUIK Inflation Indices*")
st.markdown("---")

# Sidebar Navigation
with st.sidebar:
    st.header("📋 Navigation")
    page = st.radio("Select Page:", ["📖 Introduction", "📤 Data Upload & Calculate", "📊 Graphs & Analysis"])
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("This tool calculates the Minimum Expenditure Basket (MEB) based on Turkish Statistical Institute (TUIK) inflation indices.")
    st.markdown("[📊 TUIK Official Website](https://www.tuik.gov.tr)")
    
    st.markdown("---")
    st.markdown("### Quick Info")
    st.info("💡 Upload TUIK Excel file with inflation indices to begin calculations")

# ============================================================================
# PAGE 1: INTRODUCTION
# ============================================================================
if page == "📖 Introduction":
    st.header("Welcome to MEB Calculation Tool")
    
    # HTML content from R code Intro tab
    st.markdown("""
        <div style="background-color: #f7f7f7; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                <div style="text-align: center; margin: 10px;">
                    <h3 style="color: #1f77b4;">Minimum Expenditure Basket (MEB)</h3>
                    <p style="text-align: justify; max-width: 800px; margin: auto;">
                        The term 'minimum expenditure basket' refers to the essential set of goods and services considered necessary to maintain a basic standard of living. 
                        The concept is often used in the context of poverty measurement and social policy. The basket includes items such as food, shelter, clothing, health care, and education.
                    </p>
                </div>
            </div>
            
            <hr style="margin: 20px 0;">
            
            <div style="display: flex; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 300px; padding: 10px;">
                    <h3 style="background-color: #C00000; color: white; padding: 10px; text-align: center;">MEB Components</h3>
                    <ul>
                        <li>Shelter</li>
                        <li>Wash (water, sanitation, and hygiene)</li>
                        <li>Clothing</li>
                        <li>Education</li>
                        <li>Health</li>
                        <li>Livelihoods</li>
                        <li>Tax and community contribution</li>
                        <li>Protection and security</li>
                        <li>Healthy diet food basket</li>
                    </ul>
                </div>
                <div style="flex: 2; min-width: 300px; padding: 10px;">
                    <p style="text-align: justify;">
                        Compliance with Sphere standards has been taken into account in the preparation of the MEB. In the Türkiye context, six of the groups that make up the MEB are included: 
                        <b>food, shelter, education, health, protection, and communication</b>. This basket consists mainly of food and shelter. 
                        The food basket is prepared for a balanced diet and a minimum daily energy of 2100 kcal and is based on the average demographic structure of households.
                    </p>
                </div>
            </div>
            
            <div style="overflow-x: auto; margin-top: 20px;">
                <table style="width: 100%; border-collapse: collapse; text-align: left;">
                    <thead>
                        <tr style="background-color: #C00000; color: white;">
                            <th style="padding: 10px;">Food Basket</th>
                            <th style="padding: 10px;">Education</th>
                            <th style="padding: 10px;">Health</th>
                            <th style="padding: 10px;">Shelter</th>
                            <th style="padding: 10px;">Hygiene</th>
                            <th style="padding: 10px;">Protection</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="background-color: #f4f4f4;">
                            <td style="padding: 10px;">Rice, Bulgur, Bread</td>
                            <td style="padding: 10px;">Notebook</td>
                            <td style="padding: 10px;">Medicines</td>
                            <td style="padding: 10px;">Actual Rent</td>
                            <td style="padding: 10px;">Detergents</td>
                            <td style="padding: 10px;">Transport</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px;">Yoghurt, White Cheese</td>
                            <td style="padding: 10px;">Pencil</td>
                            <td style="padding: 10px;">Specialist Doctor</td>
                            <td style="padding: 10px;">Water</td>
                            <td style="padding: 10px;">Soap</td>
                            <td style="padding: 10px;">Communication</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("📦 MEB Components Included")
        st.markdown("""
        1. **Shelter** - Housing and utilities
        2. **WASH** - Water, Sanitation, Hygiene products
        3. **Clothing** - Apparel and footwear
        4. **Education** - School supplies and materials
        5. **Health** - Medical services and medicines
        6. **Food** - Essential food items
        7. **Transportation** - Public transport costs
        8. **Communication** - Phone and utilities
        """)
    
    with col_b:
        st.subheader("🎯 Calculation Scenarios")
        scenarios_data = {
            "Scenario": ["ESSN", "CESSN", "Ineligible", "Turkish"],
            "Description": [
                "Essential Social Security",
                "Core ESSN (Reduced)",
                "Non-eligible household",
                "Turkey-specific basket"
            ],
            "Food Qty": ["Higher", "Medium", "Higher", "Highest"],
            "Shelter": ["Standard", "Standard", "Standard", "Higher"]
        }
        st.dataframe(pd.DataFrame(scenarios_data), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("📋 How to Use This Tool")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Step 1️⃣
        **Prepare Data**
        - Download TUIK inflation index Excel file
        - Ensure proper format with product codes
        """)
    
    with col2:
        st.markdown("""
        ### Step 2️⃣
        **Upload & Calculate**
        - Go to "Data Upload" section
        - Select your scenario
        - Click Calculate
        """)
    
    with col3:
        st.markdown("""
        ### Step 3️⃣
        **Analyze & Export**
        - View results in Graphs section
        - Filter by components & dates
        - Download as Excel file
        """)

# ============================================================================
# PAGE 2: DATA UPLOAD AND CALCULATION
# ============================================================================
elif page == "📤 Data Upload & Calculate":
    st.header("📤 Data Upload & Calculation")
    
    # File upload and scenario selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📁 Upload TUIK Inflation Indexes Excel File", type=["xlsx"])
    
    with col2:
        selected_option = st.selectbox(
            "Select Scenario:",
            ["ESSN", "CESSN", "Ineligible", "Turkish"],
            help="Choose the MEB calculation scenario"
        )
    
    # Information about scenarios
    with st.expander("ℹ️ About Scenarios"):
        st.markdown("""
        - **ESSN**: Essential Social Security Network - Higher quantities for basic needs
        - **CESSN**: Core ESSN - Reduced quantities, more conservative approach
        - **Ineligible**: Non-eligible household basket - Alternative calculation method
        - **Turkish**: Turkey-specific MEB - Adjusted for Turkish market conditions
        """)
    
    if uploaded_file is not None:
        try:
            # Read Excel file
            tuik_data = pd.read_excel(uploaded_file, skiprows=4)
            st.success("✅ File loaded successfully!")
            
            # Display file preview
            with st.expander("👁️ View Raw Data Preview"):
                st.subheader("First 10 rows of uploaded data:")
                st.dataframe(tuik_data.head(10), use_container_width=True)
            
            # Calculate button
            col_btn1, col_btn2 = st.columns([2, 1])
            
            with col_btn1:
                if st.button("🧮 Calculate MEB", use_container_width=True):
                    with st.spinner("🔄 Calculating MEB values..."):
                        try:
                            meb_results = calculate_meb(tuik_data, selected_option)
                            st.session_state.merge_df = meb_results
                            st.session_state.merge_df1 = meb_results.copy()
                            st.session_state.calculation_complete = True
                            st.session_state.raw_data = tuik_data
                            st.success(f"✅ MEB calculation complete for {selected_option}!")
                        
                        except Exception as e:
                            st.error(f"❌ Calculation error: {str(e)}")
            
            # Display results if calculation is complete
            if st.session_state.calculation_complete:
                st.markdown("---")
                st.subheader("📊 Calculation Results")
                
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Summary", "📋 Components", "📊 Statistics", "⬇️ Export"])
                
                with tab1:
                    st.markdown("### Total MEB Values (Last 24 months)")
                    meb_summary = st.session_state.merge_df[["Total_MEB"]].tail(24)
                    st.dataframe(meb_summary, use_container_width=True)
                    
                    # Display key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current MEB", f"₺{st.session_state.merge_df['Total_MEB'].iloc[-1]:,.2f}")
                    with col2:
                        st.metric("Average MEB", f"₺{st.session_state.merge_df['Total_MEB'].mean():,.2f}")
                    with col3:
                        st.metric("Min MEB", f"₺{st.session_state.merge_df['Total_MEB'].min():,.2f}")
                    with col4:
                        st.metric("Max MEB", f"₺{st.session_state.merge_df['Total_MEB'].max():,.2f}")
                
                with tab2:
                    st.markdown("### MEB Components Breakdown (Last 24 months)")
                    components_data = st.session_state.merge_df.tail(24)
                    st.dataframe(components_data, use_container_width=True)
                
                with tab3:
                    st.markdown("### Statistical Summary")
                    
                    stats_cols = ["Total_MEB"] + [col for col in st.session_state.merge_df.columns if col != "Total_MEB"][:5]
                    stats_available = [col for col in stats_cols if col in st.session_state.merge_df.columns]
                    
                    stats_df = st.session_state.merge_df[stats_available].describe().T
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # Additional statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        growth = ((st.session_state.merge_df["Total_MEB"].iloc[-1] / 
                                  st.session_state.merge_df["Total_MEB"].iloc[0]) - 1) * 100
                        st.metric("Total Growth %", f"{growth:.2f}%")
                    
                    with col2:
                        latest = st.session_state.merge_df["Total_MEB"].iloc[-1]
                        prev_month = st.session_state.merge_df["Total_MEB"].iloc[-2]
                        monthly_change = ((latest / prev_month) - 1) * 100
                        st.metric("Monthly Change %", f"{monthly_change:.2f}%")
                    
                    with col3:
                        std_dev = st.session_state.merge_df["Total_MEB"].std()
                        st.metric("Std Deviation", f"{std_dev:,.2f}")
                
                with tab4:
                    st.markdown("### 📥 Export Results")
                    
                    # Excel export
                    excel_buffer = create_excel_export(st.session_state.merge_df)
                    
                    st.download_button(
                        label="📥 Download Results (Excel)",
                        data=excel_buffer,
                        file_name=f"MEB_Calculation_{selected_option}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
                    # CSV export
                    csv_data = st.session_state.merge_df.to_csv().encode('utf-8')
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv_data,
                        file_name=f"MEB_Calculation_{selected_option}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    else:
        st.info("📤 Please upload an Excel file to begin")

# ============================================================================
# PAGE 3: GRAPHS AND VISUALIZATION
# ============================================================================
elif page == "📊 Graphs & Analysis":
    st.header("📈 Visualization & Analysis")
    
    if st.session_state.merge_df is not None:
        # Filters
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            selected_components = st.multiselect(
                "Select Components to Display:",
                options=[col for col in st.session_state.merge_df.columns if col != "Total_Food"],
                default=[col for col in st.session_state.merge_df.columns if col != "Total_Food"][:3],
                help="Select which components to show in visualizations"
            )
        
        with col2:
            date_range = st.date_input(
                "Select Date Range:",
                value=(st.session_state.merge_df.index.min().date(), st.session_state.merge_df.index.max().date()),
                key="date_range"
            )
        
        with col3:
            if st.button("🔄 Generate Graphs", use_container_width=True):
                st.session_state.generate_graphs = True
        
        if 'generate_graphs' in st.session_state and st.session_state.generate_graphs:
            # Filter data by date range
            if len(date_range) == 2:
                mask = (st.session_state.merge_df.index.date >= date_range[0]) & (st.session_state.merge_df.index.date <= date_range[1])
                filtered_data = st.session_state.merge_df[mask]
                
                if len(filtered_data) == 0:
                    st.warning("⚠️ No data found for selected date range")
                else:
                    # Tab structure for visualizations
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Bar Chart", "📈 Line Chart", "📉 Trend Analysis", "🔍 Detailed View"])
                    
                    with tab1:
                        st.subheader("Monthly MEB Components & Inflation Rate (Dual Axis)")
                        
                        # Filter inflation columns
                        inf_cols = [col for col in filtered_data.columns if col.startswith("Inf_")]
                        
                        fig_dual = go.Figure()
                        
                        # Add Bar traces for MEB components
                        for component in selected_components:
                            if component in filtered_data.columns:
                                fig_dual.add_trace(go.Bar(
                                    x=filtered_data['Date'],
                                    y=filtered_data[component],
                                    name=component,
                                    yaxis="y1"
                                ))
                        
                        # Add Line traces for Inflation Rates (Dual Axis)
                        for inf in inf_cols[:2]: # Show first 2 inflation rates by default
                            fig_dual.add_trace(go.Scatter(
                                x=filtered_data['Date'],
                                y=filtered_data[inf],
                                name=f"CPI ({inf.replace('Inf_', '')})",
                                yaxis="y2",
                                line=dict(width=3)
                            ))
                        
                        fig_dual.update_layout(
                            title="Monthly amount of MEB components and Annual Inflation Rates",
                            xaxis_title="Date",
                            yaxis=dict(title="MEB Component Value (₺)"),
                            yaxis2=dict(
                                title="Inflation Rate",
                                overlaying="y",
                                side="right",
                                tickformat=".1%"
                            ),
                            barmode='stack',
                            hovermode="x unified",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            height=700
                        )
                        st.plotly_chart(fig_dual, use_container_width=True)

                    with tab2:
                        st.subheader("Total MEB Trend (Line Chart)")
                        
                        fig2 = go.Figure()
                        
                        if "Total_Food" in filtered_data.columns:
                            fig2.add_trace(go.Scatter(
                                x=filtered_data.index,
                                y=filtered_data["Total_Food"],
                                mode='lines+markers',
                                name='Total MEB',
                                line=dict(color='#FF6B6B', width=3),
                                hovertemplate='<b>Total MEB</b><br>Date: %{x|%Y-%m}<br>Value: ₺%{y:,.2f}<extra></extra>'
                            ))
                        
                        for component in selected_components[:3]:
                            if component in filtered_data.columns:
                                fig2.add_trace(go.Scatter(
                                    x=filtered_data.index,
                                    y=filtered_data[component],
                                    mode='lines',
                                    name=component,
                                    opacity=0.7,
                                    hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%Y-%m}<br>Value: ₺%{y:,.2f}<extra></extra>'
                                ))
                        
                        fig2.update_layout(
                            title="Total MEB and Component Trends Over Time",
                            xaxis_title="Date",
                            yaxis_title="Cost (₺)",
                            hovermode="x unified",
                            height=600
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    with tab3:
                        st.subheader("MEB Trend Analysis")
                        
                        # Calculate growth rates
                        fig3 = go.Figure()
                        
                        for component in selected_components[:5]:
                            if component in filtered_data.columns:
                                component_data = filtered_data[component]
                                growth_rates = component_data.pct_change().fillna(0) * 100
                                
                                fig3.add_trace(go.Scatter(
                                    x=filtered_data.index,
                                    y=growth_rates,
                                    mode='lines+markers',
                                    name=f'{component} Growth %',
                                    hovertemplate='<b>%{fullData.name}</b><br>Date: %{x|%Y-%m}<br>Growth: %{y:,.2f}%<extra></extra>'
                                ))
                        
                        fig3.add_hline(y=0, line_dash="dash", line_color="gray")
                        
                        fig3.update_layout(
                            title="Month-over-Month Growth Rates (%)",
                            xaxis_title="Date",
                            yaxis_title="Growth Rate (%)",
                            hovermode="x unified",
                            height=600
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                    
                    with tab4:
                        st.subheader("Detailed Component Breakdown")
                        
                        # Show detailed table
                        display_cols = ["Total_Food"] + selected_components
                        display_cols = [col for col in display_cols if col in filtered_data.columns]
                        
                        detailed_table = filtered_data[display_cols].copy()
                        st.dataframe(detailed_table, use_container_width=True)
                        
                        # Summary statistics for selected period
                        st.subheader("Summary Statistics for Selected Period")
                        
                        summary_stats = detailed_table.describe()
                        st.dataframe(summary_stats, use_container_width=True)
            else:
                st.warning("⚠️ Please select a valid date range")
    
    else:
        st.warning("⚠️ No calculation data available. Please upload and calculate data first in the 'Data Upload' section.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.caption("🧮 MEB Calculation Tool • Built with Streamlit | Data: TUIK Inflation Indices")

with col2:
    st.caption("[GitHub](https://github.com)")

with col3:
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")
