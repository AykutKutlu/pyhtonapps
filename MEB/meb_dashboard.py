"""
MEB (Minimum Expenditure Basket) Calculation Tool
Streamlit web application for calculating Minimum Expenditure Basket based on TUIK indices
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px  # <-- Eksik olan satır bu
from datetime import datetime, timedelta
import io
from plotly.subplots import make_subplots
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
    .main { padding-top: 1rem; }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    h1 { color: #1f77b4; }
    h2 { color: #1f77b4; margin-top: 1.5rem; }
    .stTabs [role="tab"] { font-size: 16px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'merge_df' not in st.session_state:
    st.session_state.merge_df = None
if 'calculation_complete' not in st.session_state:
    st.session_state.calculation_complete = False
if 'generate_graphs' not in st.session_state:
    st.session_state.generate_graphs = False

# ============================================================================
# CALCULATION ENGINE (Your Core Logic)
# ============================================================================

def calculate_meb(tuik_data, selected_option):
    try:
        # 1. TUIK Verisini Hazırlama
        df = tuik_data.copy()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df.columns = ["Year", "Month", "Months"] + list(df.columns[3:])
        df = df[:-5] # Dipnotları temizle
        enflasyon2 = df.iloc[2:].reset_index(drop=True)
        
        start_date = datetime(2005, 1, 1)
        date_range = pd.date_range(start=start_date, periods=len(enflasyon2), freq='MS')
        enflasyon2['Date'] = date_range

        # Enflasyon Oranları (YoY)
        enf_codes = ["0", "011", "04", "06", "07", "08", "10", "13"]
        available_codes = [c for c in enf_codes if c in enflasyon2.columns]
        enf_index = enflasyon2[available_codes].apply(pd.to_numeric, errors='coerce')
        name_map = {
            "0": "Total",
            "011": "Food",
            "04": "Shelter",
            "06": "Health",
            "07": "Transportation",
            "08": "Communication",
            "10": "Education",
            "13": "Other"
        }

        # 2. Mevcut kodunuza entegre edin
        enf_rates = enf_index.pct_change(periods=12)

        # Sütun isimlerini eşleştirme yaparak değiştiriyoruz
        enf_rates.columns = [f"Inf_{name_map.get(c, c)}" if c != "0" else "Inf_Total" for c in enf_rates.columns]
        enf_rates['Date'] = date_range

        # 2. Baz Fiyat Dosyalarını Yükleme (Aralık 2025)
        base_file = "MEB_TR.xlsx" if selected_option == "Turkish" else "MEB_SSN.xlsx"
        base_prices_df = pd.read_excel(base_file)
        base_prices_df.columns = [str(c).strip() for c in base_prices_df.columns]
        base_date = datetime(2025, 12, 1)

        # 3. Ürün Eşleştirme Sözlüğü (Senin Excel başlıklarına göre uyarlandı)
        tuik_to_names = {
            "01111": ["Rice", "Bulgur"], "01113": ["Bread"], "01146": ["Yogurt"],
            "01145": ["White cheese"], "01148": ["Egg"], "01151": ["Sunflower oil"],
            "01172": ["Tomatoes", "Cucumber"], "01176": ["Dry beans"], "01181": ["Sugar"],
            "01193": ["Salt"], "01230": ["Tea"], 
            "09740": ["Notebook", "Pencil", "Other stationary"],
            "06111": ["Medicine"], "06231": ["Specialist"], "04110": ["Rent"],
            "04411": ["Water"], "04510": ["Electricity"], "04522": ["Gas canister (12 L)"],
            "05611": ["Laundry detergent", "Dishwashing liquid", "Disinfectant"],
            "13120": ["Shaving articles", "Dental hygiene", "Soap", "Haircare", "Toilet paper", "Diaper", "Hygiene Pad for Women"],
            "08320": ["Mobile service package"], "07321": ["Public transportation"]
        }

        # 4. Miktar Çarpanları
        qtys = {
            "ESSN": {"Rice": 18.9, "Bulgur": 9.45, "Bread": 47.25, "Yogurt": 9.45, "White cheese": 9.45, "Egg": 189, "Sunflower oil": 4.725, "Tomatoes": 5.67, "Cucumber": 5.67, "Dry beans": 9.45, "Sugar": 9.45, "Salt": 0.945, "Tea": 0.945, "Notebook": 2, "Pencil": 2, "Other stationary": 2, "Medicine": 3, "Specialist": 3, "Rent": 1, "Water": 15, "Electricity": 208.333, "Gas canister (12 L)": 1, "Laundry detergent": 1.5, "Dishwashing liquid": 0.75, "Disinfectant": 0.5, "Shaving articles": 2, "Dental hygiene": 1, "Soap": 1.5, "Haircare": 0.65, "Toilet paper": 12, "Diaper": 150, "Hygiene Pad for Women": 30, "Mobile service package": 1, "Public transportation": 32},
            "CESSN": {"Rice": 12.9, "Bulgur": 6.45, "Bread": 32.25, "Yogurt": 6.45, "White cheese": 6.45, "Egg": 129, "Sunflower oil": 3.225, "Tomatoes": 3.87, "Cucumber": 3.87, "Dry beans": 6.45, "Sugar": 6.45, "Salt": 0.645, "Tea": 0.645, "Notebook": 2, "Pencil": 2, "Other stationary": 2, "Medicine": 3, "Specialist": 3, "Rent": 1, "Water": 15, "Electricity": 208.333, "Gas canister (12 L)": 1, "Laundry detergent": 1.5, "Dishwashing liquid": 0.75, "Disinfectant": 0.5, "Shaving articles": 2, "Dental hygiene": 1, "Soap": 1.5, "Haircare": 0.65, "Toilet paper": 12, "Diaper": 150, "Hygiene Pad for Women": 30, "Mobile service package": 1, "Public transportation": 32},
            "Ineligible": {"Rice": 15.6, "Bulgur": 7.8, "Bread": 39, "Yogurt": 7.8, "White cheese": 7.8, "Egg": 156, "Sunflower oil": 3.9, "Tomatoes": 4.68, "Cucumber": 4.68, "Dry beans": 7.8, "Sugar": 7.8, "Salt": 0.78, "Tea": 0.78, "Notebook": 2, "Pencil": 2, "Other stationary": 2, "Medicine": 3, "Specialist": 3, "Rent": 1, "Water": 15, "Electricity": 208.333, "Gas canister (12 L)": 1, "Laundry detergent": 1.5, "Dishwashing liquid": 0.75, "Disinfectant": 0.5, "Shaving articles": 2, "Dental hygiene": 1, "Soap": 1.5, "Haircare": 0.65, "Toilet paper": 12, "Diaper": 150, "Hygiene Pad for Women": 30, "Mobile service package": 1, "Public transportation": 32},
            "Turkish": {"Rice": 15, "Bulgur": 7.5, "Bread": 37.5, "Yogurt": 7.5, "White cheese": 7.5, "Egg": 150, "Sunflower oil": 3.75, "Tomatoes": 4.5, "Cucumber": 4.5, "Dry beans": 7.5, "Sugar": 7.5, "Salt": 0.75, "Tea": 0.75, "Notebook": 2, "Pencil": 2, "Other stationary": 2, "Medicine": 3, "Specialist": 3, "Rent": 1, "Water": 18, "Electricity": 208.333, "Gas canister (12 L)": 1, "Laundry detergent": 1, "Dishwashing liquid": 0.65, "Disinfectant": 0.6, "Shaving articles": 2, "Dental hygiene": 2, "Soap": 1.2, "Haircare": 0.65, "Toilet paper": 10, "Diaper": 90, "Hygiene Pad for Women": 30, "Mobile service package": 1, "Public transportation": 32}
        }
        selected_qtys = qtys[selected_option]

        # 5. Hesaplama
        results = pd.DataFrame()
        results['Date'] = date_range

        for code, names in tuik_to_names.items():
            if code in enflasyon2.columns:
                try:
                    base_idx_val = enflasyon2.loc[enflasyon2['Date'] == base_date, code].values[0]
                    for name in names:
                        if name in base_prices_df.columns:
                            b_price = float(base_prices_df[name].iloc[0])
                            results[name] = (b_price * (enflasyon2[code].astype(float) / base_idx_val)) * selected_qtys.get(name, 0)
                        else: results[name] = 0.0
                except:
                    for name in names: results[name] = 0.0

        # Kategorik Toplamlar
        results['Total_Food'] = results[["Rice", "Bulgur", "Bread", "Yogurt", "White cheese", "Egg", "Sunflower oil", "Tomatoes", "Cucumber", "Dry beans", "Sugar", "Salt", "Tea"]].sum(axis=1)
        results['Total_NFI'] = results[["Laundry detergent", "Dishwashing liquid", "Disinfectant", "Shaving articles", "Dental hygiene", "Soap", "Haircare", "Toilet paper", "Diaper", "Hygiene Pad for Women"]].sum(axis=1)
        results['Total_Shelter'] = results[["Rent", "Water", "Electricity", "Gas canister (12 L)"]].sum(axis=1)
        results['Total_Education'] = results[["Notebook", "Pencil", "Other stationary"]].sum(axis=1)
        results['Total_Health'] = results[["Medicine", "Specialist"]].sum(axis=1)
        results['Total_Protection'] = results[["Mobile service package", "Public transportation"]].sum(axis=1)
        results['Total_MEB'] = results[['Total_Food', 'Total_NFI', 'Total_Shelter', 'Total_Education', 'Total_Health', 'Total_Protection']].sum(axis=1)
        
        return pd.merge(results, enf_rates, on='Date', how='left')

    except Exception as e:
        st.error(f"❌ Calculation error: {str(e)}")
        return None

def create_excel_export(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Full_MEB_Results", index=False)
    return output.getvalue()

# ============================================================================
# PAGE 1: INTRODUCTION
# ============================================================================
with st.sidebar:
    st.header("📋 Navigation")
    page = st.radio("Select Page:", ["📖 Introduction", "📤 Data Upload & Calculate", "📊 Graphs & Analysis"])
    st.markdown("---")
    st.info("💡 Upload TUIK Index Excel file to start.")

if page == "📖 Introduction":
    st.markdown("""
        <div style="background-color: #f7f7f7; padding: 25px; border: 1px solid #ddd; border-radius: 10px;">
            <h1 style="color:#C00000; text-align:center;">Minimum Expenditure Basket (MEB)</h1>
            <p style="font-size: 1.1rem; text-align: justify;">
                The term 'minimum expenditure basket' refers to the essential set of goods and services considered necessary to maintain a basic standard of living. 
                Compliance with <b>Sphere standards</b> has been taken into account. In the Türkiye context, six main groups are included: 
                food, shelter, education, health, protection, and communication.
            </p>
            <div style="display: flex; flex-wrap: wrap; gap: 20px; margin-top: 20px;">
                <div style="flex: 1; background: white; padding: 20px; border-radius: 8px; border: 1px solid #eee; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
                    <h3 style="color:#C00000; border-bottom: 2px solid #C00000;">MEB Components</h3>
                    <ul style="line-height: 1.8;">
                        <li><b>Shelter & Utilities:</b> Rent, Water, Electricity, Natural Gas</li>
                        <li><b>WASH:</b> Non-Food Items, Detergents for Laundry, Dishwasher Detergents, Disinfenctants an Insecticidies, Shaving Articles, Dental Hygiene Articles, Bath Soap, Shampoo, Toilet Paper, Baby Napkin, Hygiene Pad for Women</li>
                        <li><b>Education:</b> Notebooks, Pencils, Other Stationery Items</li>
                        <li><b>Health:</b> Medicines, Fees Paid to Specialist Doctor</li>
                        <li><b>Food Basket:</b> Rice, Bulgur, Bread, Yoghurt, White Cheese, Egg, Sun-Flower Oil, Tomato, Cucumber, Dry Bean, Granulated Suger, Salt, Tea</li>
                        <li><b>Protection & Communication:</b> Mobile Service Package, Public Transportation</li>
                    </ul>
                </div>
                <div style="flex: 1; background: white; padding: 20px; border-radius: 8px; border: 1px solid #eee; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);">
                    <h3 style="color:#C00000; border-bottom: 2px solid #C00000; margin-bottom: 15px;">Methodology</h3>
                    <p style="margin-bottom: 10px;">
                        The MEB estimation is conducted by the <b>Türk Kızılay KIZILAYKART M&E team</b> using a comprehensive dual-source framework:
                    </p>
                    <ul style="padding-left: 20px; font-size: 0.95em; color: #333; line-height: 1.6;">
                        <li><b>Primary Sources:</b> Direct data collection through PAB, PDM, and Inter-Sectoral surveys to capture real-time household expenditures.</li>
                        <li><b>Secondary Sources:</b> Integration of external data, including <b>TurkStat (TUIK)</b> indices and peer-reviewed scientific publications.</li>
                        <li><b>Price Indexing:</b> Calculations utilize <b>Web Scraping</b> for market averages, with <b>December 2025</b> as the reference base period.</li>
                        <li><b>Operational Scope:</b> A hybrid approach of field-based and remote data collection ensures a balance between cost-efficiency and high data reliability.</li>
                    </ul>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE 2: DATA UPLOAD AND CALCULATION
# ============================================================================
elif page == "📤 Data Upload & Calculate":
    st.header("📤 Data Upload & Calculation")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("📁 Upload TUIK Inflation Indexes Excel File", type=["xlsx"])
    with col2:
        selected_option = st.selectbox("Select Scenario:", ["ESSN", "CESSN", "Ineligible", "Turkish"])

    if uploaded_file is not None:
        if st.button("🧮 Calculate MEB", type="primary", use_container_width=True):
            with st.spinner("🔄 Calculating..."):
                raw_data = pd.read_excel(uploaded_file, skiprows=4)
                res = calculate_meb(raw_data, selected_option)
                if res is not None:
                    st.session_state.merge_df = res
                    st.session_state.calculation_complete = True
                    st.success("✅ Calculation Complete!")

    if st.session_state.calculation_complete:
        df = st.session_state.merge_df
        tab1, tab2, tab3 = st.tabs(["📈 Summary", "📋 Detailed Data", "⬇️ Export"])
        
        with tab1:
            st.subheader("Results Summary (Last 12 Months)")
            cols = ['Date', 'Total_Food', 'Total_Shelter', 'Total_NFI', 'Total_MEB']
            st.dataframe(df[cols].tail(12).set_index('Date'), use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Latest Total MEB", f"₺{df['Total_MEB'].iloc[-1]:,.2f}")
            c2.metric("YoY Increase (Total)", f"{((df['Total_MEB'].iloc[-1]/df['Total_MEB'].iloc[-13])-1)*100:.1f}%")
            c3.metric("Food Basket Share", f"{(df['Total_Food'].iloc[-1]/df['Total_MEB'].iloc[-1])*100:.1f}%")

        with tab2:
            st.dataframe(df, use_container_width=True)

        with tab3:
            st.download_button("📥 Download Full Results (Excel)", 
                             data=create_excel_export(df), 
                             file_name=f"MEB_Results_{selected_option}.xlsx",
                             use_container_width=True)

# ============================================================================
# PAGE 3: GRAPHS AND VISUALIZATION
# ============================================================================
elif page == "📊 Graphs & Analysis":
    st.header("📈 Visualization & Analysis")
    
    if st.session_state.merge_df is not None:
        df = st.session_state.merge_df
        
        # Sekmeli Grafik Yapısı
        tab_shares, tab_bar, tab_trend = st.tabs([
            "🍰 Component Shares", 
            "📊 Inflation & Growth", 
            "📈 Total Trend"
        ])
        
        # 1. SEKME: MEB Bileşenlerinin Payları (Pie Chart)
        with tab_shares:
            st.subheader("Composition of Minimum Expenditure Basket")
            col_pie1, col_pie2 = st.columns([1, 2])
            
            with col_pie1:
                # Son aya ait verileri al
                latest_data = df.iloc[-1]
                components = ['Total_Food', 'Total_Shelter', 'Total_NFI', 'Total_Education', 'Total_Health']
                values = [latest_data[c] for c in components]
                
                # Tarih seçici (Opsiyonel: Kullanıcı hangi ayın payını görmek isterse)
                selected_date = st.selectbox("Select Month for Share Analysis:", 
                                           options=df['Date'].dt.strftime('%Y-%m').unique(),
                                           index=len(df)-1)
                
                current_month_data = df[df['Date'].dt.strftime('%Y-%m') == selected_date].iloc[0]
                current_values = [current_month_data[c] for c in components]

            with col_pie2:
                fig_pie = px.pie(
                    values=current_values, 
                    names=components,
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title=f"Expenditure Shares in {selected_date}"
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

        # 2. SEKME: Bileşenler ve Artış Oranları (Dual Axis Bar Chart)
        with tab_bar:
            st.subheader("Monthly MEB Components & Annual Inflation Rates")
            
            selected_components = st.multiselect(
                "Select Components to Show:",
                options=['Total_Food', 'Total_Shelter', 'Total_NFI', 'Total_Education', 'Total_Health'],
                default=['Total_Food', 'Total_Shelter', 'Total_NFI']
            )

            fig_dual = go.Figure()
            
            # Y1 Ekseni: MEB Bileşenleri (Yığılmış Bar)
            for comp in selected_components:
                fig_dual.add_trace(go.Bar(
                    x=df['Date'], 
                    y=df[comp], 
                    name=comp, 
                    yaxis="y1",
                    hovertemplate='₺%{y:,.2f}'
                ))
            
            # Y2 Ekseni: Artış Oranları (Enflasyon Bilgileri)
            # TUIK'ten gelen Inf_0 (Genel), Inf_011 (Gıda) vb. bilgileri çizgi olarak ekle
            inf_options = [c for c in df.columns if c.startswith('Inf_')]
            selected_infs = st.multiselect("Select Inflation Rates (YoY) to Overlay:", 
                                         options=inf_options, 
                                         default=['Inf_Total'] if 'Inf_Total' in df.columns else [])

            for inf in selected_infs:
                fig_dual.add_trace(go.Scatter(
                    x=df['Date'], 
                    y=df[inf], 
                    name=f"Growth: {inf}", 
                    yaxis="y2",
                    line=dict(width=3),
                    mode='lines'
                ))

            fig_dual.update_layout(
                xaxis_title="Date",
                yaxis=dict(title="Monthly Cost (₺)", side="left"),
                yaxis2=dict(
                    title="Annual Inflation / Growth Rate",
                    overlaying="y",
                    side="right",
                    tickformat=".1%",
                    showgrid=False
                ),
                barmode='stack',
                legend=dict(orientation="h", y=1.1),
                height=600,
                hovermode="x unified"
            )
            st.plotly_chart(fig_dual, use_container_width=True)

        # 3. SEKME: Genel Trend
        with tab_trend:
            st.subheader("Total MEB & General Inflation Analysis")
            
            # Debug: Sütun isimlerini Streamlit'te görerek kontrol edebilirsin (Gerekirse açarsın)
            # st.write(df.columns.tolist()) 

            fig_summary = make_subplots(specs=[[{"secondary_y": True}]])

            # MEB Toplam Tutarı (Sol Eksen)
            fig_summary.add_trace(
                go.Scatter(x=df['Date'], y=df['Total_MEB'], name="Total MEB Cost (₺)", fill='tozeroy'),
                secondary_y=False,
            )

            # Inf_Total'i bulmaya çalışıyoruz
            target_inf = "Inf_Total" 
            
            if target_inf in df.columns:
                fig_summary.add_trace(
                    go.Scatter(x=df['Date'], y=df[target_inf], name="Annual Inflation (%)", 
                            line=dict(color='red', width=3)),
                    secondary_y=True,
                )
            else:
                # Eğer hala bulamıyorsa, Inf_ ile başlayan ilk sütunu (genelde genel enflasyondur) alalım
                inf_cols = [c for c in df.columns if c.startswith('Inf_')]
                if inf_cols:
                    fig_summary.add_trace(
                        go.Scatter(x=df['Date'], y=df[inf_cols[0]], name=f"Inflation ({inf_cols[0]})", 
                                line=dict(color='red', width=3)),
                        secondary_y=True,
                    )
                else:
                    st.warning("Enflasyon sütunu (Inf_...) bulunamadı!")

            fig_summary.update_layout(height=500, hovermode="x unified")
            fig_summary.update_yaxes(title_text="Cost (₺)", secondary_y=False)
            fig_summary.update_yaxes(title_text="Inflation Rate", secondary_y=True, tickformat=".1%")

            st.plotly_chart(fig_summary, use_container_width=True)
    else:
        st.warning("⚠️ No calculation data available. Please upload and calculate data first.")