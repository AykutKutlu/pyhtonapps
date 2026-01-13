import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import sqlite3
import os

# --- DOSYA YOLU AYARI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "giderler.db")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Finansal Dashboard Pro", layout="wide", page_icon="📊")

# --- VERİTABANI FONKSİYONLARI ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS giderler
                 (ID TEXT PRIMARY KEY, odeme_tarihi TEXT, gider_kalemi TEXT, 
                  miktar REAL, tip TEXT, zam_ayi TEXT, zam_orani REAL)''')
    try:
        c.execute("ALTER TABLE giderler ADD COLUMN zam_ayi TEXT DEFAULT 'Yok'")
        c.execute("ALTER TABLE giderler ADD COLUMN zam_orani REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=['ID', 'odeme_tarihi', 'gider_kalemi', 'miktar', 'tip', 'zam_ayi', 'zam_orani'])
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM giderler", conn)
    conn.close()
    if df.empty:
        return pd.DataFrame(columns=['ID', 'odeme_tarihi', 'gider_kalemi', 'miktar', 'tip', 'zam_ayi', 'zam_orani'])
    
    df.columns = ['ID', 'odeme_tarihi', 'gider_kalemi', 'miktar', 'tip', 'zam_ayi', 'zam_orani']
    df['odeme_tarihi'] = pd.to_datetime(df['odeme_tarihi'])
    return df

def update_item_zam(kalem_adi, yeni_zam_ayi, yeni_zam_orani):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE giderler SET zam_ayi = ?, zam_orani = ? WHERE gider_kalemi = ?", 
                 (yeni_zam_ayi, yeni_zam_orani, kalem_adi))
    conn.commit()
    conn.close()

def delete_from_db(id_val):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM giderler WHERE ID = ?", (id_val,))
    conn.commit()
    conn.close()

def clear_all_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM giderler")
    conn.commit()
    conn.close()

init_db()
if 'gider_listesi' not in st.session_state:
    st.session_state.gider_listesi = load_data()

# --- YAN PANEL ---
st.sidebar.header("⚙️ Finansal Parametreler")
mevcut_birikim = st.sidebar.number_input("Mevcut Birikim (TL)", min_value=0.0, value=50000.0)
baslangic_gelir = st.sidebar.number_input("Güncel Maaş (TL)", min_value=0.0, value=35000.0)

st.sidebar.divider()
st.sidebar.header("📈 Maaş Zam Senaryosu")
subat_zammi = st.sidebar.slider("Şubat Zam Oranı (%)", 0, 100, 30)
agustos_zammi = st.sidebar.slider("Ağustos Zam Oranı (%)", 0, 100, 20)

if st.sidebar.button("🗑️ Tüm Verileri Temizle"):
    clear_all_db()
    st.session_state.gider_listesi = load_data()
    st.rerun()

# --- ANA EKRAN ---
st.title("🛡️ Akıllı Finansal Yönetim")
tab1, tab2, tab3, tab4 = st.tabs(["➕ Ödeme Girişi", "📊 Detaylı Raporlama", "⚙️ Zam Yönetimi", "❌ Kayıt Silme"])

# --- TAB 1: ÖDEME GİRİŞİ ---
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📍 Tek Seferlik Ödeme")
        with st.form("tek"):
            t_tarih = st.date_input("Ödeme Tarihi", date.today())
            t_kalem = st.text_input("Gider Kalemi")
            t_miktar = st.number_input("Tutar", min_value=0.0)
            if st.form_submit_button("Kaydet"):
                conn = sqlite3.connect(DB_PATH)
                yid = f"{int(datetime.now().timestamp())}_0_{t_kalem[:3]}"
                conn.execute("INSERT INTO giderler VALUES (?, ?, ?, ?, ?, ?, ?)", 
                             (yid, t_tarih.strftime('%Y-%m-%d'), t_kalem, t_miktar, "Değişken", "Yok", 0))
                conn.commit()
                conn.close()
                st.session_state.gider_listesi = load_data()
                st.rerun()

    with c2:
        st.subheader("🔄 Sabit Gider / Taksit")
        with st.form("sabit"):
            s_tarih = st.date_input("Başlangıç Tarihi", date.today())
            s_kalem = st.text_input("Gider Tanımı")
            s_miktar = st.number_input("Aylık Tutar", min_value=0.0)
            s_sure = st.number_input("Süre (Ay)", min_value=1, value=12)
            if st.form_submit_button("Planı Oluştur"):
                conn = sqlite3.connect(DB_PATH)
                for i in range(int(s_sure)):
                    yeni_tarih = s_tarih + relativedelta(months=i)
                    yid = f"{int(datetime.now().timestamp())}_{i}_{s_kalem[:3]}"
                    conn.execute("INSERT INTO giderler VALUES (?, ?, ?, ?, ?, ?, ?)", 
                                 (yid, yeni_tarih.strftime('%Y-%m-%d'), s_kalem, s_miktar, "Sabit", "Yok", 0))
                conn.commit()
                conn.close()
                st.session_state.gider_listesi = load_data()
                st.rerun()

# --- TAB 2: DETAYLI RAPORLAMA (İSTEDİĞİN ESKİ HALİ) ---
with tab2:
    if not st.session_state.gider_listesi.empty:
        df = st.session_state.gider_listesi.copy()
        
        # Projeksiyon Hazırlığı
        proj_data = []
        baslangic_ayi = date.today() + relativedelta(months=1)
        kasa = mevcut_birikim
        guncel_gelir = baslangic_gelir
        tr_aylar_list = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]

        for i in range(12):
            ay_dt = baslangic_ayi + relativedelta(months=i)
            ay_tr_isim = tr_aylar_list[ay_dt.month - 1]
            ay_str = ay_dt.strftime("%Y-%m")

            # Gelir Zamları
            if ay_dt.month == 2: guncel_gelir *= (1 + subat_zammi/100)
            if ay_dt.month == 8: guncel_gelir *= (1 + agustos_zammi/100)

            # Giderler ve Özel Zamlar
            aylik_df = df[df['odeme_tarihi'].dt.strftime('%Y-%m') == ay_str].copy()
            toplam_harcama = 0
            for _, row in aylik_df.iterrows():
                m = row['miktar']
                if row['zam_ayi'] == ay_tr_isim:
                    m *= (1 + row['zam_orani']/100)
                toplam_harcama += m
            
            tasarruf = guncel_gelir - toplam_harcama
            kasa += tasarruf
            proj_data.append({"Ay": ay_str, "Gelir": guncel_gelir, "Gider": toplam_harcama, "Bakiye": kasa, "Aylık Tasarruf": tasarruf})
        
        proj_df = pd.DataFrame(proj_data)

        # Üst Metrikler
        m1, m2, m3 = st.columns(3)
        m1.metric("Ortalama Aylık Gider", f"{proj_df['Gider'].mean():,.0f} TL")
        m2.metric("Öngörülen 12 Aylık Birikim", f"{kasa:,.0f} TL")
        avg_savings_rate = (1 - (proj_df['Gider'].mean() / proj_df['Gelir'].mean())) * 100
        m3.metric("Bütçe Sağlığı", f"%{avg_savings_rate:.1f}", delta="Tasarruf Oranı")

        st.divider()

        # GRAFİK 1: AYLIK GİDER TRENDİ
        st.subheader("📉 Aylık Gider Değişimi (Trend)")
        fig_line = px.line(proj_df, x='Ay', y='Gider', markers=True, 
                           line_shape="spline", color_discrete_sequence=['#E74C3C'])
        fig_line.update_layout(hovermode="x unified")
        st.plotly_chart(fig_line, use_container_width=True)

        col_left, col_right = st.columns(2)
        
        with col_left:
            # GRAFİK 2: HARCAMA TİPİ DAĞILIMI
            st.subheader("⚖️ Gider Yapısı (Sabit vs Değişken)")
            tip_df = df.groupby('tip')['miktar'].sum().reset_index()
            fig_donut = px.pie(tip_df, values='miktar', names='tip', hole=0.5,
                               color_discrete_map={'Sabit': '#34495E', 'Değişken': '#F1C40F'})
            st.plotly_chart(fig_donut, use_container_width=True)

        with col_right:
            # GRAFİK 3: GELİR/GİDER KIYASLAMASI
            st.subheader("📊 Nakit Giriş/Çıkış Dengesi")
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=proj_df['Ay'], y=proj_df['Gelir'], name='Gelir', marker_color='#27AE60'))
            fig_bar.add_trace(go.Bar(x=proj_df['Ay'], y=proj_df['Gider'], name='Gider', marker_color='#E67E22'))
            fig_bar.update_layout(barmode='group')
            st.plotly_chart(fig_bar, use_container_width=True)

        # GRAFİK 4: VARLIK PROJEKSİYONU
        st.subheader("💰 Toplam Varlık Gelişimi")
        fig_area = px.area(proj_df, x="Ay", y="Bakiye", color_discrete_sequence=['#2ECC71'])
        st.plotly_chart(fig_area, use_container_width=True)

    else:
        st.info("Analiz için henüz veri girişi yapılmadı.")

# --- TAB 3: ZAM YÖNETİMİ ---
with tab3:
    st.subheader("🛠️ Gider Bazlı Zam Tanımlama")
    df_raw = st.session_state.gider_listesi
    if not df_raw.empty:
        benzersiz_kalemler = sorted(df_raw['gider_kalemi'].unique())
        secilen_kalem = st.selectbox("Zam Uygulanacak Gider Kalemini Seçin:", benzersiz_kalemler)
        item_data = df_raw[df_raw['gider_kalemi'] == secilen_kalem].iloc[0]
        st.info(f"Kalem: **{secilen_kalem}** | Aktif Zam: **{item_data['zam_ayi']}** ayında **%{item_data['zam_orani']}**")
        tr_aylar = ["Yok", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
        c_z1, c_z2 = st.columns(2)
        with c_z1:
            curr_month_idx = tr_aylar.index(item_data['zam_ayi']) if item_data['zam_ayi'] in tr_aylar else 0
            yeni_z_ayi = st.selectbox("Zam Uygulanacak Ay:", tr_aylar, index=curr_month_idx)
        with c_z2:
            yeni_z_orani = st.number_input("Zam Oranı (%):", min_value=0.0, value=float(item_data['zam_orani']))
        if st.button(f"{secilen_kalem} Ayarlarını Güncelle"):
            update_item_zam(secilen_kalem, yeni_z_ayi, yeni_z_orani)
            st.session_state.gider_listesi = load_data()
            st.success("Zam ayarları güncellendi!")
            st.rerun()
    else:
        st.info("Kayıt bulunmuyor.")

# --- TAB 4: KAYIT SİLME ---
with tab4:
    if not st.session_state.gider_listesi.empty:
        all_df = st.session_state.gider_listesi.copy()
        options = all_df['ID'].tolist()
        def label_f(id_val):
            r = all_df[all_df['ID'] == id_val].iloc[0]
            return f"{r['odeme_tarihi'].strftime('%Y-%m-%d')} | {r['gider_kalemi']} | {r['miktar']} TL"
        del_id = st.selectbox("Silinecek Kayıt:", options=options, format_func=label_f)
        if st.button("Seçileni Sil"):
            delete_from_db(del_id)
            st.session_state.gider_listesi = load_data()
            st.rerun()