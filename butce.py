import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import sqlite3
import os
import ollama

# --- DOSYA YOLU VE SAYFA AYARI ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hane_finans.db")
st.set_page_config(page_title="Hanehalkı Finans Merkezi + AI", layout="wide", page_icon="🏠")

# --- VERİTABANI SİSTEMİ ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS giderler
                 (ID TEXT PRIMARY KEY, kisi TEXT, odeme_tarihi TEXT, gider_kalemi TEXT, 
                  miktar REAL, zam_ayi TEXT, zam_orani REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS gelirler
                 (kisi TEXT PRIMARY KEY, aylik_gelir REAL, birikim REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS maas_zamlari
                 (kisi TEXT, zam_tarihi TEXT, zam_orani REAL, PRIMARY KEY(kisi, zam_tarihi))''')
    conn.commit()
    conn.close()

def load_all_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    gider_df = pd.read_sql_query("SELECT * FROM giderler", conn)
    gelir_df = pd.read_sql_query("SELECT * FROM gelirler", conn)
    zam_df = pd.read_sql_query("SELECT * FROM maas_zamlari", conn)
    conn.close()
    if not gider_df.empty:
        gider_df['odeme_tarihi'] = pd.to_datetime(gider_df['odeme_tarihi'])
    return gider_df, gelir_df, zam_df

# --- PROJEKSİYON HESAPLAMA MOTORU (Global Erişim İçin) ---
def calculate_projection(gider_df, gelir_df, maas_zam_df, secilen_kisi="Tüm Hane"):
    if gider_df.empty or gelir_df.empty:
        return pd.DataFrame()

    min_date = gider_df['odeme_tarihi'].min().replace(day=1)
    max_date = gider_df['odeme_tarihi'].max().replace(day=1)
    current_date = min_date
    date_range = []
    while current_date <= max_date or len(date_range) < 12:
        date_range.append(current_date)
        current_date += relativedelta(months=1)

    tr_aylar = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
    hane_proj = []
    
    if secilen_kisi == "Tüm Hane":
        aktif_gelirler = gelir_df.set_index('kisi')['aylik_gelir'].to_dict()
        kasa = gelir_df['birikim'].sum()
    else:
        kisi_data = gelir_df[gelir_df['kisi'] == secilen_kisi].iloc[0]
        aktif_gelirler = {secilen_kisi: kisi_data['aylik_gelir']}
        kasa = kisi_data['birikim']

    for d in date_range:
        ay_str = d.strftime("%Y-%m")
        ay_ad = tr_aylar[d.month-1]
        
        for k in aktif_gelirler:
            ozel_zam = maas_zam_df[(maas_zam_df['kisi'] == k) & (maas_zam_df['zam_tarihi'] == ay_str)]
            if not ozel_zam.empty:
                aktif_gelirler[k] *= (1 + ozel_zam['zam_orani'].values[0]/100)
        
        toplam_gelir = sum(aktif_gelirler.values())
        
        if secilen_kisi == "Tüm Hane":
            aylik_g_df = gider_df[gider_df['odeme_tarihi'].dt.strftime('%Y-%m') == ay_str]
        else:
            aylik_g_df = gider_df[(gider_df['odeme_tarihi'].dt.strftime('%Y-%m') == ay_str) & (gider_df['kisi'] == secilen_kisi)]
        
        toplam_gider = 0
        for _, r in aylik_g_df.iterrows():
            tut = r['miktar']
            if r['zam_ayi'] == ay_ad: tut *= (1 + r['zam_orani']/100)
            toplam_gider += tut
        
        tasarruf = toplam_gelir - toplam_gider
        kasa += tasarruf
        hane_proj.append({"Ay": ay_str, "Gelir": toplam_gelir, "Gider": toplam_gider, "Bakiye": kasa, "Tasarruf": tasarruf})
    
    return pd.DataFrame(hane_proj)

# --- BAŞLATMA ---
init_db()
gider_df, gelir_df, maas_zam_df = load_all_data()

tabs = st.tabs(["👤 Kişisel Tanımlamalar", "💸 Ödeme Girişi", "📊 Detaylı Analiz", "🤖 AI Finans Koçu", "⚙️ Yönetim"])

# --- TAB 1: KİŞİSEL ---
with tabs[0]:
    st.subheader("👥 Hane Üyeleri")
    c1, c2 = st.columns(2)
    with c1:
        with st.form("kisi_ekle"):
            k_ad = st.text_input("Kişi Adı")
            k_gelir = st.number_input("Maaş (TL)", min_value=0.0)
            k_birikim = st.number_input("Birikim (TL)", min_value=0.0)
            if st.form_submit_button("Kaydet"):
                conn = sqlite3.connect(DB_PATH)
                conn.execute("INSERT OR REPLACE INTO gelirler VALUES (?, ?, ?)", (k_ad, k_gelir, k_birikim))
                conn.commit(); conn.close()
                st.rerun()
    with c2:
        if not gelir_df.empty: st.dataframe(gelir_df, use_container_width=True, hide_index=True)

# --- TAB 2: ÖDEME ---
with tabs[1]:
    if not gelir_df.empty:
        st.subheader("📝 Ödeme Girişi")
        with st.form("gider_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                g_kisi = st.selectbox("Ödeyen", gelir_df['kisi'])
                g_kalem = st.text_input("Gider Tanımı")
            with col2:
                g_miktar = st.number_input("Tutar (TL)", min_value=0.0)
                g_tarih = st.date_input("Başlangıç", date.today())
            with col3:
                g_sure = st.number_input("Süre (Ay)", min_value=1, value=1)
            if st.form_submit_button("Sisteme İşle"):
                conn = sqlite3.connect(DB_PATH)
                ts = int(datetime.now().timestamp())
                for i in range(int(g_sure)):
                    y_tar = g_tarih + relativedelta(months=i)
                    yid = f"{ts}_{i}_{g_kisi[:2]}"
                    conn.execute("INSERT INTO giderler VALUES (?,?,?,?,?,?,?)", (yid, g_kisi, y_tar.strftime('%Y-%m-%d'), g_kalem, g_miktar, "Yok", 0))
                conn.commit(); conn.close()
                st.rerun()

# --- TAB 3: ANALİZ ---
with tabs[2]:
    if not gider_df.empty and not gelir_df.empty:
        sel_k = st.selectbox("Analiz Kapsamı Seçin:", ["Tüm Hane"] + list(gelir_df['kisi'].unique()))
        h_df = calculate_projection(gider_df, gelir_df, maas_zam_df, sel_k)
        
        # 1. ÜST METRİKLER VE GAUGE
        st.subheader("📌 Temel Finansal Göstergeler")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Aylık Ortalama Gider", f"{h_df['Gider'].mean():,.0f} TL")
        m2.metric("Projeksiyon Sonu Bakiye", f"{h_df['Bakiye'].iloc[-1]:,.0f} TL", delta=f"{h_df['Bakiye'].iloc[-1] - h_df['Bakiye'].iloc[0]:,.0f}")
        
        savings_rate = (h_df['Tasarruf'].sum() / h_df['Gelir'].sum()) * 100
        m3.metric("Genel Tasarruf Oranı", f"%{savings_rate:.1f}")
        
        expense_ratio = (h_df['Gider'].mean() / h_df['Gelir'].mean()) * 100
        m4.metric("Gider/Gelir Dengesi", f"%{expense_ratio:.1f}", delta="-İyi" if expense_ratio < 70 else "+Riskli", delta_color="inverse")

        st.divider()

        # 2. ANA NAKİT AKIŞI GRAFİĞİ (GELİR - GİDER - BAKİYE)
        st.subheader("📉 Nakit Akışı ve Varlık İlerlemesi")
        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(x=h_df['Ay'], y=h_df['Gelir'], name="Aylık Gelir", line=dict(color="#27AE60", width=4, shape='hv')))
        fig_main.add_trace(go.Bar(x=h_df['Ay'], y=h_df['Gider'], name="Aylık Gider", marker_color="#E74C3C", opacity=0.6))
        fig_main.add_trace(go.Scatter(x=h_df['Ay'], y=h_df['Bakiye'], name="Kümülatif Varlık", yaxis="y2", line=dict(color="#3498DB", dash="dot", width=3)))
        
        fig_main.update_layout(
            yaxis2=dict(title="Kümülatif Varlık (TL)", overlaying="y", side="right"),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_main, use_container_width=True)
        

        st.divider()

        # 3. PASTA GRAFİKLER (DAĞILIM)
        col_pie1, col_pie2 = st.columns(2)
        with col_pie1:
            st.subheader("🍕 Gider Kalemi Dağılımı")
            sec_ay = st.selectbox("Dağılım Ayı Seçin:", h_df['Ay'], index=0)
            ay_g = gider_df[gider_df['odeme_tarihi'].dt.strftime('%Y-%m') == sec_ay]
            if sel_k != "Tüm Hane": ay_g = ay_g[ay_g['kisi'] == sel_k]
            
            if not ay_g.empty:
                st.plotly_chart(px.pie(ay_g, values='miktar', names='gider_kalemi', hole=0.5, color_discrete_sequence=px.colors.qualitative.Set3), use_container_width=True)
                
            else:
                st.info("Seçilen ayda harcama kaydı yok.")

        with col_pie2:
            st.subheader("👥 Hane Paylaşım Analizi")
            if sel_k == "Tüm Hane":
                ay_kisi = ay_g.groupby('kisi')['miktar'].sum().reset_index()
                st.plotly_chart(px.pie(ay_kisi, values='miktar', names='kisi', hole=0.5, color_discrete_sequence=px.colors.sequential.RdBu), use_container_width=True)
                
            else:
                st.write(f"**{sel_k}** için bu ayki toplam harcama: **{ay_g['miktar'].sum():,.2f} TL**")

        st.divider()

        # 4. TASARRUF TRENDİ VE BİRİKİM HIZI
        st.subheader("💰 Aylık Tasarruf ve Birikim Trendi")
        fig_tas = px.area(h_df, x="Ay", y="Tasarruf", title="Aylık Net Birikim (Gelir - Gider)", color_discrete_sequence=['#F1C40F'])
        fig_tas.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Kritik Eşik")
        st.plotly_chart(fig_tas, use_container_width=True)
        

        st.subheader("📋 Detaylı Projeksiyon Tablosu")
        st.dataframe(h_df.style.format({"Gelir": "{:,.0f} TL", "Gider": "{:,.0f} TL", "Bakiye": "{:,.0f} TL", "Tasarruf": "{:,.0f} TL"}), use_container_width=True)

    else:
        st.warning("Lütfen 'Tanımlamalar' ve 'Ödemeler' sekmelerinden veri girişi yapın.")
# --- TAB 4: AI KOÇ (OLLAMA) ---
with tabs[3]:
    st.subheader("🤖 AI Finansal Danışman (Ollama)")
    # Analiz kısmındaki hesaplamayı AI için tekrar yapıyoruz (NameError engellemek için)
    h_df_ai = calculate_projection(gider_df, gelir_df, maas_zam_df, "Tüm Hane")
    
    if not h_df_ai.empty:
        if st.button("Bütçemi Analiz Et"):
            with st.spinner("Llama 3 verileri yorumluyor..."):
                prompt = f"""
                Verilerimi analiz et ve Türkçe tavsiyeler ver:
                - Aylık Gelir: {h_df_ai['Gelir'].mean():,.0f} TL
                - Aylık Gider: {h_df_ai['Gider'].mean():,.0f} TL
                - Tasarruf Oranı: %{(h_df_ai['Tasarruf'].sum()/h_df_ai['Gelir'].sum())*100:.1f}
                - Gelecek 12 ay sonundaki para: {h_df_ai['Bakiye'].iloc[-1]:,.0f} TL
                - En borçlu/harcamalı ay: {h_df_ai.loc[h_df_ai['Gider'].idxmax(), 'Ay']}
                """
                try:
                    res = ollama.chat(model='llama3', messages=[
                        {'role': 'system', 'content': 'Sen bir finans koçusun. Kısa ve aksiyon odaklı konuş.'},
                        {'role': 'user', 'content': prompt}
                    ])
                    st.success("Analiz Tamamlandı")
                    st.write(res['message']['content'])
                except Exception as e:
                    st.error(f"Ollama'ya ulaşılamadı. Hata: {e}")
    else:
        st.info("AI analizi için önce veri girin.")

# --- TAB 5: YÖNETİM ---
with tabs[4]:
    st.subheader("⚙️ Yönetim")
    # Zam Yönetimi (Gider & Maaş) ve Silme buraya...
    c_y1, c_y2 = st.columns(2)
    with c_y1:
        st.write("💰 **Maaş Zammı**")
        if not gelir_df.empty:
            with st.form("m_zam"):
                mzk = st.selectbox("Kişi", gelir_df['kisi'])
                mzt = st.date_input("Ay").strftime("%Y-%m")
                mzo = st.number_input("Artış %")
                if st.form_submit_button("Kaydet"):
                    conn = sqlite3.connect(DB_PATH); conn.execute("INSERT OR REPLACE INTO maas_zamlari VALUES (?,?,?)", (mzk, mzt, mzo))
                    conn.commit(); conn.close(); st.rerun()
    with c_y2:
        st.write("🛒 **Gider Artışı**")
        if not gider_df.empty:
            with st.form("g_zam"):
                gzk = st.selectbox("Kalem", sorted(gider_df['gider_kalemi'].unique()))
                gza = st.selectbox("Ay", ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"])
                gzo = st.number_input("Artış %")
                if st.form_submit_button("Uygula"):
                    conn = sqlite3.connect(DB_PATH); conn.execute("UPDATE giderler SET zam_ayi = ?, zam_orani = ? WHERE gider_kalemi = ?", (gza, gzo, gzk))
                    conn.commit(); conn.close(); st.rerun()

    st.divider()
    if not gider_df.empty:
        sel = st.dataframe(gider_df, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="multi-row")
        if sel.selection.rows and st.button("Seçili Kayıtları Sil"):
            ids = gider_df.iloc[sel.selection.rows]['ID'].tolist()
            conn = sqlite3.connect(DB_PATH); conn.executemany("DELETE FROM giderler WHERE ID = ?", [(x,) for x in ids])
            conn.commit(); conn.close(); st.rerun()