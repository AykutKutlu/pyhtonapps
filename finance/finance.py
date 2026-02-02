from core.utils import (
    kapsamli_teknik_analiz,
    piyasa_radari_tara,
    calculate_fibonacci_levels,
    dinamik_trend_analizi,
    tarihsel_seviye_analizi,
    gelismis_strateji_motoru,
    osilatör_analizi,
    hacim_profili_hesapla,
    coklu_strateji_analizi,
    hibrit_sinyal_motoru
)
from core import interface
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

from datetime import datetime
import numpy as np
import core.utils as utils
import importlib
importlib.reload(utils)

def get_symbol_lists(market_type):
    """Piyasa türüne göre sembol listesini döner."""
    if market_type == "BIST 100":
        return [
            "GARAN.IS", "KCHOL.IS", "THYAO.IS", "FROTO.IS", "ISCTR.IS", "BIMAS.IS", "TUPRS.IS", "ENKAI.IS", "ASELS.IS", "AKBNK.IS", 
            "YKBNK.IS", "VAKBN.IS", "TCELL.IS", "SAHOL.IS", "SASA.IS", "TTKOM.IS", "EREGL.IS", "CCOLA.IS", "PGSUS.IS", "SISE.IS", 
            "AEFES.IS", "HALKB.IS", "TOASO.IS", "ARCLK.IS", "TAVHL.IS", "ASTOR.IS", "MGROS.IS", "TTRAK.IS", "AGHOL.IS", "OYAKC.IS", 
            "KOZAL.IS", "ENJSA.IS", "BRSAN.IS", "TURSG.IS", "GUBRF.IS", "MPARK.IS", "OTKAR.IS", "BRYAT.IS", "ISMEN.IS", "PETKM.IS", 
            "ULKER.IS", "CLEBI.IS", "DOAS.IS", "AKSEN.IS", "ANSGR.IS", "ALARK.IS", "EKGYO.IS", "TABGD.IS", "RGYAS.IS", "DOHOL.IS", 
            "TSKB.IS", "ENERY.IS", "KONYA.IS", "EGEEN.IS", "AKSA.IS", "CIMSA.IS", "HEKTS.IS", "MAVI.IS", "VESBE.IS", "KONTR.IS", 
            "TKFEN.IS", "BTCIM.IS", "ECILC.IS", "KCAER.IS", "KRDMD.IS", "SOKM.IS", "KOZAA.IS", "SMRTG.IS", "CWENE.IS", "ZOREN.IS", 
            "EUPWR.IS", "REDR.IS", "VESTL.IS", "MIATK.IS", "ALFAS.IS", "GESAN.IS", "OBAM.IS", "AKFYE.IS", "KLSER.IS", "AGROT.IS", 
            "YEOTK.IS", "BINHO1000.IS", "KARSN.IS", "TMSN.IS", "SKBNK.IS", "FENER.IS", "CANTE.IS", "TUKAS.IS", "KTLEV.IS", "ADEL.IS", 
            "BERA.IS", "ODAS.IS", "AKFGY.IS", "GOLTS.IS", "ARDYZ.IS", "BJKAS.IS", "PEKGY.IS", "PAPIL.IS", "LMKDC.IS", "ALTNY.IS", 
            "NTHOL.IS", "TRKCM.IS", "AYGAZ.IS", "TGSAS.IS", "BAGFS.IS", "ISDMR.IS", "KERVN.IS", "LOGO.IS", "NIBAS.IS", "PRKME.IS", 
            "SNGYO.IS", "TSPOR.IS", "ULUFA.IS", "VAKKO.IS", "YATAS.IS", "FORTE.IS"
        ]
    elif market_type == "Kripto Paralar":
        return [
            "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD",
            "LTC-USD", "BCH-USD", "LINK-USD", "ICP-USD", "ARB-USD", "XLM-USD", "HBAR-USD", "FIL-USD", "VET-USD", "INJ-USD",
            "APT-USD", "PEPE-USD", "RNDR-USD", "QNT-USD", "ALGO-USD", "IMX-USD", "AAVE-USD", "GRT-USD", "MKR-USD", "EGLD-USD",
            "FTM-USD", "THETA-USD", "SAND-USD", "AXS-USD", "NEAR-USD", "CHZ-USD", "LDO-USD", "FET-USD"
        ]
    elif market_type == "Emtialar (Maden/Enerji)":
        return [
            "GC=F", "SI=F", "HG=F", "PL=F", "PA=F", # Madenler
            "CL=F", "NG=F", "RB=F", "HO=F",         # Enerji
            "ZC=F", "ZS=F", "KE=F", "KC=F", "CT=F"  # Tarım
        ]
    return []
    
def get_ui_names():
    return {
        # --- MADENLER, ENERJİ VE EMTİALAR ---
        "GC=F": "Altın ONS (Gold)",
        "SI=F": "Gümüş ONS (Silver)",
        "HG=F": "Bakır (Copper)",
        "PL=F": "Platin (Platinum)",
        "PA=F": "Paladyum (Palladium)",
        "CL=F": "Ham Petrol (Crude Oil)",
        "NG=F": "Doğalgaz (Natural Gas)",
        "RB=F": "RBOB Benzin",
        "HO=F": "Isınma Yakıtı",
        "ZC=F": "Mısır (Corn)",
        "ZS=F": "Soya Fasulyesi",
        "KE=F": "Buğday (Wheat)",
        "KC=F": "Kahve (Coffee)",
        "CT=F": "Pamuk (Cotton)",

        # --- KRİPTO PARALAR ---
        "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "BNB-USD": "Binance Coin",
        "SOL-USD": "Solana", "XRP-USD": "Ripple", "ADA-USD": "Cardano",
        "DOGE-USD": "Dogecoin", "AVAX-USD": "Avalanche", "DOT-USD": "Polkadot",
        "MATIC-USD": "Polygon", "LTC-USD": "Litecoin", "BCH-USD": "Bitcoin Cash",
        "LINK-USD": "Chainlink", "ICP-USD": "Internet Computer", "ARB-USD": "Arbitrum",
        "XLM-USD": "Stellar", "HBAR-USD": "Hedera", "FIL-USD": "Filecoin",
        "VET-USD": "VeChain", "INJ-USD": "Injective", "APT-USD": "Aptos",
        "PEPE-USD": "Pepe", "RNDR-USD": "Render", "QNT-USD": "Quant",
        "ALGO-USD": "Algorand", "IMX-USD": "Immutable", "AAVE-USD": "Aave",
        "GRT-USD": "The Graph", "MKR-USD": "Maker", "EGLD-USD": "MultiversX",
        "FTM-USD": "Fantom", "THETA-USD": "Theta Network", "SAND-USD": "The Sandbox",
        "AXS-USD": "Axie Infinity", "NEAR-USD": "Near Protocol", "CHZ-USD": "Chiliz",
        "LDO-USD": "Lido DAO", "FET-USD": "Fetch.ai",

        # --- BIST 100 ŞİRKETLERİ ---
        "GARAN.IS": "Garanti BBVA", "KCHOL.IS": "Koç Holding", "THYAO.IS": "Türk Hava Yolları",
        "FROTO.IS": "Ford Otosan", "ISCTR.IS": "İş Bankası (C)", "BIMAS.IS": "BİM Mağazalar",
        "TUPRS.IS": "Tüpraş", "ENKAI.IS": "Enka İnşaat", "ASELS.IS": "Aselsan",
        "AKBNK.IS": "Akbank", "YKBNK.IS": "Yapı Kredi Bankası", "VAKBN.IS": "Vakıfbank",
        "TCELL.IS": "Turkcell", "SAHOL.IS": "Sabancı Holding", "SASA.IS": "Sasa Polyester",
        "TTKOM.IS": "Türk Telekom", "EREGL.IS": "Erdemir", "CCOLA.IS": "Coca-Cola İçecek",
        "PGSUS.IS": "Pegasus", "SISE.IS": "Şişecam", "AEFES.IS": "Anadolu Efes",
        "HALKB.IS": "Halkbank", "TOASO.IS": "Tofaş Oto", "ARCLK.IS": "Arçelik",
        "TAVHL.IS": "TAV Havalimanları", "ASTOR.IS": "Astor Enerji", "MGROS.IS": "Migros",
        "TTRAK.IS": "Türk Traktör", "AGHOL.IS": "Anadolu Grubu Hol.", "OYAKC.IS": "Oyak Çimento",
        "KOZAL.IS": "Koza Altın", "ENJSA.IS": "Enerjisa Enerji", "BRSAN.IS": "Borusan Boru",
        "TURSG.IS": "Türkiye Sigorta", "GUBRF.IS": "Gübre Fabrikaları", "MPARK.IS": "MLP Care (Medical Park)",
        "OTKAR.IS": "Otokar", "BRYAT.IS": "Borusan Yatırım", "ISMEN.IS": "İş Menkul Değerler",
        "PETKM.IS": "Petkim", "ULKER.IS": "Ülker Bisküvi", "CLEBI.IS": "Çelebi Hava Servisi",
        "DOAS.IS": "Doğuş Otomotiv", "AKSEN.IS": "Aksa Enerji", "ANSGR.IS": "Anadolu Sigorta",
        "ALARK.IS": "Alarko Holding", "EKGYO.IS": "Emlak Konut GYO", "TABGD.IS": "Tab Gıda",
        "RGYAS.IS": "Rönesans Gayrimenkul", "DOHOL.IS": "Doğan Holding", "TSKB.IS": "TSKB",
        "ENERY.IS": "Enerya Enerji", "KONYA.IS": "Konya Çimento", "EGEEN.IS": "Ege Endüstri",
        "AKSA.IS": "Aksa", "CIMSA.IS": "Çimsa", "HEKTS.IS": "Hektaş",
        "MAVI.IS": "Mavi Giyim", "VESBE.IS": "Vestel Beyaz Eşya", "KONTR.IS": "Kontrolmatik",
        "TKFEN.IS": "Tekfen Holding", "BTCIM.IS": "Batıçim", "ECILC.IS": "Eczacıbaşı İlaç",
        "KCAER.IS": "Kocaer Çelik", "KRDMD.IS": "Kardemir (D)", "SOKM.IS": "Şok Marketler",
        "KOZAA.IS": "Koza Madencilik", "SMRTG.IS": "Smart Güneş Enerjisi", "CWENE.IS": "CW Enerji",
        "ZOREN.IS": "Zorlu Enerji", "EUPWR.IS": "Europower Enerji", "REDR.IS": "Reeder Teknoloji",
        "VESTL.IS": "Vestel", "MIATK.IS": "Mia Teknoloji", "ALFAS.IS": "Alfa Solar Enerji",
        "GESAN.IS": "Girişim Elektrik", "OBAM.IS": "Oba Makarnacılık", "AKFYE.IS": "Akfen Yen. Enerji",
        "KLSER.IS": "Kaleseramik", "AGROT.IS": "Agrotech", "YEOTK.IS": "Yeo Teknoloji",
        "BINHO1000.IS": "1000 Yatırımlar Hol.", "KARSN.IS": "Karsan", "TMSN.IS": "Tümosan",
        "SKBNK.IS": "Şekerbank", "FENER.IS": "Fenerbahçe", "CANTE.IS": "Çan2 Termik",
        "TUKAS.IS": "Tukaş", "KTLEV.IS": "Katılımevim", "ADEL.IS": "Adel Kalemcilik",
        "BERA.IS": "Bera Holding", "ODAS.IS": "Odaş Elektrik", "AKFGY.IS": "Akfen GYO",
        "GOLTS.IS": "Göltaş Çimento", "ARDYZ.IS": "Ardyz Yazılım", "BJKAS.IS": "Beşiktaş JK",
        "PEKGY.IS": "Peker GYO", "PAPIL.IS": "Papilon Savunma", "LMKDC.IS": "Limak Doğu Anadolu Çimento",
        "ALTNY.IS": "Altınay Savunma", "NTHOL.IS": "Net Holding", "AYGAZ.IS": "Aygaz",
        "LOGO.IS": "Logo Yazılım", "SNGYO.IS": "Sinpaş GYO", "VAKKO.IS": "Vakko", "YATAS.IS": "Yataş", "FORTE.IS": "Forte bilişim"
    }  
@st.cache_data
def load_data(symbol, start_date="2020-01-01"):
    """
    Önbelleğe alarak bir sembol için geçmiş verileri indirir.
    yfinance'in kendi oturum yönetimine (curl_cffi) güvenerek kararlılığı artırır.
    """
    try:
        ticker = yf.Ticker(symbol)
        
        data = ticker.history(period="5y") 
        
        if data.empty:
            return None
        
        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)
            
        return data.loc[start_date:] 
        
    except Exception as e:
        print(f"Veri yükleme hatası ({symbol}): {e}")
        return None

st.set_page_config(page_title="Finansal Analiz Pro", layout="wide")

interface.apply_custom_css()
st.title("📈 Hisse & Kripto Tahminleme ve Stratejiler")

if 'tahmin_sonucu' not in st.session_state:
    st.session_state.tahmin_sonucu = None
if 'tahmin_yorumu' not in st.session_state:
    st.session_state.tahmin_yorumu = None
if 'strateji_grafigi' not in st.session_state:
    st.session_state.strateji_grafigi = None
if 'strateji_yorumu' not in st.session_state:
    st.session_state.strateji_yorumu = None
if 'secilen_sembol' not in st.session_state:
    st.session_state.secilen_sembol = None

# ===== RADAR'DAN GELEN VERİLERİ ÖNCELİKLE İŞLE =====
if st.session_state.get("selected_market_radar"):
    st.session_state.current_market = st.session_state.selected_market_radar
    st.session_state.market_selectbox = st.session_state.selected_market_radar
    st.session_state.selected_market_radar = None

if st.session_state.get("selected_symbol_radar"):
    st.session_state.current_symbol = st.session_state.selected_symbol_radar
    st.session_state.symbol_selectbox = st.session_state.selected_symbol_radar
    st.session_state.selected_symbol_radar = None

with st.sidebar:
    st.header("🎮 Terminal Kontrol")
    
    # Market type varsayılanları
    market_options = ["BIST 100", "Kripto Paralar", "Emtialar (Maden/Enerji)"]
    
    # ===== İLK KEZI İNCELE VE BAŞLAT =====
    if "current_market" not in st.session_state:
        st.session_state.current_market = market_options[0]
    
    if "current_symbol" not in st.session_state:
        st.session_state.current_symbol = get_symbol_lists(market_options[0])[0]
    
    # ===== SIDEBAR KONTROLLER =====
    def on_market_change():
        # Piyasa değişince sembolü sıfırla
        new_market = st.session_state.market_selectbox
        if new_market != st.session_state.current_market:
            st.session_state.current_market = new_market
            # Yeni piyasanın ilk sembolünü seç
            st.session_state.current_symbol = get_symbol_lists(new_market)[0]
            st.session_state.symbol_selectbox = st.session_state.current_symbol
    
    market_type = st.selectbox(
        "📊 Piyasa Seçiniz", 
        market_options, 
        index=market_options.index(st.session_state.current_market),
        key="market_selectbox",
        on_change=on_market_change
    )
    
    symbols = get_symbol_lists(st.session_state.current_market)
    ui_names = get_ui_names()
    
    # Sembol listesinde mevcut sembolü kontrol et
    if st.session_state.current_symbol not in symbols:
        st.session_state.current_symbol = symbols[0]
    
    # Session state'i selectbox key'i ile senkronize et
    if "symbol_selectbox" not in st.session_state:
        st.session_state.symbol_selectbox = st.session_state.current_symbol
    
    def on_symbol_change():
        # Sembol seçimi değişince state'i güncelle
        st.session_state.current_symbol = st.session_state.symbol_selectbox

    selected_symbol = st.selectbox(
        "📌 Sembol Seçiniz", 
        symbols, 
        index=symbols.index(st.session_state.symbol_selectbox) if st.session_state.symbol_selectbox in symbols else 0,
        format_func=lambda x: ui_names.get(x, x),
        key="symbol_selectbox",
        on_change=on_symbol_change
    )

if 'secilen_sembol' not in st.session_state:
    st.session_state.secilen_sembol = st.session_state.current_symbol

if st.session_state.secilen_sembol != st.session_state.current_symbol:
    st.session_state.tahmin_sonucu = None
    st.session_state.tahmin_yorumu = None
    st.session_state.strateji_grafigi = None
    st.session_state.strateji_yorumu = None
    st.session_state.secilen_sembol = st.session_state.current_symbol

# Ana sekmeleri yeniden düzenliyoruz.
tab_names = ["📈 Analiz Paneli", "🎯 Yatırım Radarı"]
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = tab_names[0]

# HATA DÜZELTMESİ: Butondan gelen sekme değiştirme isteğini burada işliyoruz.
if 'next_tab' in st.session_state:
    st.session_state.active_tab = st.session_state.next_tab
    del st.session_state.next_tab 

selected_tab = st.radio(
    "Sekmeler", 
    tab_names, 
    key="active_tab", 
    horizontal=True, 
    label_visibility="collapsed"
)


from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- VERİ YÖNETİMİ (TÜM SEKMELER İÇİN ORTAK) ---
if "chart_data" not in st.session_state or st.session_state.get("last_symbol") != st.session_state.current_symbol:
    raw_data = yf.download(st.session_state.current_symbol, period="2y", interval="1d")
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)
    st.session_state["chart_data"] = raw_data
    st.session_state["last_symbol"] = st.session_state.current_symbol

data = st.session_state["chart_data"]


if selected_tab == "📈 Analiz Paneli":
    
    # Genel özet bilgileri hesapla
    if not data.empty:
        analiz_ozet = kapsamli_teknik_analiz(data)
        konsensus_ozet = coklu_strateji_analizi(data)
        
        current_price = data.iloc[-1]['Close']
        hedef_fiyat = analiz_ozet.get('hedef', 0)
        rsi_deger = analiz_ozet['df'].iloc[-1]['RSI']
        skor = konsensus_ozet.get('skor', 0)
        
        # Her subtab'ın sinyalini al
        teknik_durum = analiz_ozet.get('durum', 'NÖTR')
        strateji_durum = konsensus_ozet.get('final_signal', 'NÖTR')
        
        # Sinyalleri normalize et (AL/BUY, SAT/SELL olabilir)
        teknik_sinyal = "BUY" if "AL" in teknik_durum or "BUY" in teknik_durum else "SELL" if "SAT" in teknik_durum or "SELL" in teknik_durum else "NEUTRAL"
        strateji_sinyal = "BUY" if "AL" in strateji_durum or "BUY" in strateji_durum else "SELL" if "SAT" in strateji_durum or "SELL" in strateji_durum else "NEUTRAL"
        
        # Sinyaller uyuşmuyorsa uyarı ekle
        sinyal_uyusmus = teknik_sinyal == strateji_sinyal
        
        # Renk seçimi
        if sinyal_uyusmus:
            durum_rengi = "#00FF88" if teknik_sinyal == "BUY" else "#FF3D00" if teknik_sinyal == "SELL" else "#FFD600"
            durum_metni = f"{teknik_durum}"
        else:
            durum_rengi = "#FFA500"  # Turuncu - uyuşmazlık
            durum_metni = f"⚠️ {teknik_durum} / {strateji_durum}"
        
        # Başlık
        st.markdown(f"## 📊 {st.session_state.current_symbol} - Genel Durum Özeti")
        
        # Metrik kartları
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="📈 Güncel Fiyat",
                value=f"{current_price:.2f}",
                delta=None
            )
        
        with col2:
            fiyat_fark = hedef_fiyat - current_price
            fiyat_yuzde = (fiyat_fark / current_price * 100) if current_price != 0 else 0
            st.metric(
                label="🎯 Hedef Fiyat",
                value=f"{hedef_fiyat:.2f}",
                delta=f"{fiyat_yuzde:+.1f}%"
            )
        
        with col3:
            st.metric(
                label="📊 RSI",
                value=f"{int(rsi_deger)}",
                delta="Aşırı Alım" if rsi_deger > 70 else "Aşırı Satım" if rsi_deger < 30 else "Nötr"
            )
        
        with col4:
            st.metric(
                label="⭐ Strateji Skoru",
                value=f"{skor}/5",
                delta=None
            )
        
        with col5:
            if sinyal_uyusmus:
                st.markdown(f"""
                <div style='
                    text-align: center;
                    padding: 20px;
                    background-color: rgba(0, 0, 0, 0.3);
                    border-radius: 8px;
                    border-left: 4px solid {durum_rengi};
                '>
                    <div style='color: #888; font-size: 12px; margin-bottom: 8px;'>Mevcut Durum</div>
                    <div style='color: {durum_rengi}; font-weight: bold; font-size: 16px;'>{durum_metni}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='
                    text-align: center;
                    padding: 20px;
                    background-color: rgba(0, 0, 0, 0.3);
                    border-radius: 8px;
                    border-left: 4px solid {durum_rengi};
                '>
                    <div style='color: #888; font-size: 11px; margin-bottom: 6px;'>Sinyaller Uyuşmuyor!</div>
                    <div style='color: {durum_rengi}; font-weight: bold; font-size: 14px;'>{durum_metni}</div>
                    <div style='color: #FFB74D; font-size: 10px; margin-top: 6px;'>Teknik ≠ Strateji</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
    
    # "Analiz Paneli" altında iki alt sekme oluşturuyoruz.
    sub_tab_teknik, sub_tab_strateji = st.tabs(["Stratejik Teknik", "Strateji Laboratuvarı"])

    with sub_tab_teknik:
        st.header(f"🔍 {st.session_state.current_symbol} - Profesyonel Strateji Paneli")
        if not data.empty:
            # Teknik katmanları hesapla
            analiz = kapsamli_teknik_analiz(data)
            aktif_trendler = dinamik_trend_analizi(data)
            fibo_levels = calculate_fibonacci_levels(data) 
            seviyeler = tarihsel_seviye_analizi(data)
            df_plot = analiz['df'].tail(720) 

            # --- DÜZEN VE FİLTRELER ---
            col_chart, col_filter = st.columns([5, 1])

            with col_filter:
                st.markdown("### 🛠️ Katmanlar")
                f_sig = st.checkbox("Sinyal Okları", value=True, key="teknik_sig")
                f_levels = st.checkbox("🎯 Hedef & 🛑 Stop", value=True, key="teknik_levels")
                f_trend = st.checkbox("Trend Hatları", value=True, key="teknik_trend")
                f_seviye = st.checkbox("Destek/Direnç", value=True, key="teknik_seviye")
                f_fibo = st.checkbox("Fibonacci", value=False, key="teknik_fibo")
                f_sma50 = st.checkbox("SMA 50", value=True, key="teknik_sma50")
                f_sma200 = st.checkbox("SMA 200", value=True, key="teknik_sma200")
                
                st.divider()
                st.markdown("### 📉 Alt Göstergeler")
                show_rsi = st.checkbox("RSI Göster", value=True, key="teknik_rsi")
                show_macd = st.checkbox("MACD Göster", value=True, key="teknik_macd")

            with col_chart:
                # Durum Başlığı
                yon_rengi = "#00FF88" if analiz['signal_type'] == "BUY" else "#FF3D00"
                st.markdown(f"### Mevcut Durum: <span style='color:{yon_rengi};'>{analiz['durum']}</span>", unsafe_allow_html=True)

                # --- ALT GRAFİK YAPILANDIRMASI ---
                rows = 1
                row_heights = [0.7]
                if show_rsi: 
                    rows += 1
                    row_heights.append(0.15)
                if show_macd: 
                    rows += 1
                    row_heights.append(0.15)

                total = sum(row_heights)
                row_heights = [h/total for h in row_heights]

                fig = make_subplots(
                    rows=rows, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.03, row_heights=row_heights
                )

                # 1. ANA GRAFİK (CANDLESTICK)
                fig.add_trace(go.Candlestick(
                    x=df_plot.index, open=df_plot['Open'], high=df_plot['High'],
                    low=df_plot['Low'], close=df_plot['Close'], name="Fiyat",
                    increasing_line_color='#00FF88', decreasing_line_color='#FF3D00'
                ), row=1, col=1)

                # SMA Katmanları (Güvenli Kontrol)
                if f_sma50 and 'SMA50' in df_plot.columns:
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA50'], line=dict(color='#FFD600', width=1), name="SMA 50"), row=1, col=1)
                if f_sma200 and 'SMA200' in df_plot.columns:
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['SMA200'], line=dict(color='#E53935', width=1.5), name="SMA 200"), row=1, col=1)

                # SİNYAL OKLARI (TEKİL VE NET)
                if f_sig:
                    for sig in analiz['all_signals']:
                        if sig['date'] in df_plot.index:
                            is_match = sig.get('match', False)
                            
                            if sig['type'] == "BUY":
                                color, symbol, y_val, text, t_pos = "#1C8108" if is_match else "#641DA6", "triangle-up", sig['low'], "🚀 Buy" if is_match else "AL", "bottom center"
                            else:
                                color, symbol, y_val, text, t_pos = "#840808" if is_match else "#B3761A", "triangle-down", sig['high'], "⚠️ Sell" if is_match else "SAT", "top center"

                            fig.add_trace(go.Scatter(
                                x=[sig['date']], y=[y_val], mode="markers+text",
                                marker=dict(symbol=symbol, size=18 if is_match else 14, color=color, line=dict(width=1.5, color="white")),
                                text=[text], textposition=t_pos, textfont=dict(color=color, size=11, family="Arial Black"),
                                name="Hibrit Sinyal", showlegend=False
                            ), row=1, col=1)
                
                # HEDEF & STOP ÇİZGİLERİ
                if f_levels and analiz['signal_type'] != "NEUTRAL":
                    rect_color = "rgba(0, 255, 136, 0.1)" if analiz['signal_type'] == "BUY" else "rgba(255, 61, 0, 0.1)"
                    fig.add_hrect(y0=analiz['signal_price'], y1=analiz['hedef'], fillcolor=rect_color, line_width=0, row=1, col=1)
                    fig.add_shape(type="line", x0=analiz['signal_date'], y0=analiz['hedef'], x1=df_plot.index[-1], y1=analiz['hedef'], line=dict(color="#00FF88", width=2, dash="dash"), row=1, col=1)
                    fig.add_shape(type="line", x0=analiz['signal_date'], y0=analiz['stop'], x1=df_plot.index[-1], y1=analiz['stop'], line=dict(color="#FF3D00", width=2, dash="dashdot"), row=1, col=1)

                # TREND HATLARINI EKLE
                if f_trend:
                    for line in aktif_trendler:
                        fig.add_trace(go.Scatter(x=line['x'], y=line['y'], mode='lines', line=dict(color=line['color'], width=3), name="Trend", showlegend=False), row=1, col=1)

                # DESTEK/DİRENÇ SEVİYELERİ
                if f_seviye:
                    for lvl in seviyeler:
                        fig.add_shape(type="line", x0=lvl['date'], y0=lvl['val'], x1=df_plot.index[-1], y1=lvl['val'], line=dict(color=lvl['color'], width=1, dash="dot"), row=1, col=1)

                # FIBONACCI
                if f_fibo:
                    for lvl, val in fibo_levels.items():
                        fig.add_hline(y=val, line_width=0.5, line_dash="dot", line_color="rgba(255,255,255,0.3)", annotation_text=f"Fibo {lvl}", row=1, col=1)

                # ALT GÖSTERGELER
                current_row = 2
                if show_rsi:
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], line=dict(color='#B388FF', width=1.5), name="RSI"), row=current_row, col=1)
                    fig.add_hline(y=70, line_dash="dot", line_color="red", row=current_row, col=1)
                    fig.add_hline(y=30, line_dash="dot", line_color="green", row=current_row, col=1)
                    current_row += 1

                if show_macd:
                    diff = df_plot['MACD'] - df_plot['Signal_Line']
                    fig.add_trace(go.Bar(x=df_plot.index, y=diff, marker_color=['#00FF88' if x>=0 else '#FF3D00' for x in diff], name="Momentum"), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MACD'], line=dict(color='#2979FF', width=1), name="MACD"), row=current_row, col=1)
                    fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Signal_Line'], line=dict(color='#FF9100', width=1), name="Sinyal"), row=current_row, col=1)

                # Grafik Ayarları
                fig.update_layout(template="plotly_dark", height=900, xaxis_rangeslider_visible=False, margin=dict(l=0, r=10, t=10, b=0))
                fig.update_yaxes(side="right", showgrid=False)
                st.plotly_chart(fig, use_container_width=True)

    with sub_tab_strateji:
        st.header(f"🏛️ {st.session_state.current_symbol} - Strateji Laboratuvarı")
        st.markdown("---")
        
        konsensus = coklu_strateji_analizi(data)
        inds = konsensus['indicators']
        plot_data = data.tail(200)

        col_chart, col_side = st.columns([4, 1])

        with col_side:
            st.markdown("### 🛠️ Görsel Katmanlar")
            g_ema = st.checkbox("EMA Cross (9/21)", value=True, key="strat_ema")
            g_donchian = st.checkbox("Turtle (Donchian)", value=False, key="strat_donchian")
            g_vwap = st.checkbox("VWAP Hattı", value=True, key="strat_vwap")
            g_bb = st.checkbox("Bollinger / Volatilite", value=False, key="strat_bb")
            g_zscore = st.checkbox("Z-Score Sinyalleri", value=True, key="strat_zscore")
            
            st.divider()
            
            st.markdown("### 📊 Strateji Skorları")
            for s_name, s_val in konsensus['detay'].items():
                clr = "#00FF88" if s_val == "BUY" else "#FF3D00" if s_val == "SELL" else "#9E9E9E"
                icon = "🟢" if s_val == "BUY" else "🔴" if s_val == "SELL" else "⚪"
                st.markdown(f"{icon} **{s_name}:** <span style='color:{clr}; font-weight:bold;'>{s_val}</span>", unsafe_allow_html=True)
            
            st.divider()
            if st.button("🔄 Terminali Yenile", use_container_width=True, key="strat_refresh"):
                st.rerun()

        with col_chart:
            f_sig = konsensus['final_signal']
            f_clr = "#00FF88" if "AL" in f_sig else "#FF3D00" if "SAT" in f_sig else "#9E9E9E"
            st.markdown(f"## Konsensüs Sinyali: <span style='color:{f_clr}'>{f_sig}</span>", unsafe_allow_html=True)

            fig_strat = go.Figure()

            fig_strat.add_trace(go.Candlestick(x=plot_data.index, open=plot_data['Open'], high=plot_data['High'], low=plot_data['Low'], close=plot_data['Close'], name="Fiyat", increasing_line_color='#00FF88', decreasing_line_color='#FF3D00'))

            hibrit_sinyaller = hibrit_sinyal_motoru(plot_data)
        
            for sig in hibrit_sinyaller:
                color = "#00FF88" if sig['type'] == 'BUY' else "#FF3D00"
                ay_pos = -40 if sig['type'] == 'BUY' else 40
                fig_strat.add_annotation(x=sig['date'], y=sig['price'], text=sig['label'], showarrow=True, arrowhead=2, arrowcolor=color, ax=0, ay=ay_pos, font=dict(color=color, size=10), bgcolor="rgba(0,0,0,0.8)")

            if g_ema:
                fig_strat.add_trace(go.Scatter(x=plot_data.index, y=inds['ema9'].tail(200), line=dict(color='#00FF88', width=1.2), name="EMA 9"))
                fig_strat.add_trace(go.Scatter(x=plot_data.index, y=inds['ema21'].tail(200), line=dict(color='#FF3D00', width=1.2), name="EMA 21"))
            if g_vwap:
                fig_strat.add_trace(go.Scatter(x=plot_data.index, y=inds['vwap'].tail(200), line=dict(color='#00B0FF', width=2), name="VWAP"))
            if g_donchian:
                fig_strat.add_trace(go.Scatter(x=plot_data.index, y=inds['u_donchian'].tail(200), line=dict(color='rgba(255,255,255,0.3)', dash='dot'), name="Turtle Üst"))
                fig_strat.add_trace(go.Scatter(x=plot_data.index, y=inds['l_donchian'].tail(200), line=dict(color='rgba(255,255,255,0.3)', dash='dot'), name="Turtle Alt"))
            if g_bb:
                fig_strat.add_trace(go.Scatter(x=plot_data.index, y=inds['u_bb'].tail(200), line=dict(color='rgba(179, 136, 255, 0.4)'), name="BB Üst"))
                fig_strat.add_trace(go.Scatter(x=plot_data.index, y=inds['l_bb'].tail(200), line=dict(color='rgba(179, 136, 255, 0.4)'), fill='tonexty', name="BB Alt"))
            if g_zscore:
                z_score = (plot_data['Close'] - plot_data['Close'].rolling(20).mean()) / plot_data['Close'].rolling(20).std()
                oversold, overbought = z_score < -2.2, z_score > 2.2
                if any(oversold):
                    fig_strat.add_trace(go.Scatter(x=plot_data.index[oversold], y=plot_data['Low'][oversold] * 0.98, mode='markers', marker=dict(symbol='diamond', size=10, color='#00FF88'), name="Z-Score Dip"))
                if any(overbought):
                    fig_strat.add_trace(go.Scatter(x=plot_data.index[overbought], y=plot_data['High'][overbought] * 1.02, mode='markers', marker=dict(symbol='diamond', size=10, color='#FF3D00'), name="Z-Score Tepe"))

            fig_strat.update_layout(template="plotly_dark", height=850, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(side="right", showgrid=False), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_strat, use_container_width=True)


if selected_tab == "🎯 Yatırım Radarı":
    st.header("🎯 Profesyonel Yatırım Radarı")
    st.markdown("---")
    
    if "radar_cache" not in st.session_state:
        st.session_state.radar_cache = None

    mod = st.radio("Görünüm Filtresi:", ["🚀 Yükseliş Fırsatları", "🚨 Düşüş Riskleri", "✅ Hedef / Nötr"], horizontal=True)

    if st.button("🔥 Tüm Piyasaları Sniper Modunda Tara"):
        with st.spinner("Piyasalar derinlemesine analiz ediliyor..."):
            ui_names = get_ui_names()
            piyasalar = {
                "🇹🇷 BIST 100": get_symbol_lists("BIST 100"),
                "₿ Kripto": get_symbol_lists("Kripto Paralar"),
                "🏗️ Emtia": get_symbol_lists("Emtialar (Maden/Enerji)")
            }
            taramalar = {p_adi: piyasa_radari_tara(s_list, ui_names) for p_adi, s_list in piyasalar.items()}
            st.session_state.radar_cache = taramalar
            st.rerun()

    if st.session_state.radar_cache is not None:
        for p_adi, veriler in st.session_state.radar_cache.items():
            if "Yükseliş" in mod:
                onayli, v_color = [v for v in veriler if ("AL" in v['durum'] or "BUY" in v['durum']) and not v.get('hedefe_vardi', False)], "#00FF88"
            elif "Düşüş" in mod:
                onayli, v_color = [v for v in veriler if ("SAT" in v['durum'] or "SELL" in v['durum']) and not v.get('hedefe_vardi', False)], "#FF3D00"
            else: 
                onayli, v_color = [v for v in veriler if v.get('hedefe_vardi', False) or "NÖTR" in v['durum'] or v['skor'] == 0], "#FFD600"

            if onayli:
                st.subheader(f"{p_adi} ({len(onayli)})")
                for i in range(0, len(onayli), 2):
                    cols = st.columns(2)
                    for idx, item in enumerate(onayli[i:i+2]):
                        with cols[idx]:
                            with st.container(border=True):
                                st.markdown(f"<h3 style='color:{v_color}; margin:0;'>{item['display_name']}</h3>", unsafe_allow_html=True)
                                
                                if item.get('hedefe_vardi', False): st.info(f"🏆 {item['durum']}")
                                elif "MATCH" in item['durum']: st.warning(f"🔥 {item['durum']} (Güçlü Teyit)")
                                elif "AL" in item['durum'] or "BUY" in item['durum']: st.success(f"📈 {item['durum']} (Skor: {item['skor']}/5)")
                                else: st.error(f"📉 {item['durum']}")

                                m1, m2, m3 = st.columns(3)
                                m1.metric("Fiyat", f"{item['fiyat']:.2f}")
                                m2.metric("RSI", f"{int(item['rsi'])}")
                                
                                if item.get('hedefe_vardi', False): m3.metric("Görülen Hedef", f"{item['hedef']:.2f}")
                                elif "AL" in item['durum'] or "BUY" in item['durum']: m3.metric("Potansiyel", f"%{item['kazanc_beklentisi']:.1f}")
                                else: m3.metric("Hedef", f"{item['hedef']:.1f}")

                                st.markdown(f"**💡 Analiz:** {item['notlar']}")
                                
                                if st.button(f"🔍 {item['display_name']} Analizine Git", key=f"radar_btn_{item['symbol']}"):
                                    st.session_state.selected_symbol_radar = item['symbol']
                                    # Piyasa türünü de kaydet
                                    market_map = {"🇹🇷 BIST 100": "BIST 100", "₿ Kripto": "Kripto Paralar", "🏗️ Emtia": "Emtialar (Maden/Enerji)"}
                                    st.session_state.selected_market_radar = market_map.get(p_adi, p_adi)
                                    # Buton artık ana "Analiz Paneli" sekmesine yönlendiriyor.
                                    st.session_state.next_tab = "📈 Analiz Paneli"
                                    st.rerun()
            else:
                st.caption(f"🔍 {p_adi} kategorisinde bu filtreye uygun sonuç yok.")
    else:
        st.info("👆 Piyasaları taramak için yukarıdaki butona basın.")