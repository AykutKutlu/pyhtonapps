from core.utils import (
    kapsamli_teknik_analiz,
    piyasa_radari_tara,
    calculate_fibonacci_levels,
    dinamik_trend_analizi,
    tarihsel_seviye_analizi
)
from core import interface
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

from datetime import datetime
import numpy as np

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

with st.sidebar:
    market_type = st.selectbox("📊 Piyasa Seçiniz", ["BIST 100", "Kripto Paralar", "Emtialar (Maden/Enerji)"])
    
    symbols = get_symbol_lists(market_type)
    ui_names = get_ui_names()
    
    # format_func sayesinde kullanıcı UI ismini görür ama kod arka planda sembolü (GC=F) tutar
    selected_symbol = st.selectbox(
        "📌 Sembol Seçiniz", 
        symbols, 
        format_func=lambda x: ui_names.get(x, x)
    )

    st.divider()

if st.session_state.secilen_sembol != selected_symbol:
    st.session_state.tahmin_sonucu = None
    st.session_state.tahmin_yorumu = None
    st.session_state.strateji_grafigi = None
    st.session_state.strateji_yorumu = None
    st.session_state.secilen_sembol = selected_symbol

tabs = st.tabs([
    "📈 Stratejik Teknik", 
    "🎯 Yatırım Radarı"
])





with tabs[0]:
    st.header(f"🔍 {selected_symbol} - Profesyonel Strateji Paneli")

    # --- VERİ YÖNETİMİ ---
    if "chart_data" not in st.session_state or st.session_state.get("last_symbol") != selected_symbol:
        raw_data = yf.download(selected_symbol, period="2y", interval="1d")
        if isinstance(raw_data.columns, pd.MultiIndex):
            raw_data.columns = raw_data.columns.get_level_values(0)
        st.session_state["chart_data"] = raw_data
        st.session_state["last_symbol"] = selected_symbol

    data = st.session_state["chart_data"]

    if not data.empty:
        # Teknik katmanları hesapla
        analiz = kapsamli_teknik_analiz(data)
        aktif_trendler = dinamik_trend_analizi(data)
        fibo_levels = calculate_fibonacci_levels(data) 
        seviyeler = tarihsel_seviye_analizi(data)

        # --- DÜZEN: SOL GRAFİK (%85), SAĞ FİLTRE (%15) ---
        col_chart, col_filter = st.columns([5, 1])

        with col_filter:
            st.markdown("### 🛠️ Katmanlar")
            f_sig = st.checkbox("Sinyal Okları (AL/SAT)", value=True)
            f_levels = st.checkbox("🎯 Hedef & 🛑 Stop", value=True) # YENİ FİLTRE
            f_trend = st.checkbox("Trend Hatları", value=True)
            f_seviye = st.checkbox("Destek/Direnç", value=True)
            f_fibo = st.checkbox("Fibonacci", value=False)
            f_sma50 = st.checkbox("SMA 50", value=True)
            f_sma200 = st.checkbox("SMA 200", value=True)
            st.divider()
            if st.button("🔄 Veriyi Güncelle"):
                st.session_state.pop("chart_data")
                st.rerun()

        with col_chart:
            # --- DURUM VE YÖN BİLGİSİ ---
            yon_rengi = "#00FF88" if analiz['signal_type'] == "BUY" else "#FF3D00"
            st.markdown(f"### Mevcut Durum: <span style='color:{yon_rengi};'>{analiz['durum']}</span>", unsafe_allow_html=True)

            # --- ÜST METRİKLER ---
            m = st.columns(5)
            m[0].metric("Anlık Fiyat", f"{analiz['fiyat']:.2f}")
            # Hedef ve Stop renklerini yöne göre dinamik yaptık
            m[1].metric("🎯 Hedef", f"{analiz['hedef']:.2f}", f"%{analiz['kazanc_beklentisi']:.1f}")
            m[2].metric("🛑 Stop", f"{analiz['stop']:.2f}", delta_color="inverse")
            m[3].metric("📊 Skor", f"{analiz['skor']}/5")
            m[4].metric("🔔 Sinyal Fiyatı", f"{analiz['signal_price']:.2f}")

            # --- GRAFİK ÇİZİMİ ---
            plot_data = data.tail(720)
            fig = go.Figure()

            # Candlestick
            fig.add_trace(go.Candlestick(
                x=plot_data.index, open=plot_data['Open'], high=plot_data['High'],
                low=plot_data['Low'], close=plot_data['Close'], name="Fiyat",
                increasing_line_color='#00FF88', decreasing_line_color='#FF3D00'
            ))

            # SMA Katmanları
            if f_sma50:
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Close'].rolling(50).mean(), line=dict(color='#FFD600', width=1.2), name="SMA 50"))
            if f_sma200:
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Close'].rolling(200).mean(), line=dict(color='#E53935', width=1.8), name="SMA 200"))

            # SİNYAL OKLARI
            if f_sig:
                for sig in analiz['all_signals']:
                    is_latest = (sig['date'] == analiz['signal_date'])
                    if sig['type'] == "BUY":
                        color, symbol, y_val, shift = ("#00FF88", "▲", sig['low'], -20) if is_latest else ("rgba(0, 255, 136, 0.3)", "▲", sig['low'], -15)
                    else:
                        color, symbol, y_val, shift = ("#FF3D00", "▼", sig['high'], 20) if is_latest else ("rgba(255, 61, 0, 0.3)", "▼", sig['high'], 15)

                    fig.add_annotation(
                        x=sig['date'], y=y_val, text=symbol, showarrow=False,
                        yshift=shift, font=dict(color=color, size=12 if is_latest else 10)
                    )
            
            # HEDEF VE STOP ÇİZGİLERİ (Sadece en son sinyale göre ve filtre aktifse)
            if f_levels and analiz['signal_type'] != "NEUTRAL":
                # Çizgi Renkleri
                target_color = "#00FF88" if analiz['signal_type'] == "BUY" else "#FF3D00"
                stop_color = "#FF3D00" if analiz['signal_type'] == "BUY" else "#00FF88"
                
                # Hedef Çizgisi
                fig.add_shape(type="line", x0=analiz['signal_date'], y0=analiz['hedef'], x1=data.index[-1], y1=analiz['hedef'], 
                             line=dict(color=target_color, width=3, dash="dash"))
                fig.add_annotation(x=data.index[-1], y=analiz['hedef'], text="🎯 HEDEF", showarrow=False, xanchor="left", font=dict(color=target_color, size=10))

                # Stop Çizgisi
                fig.add_shape(type="line", x0=analiz['signal_date'], y0=analiz['stop'], x1=data.index[-1], y1=analiz['stop'], 
                             line=dict(color=stop_color, width=3, dash="dashdot"))
                fig.add_annotation(x=data.index[-1], y=analiz['stop'], text="🛑 STOP-LOSS", showarrow=False, xanchor="left", font=dict(color=stop_color, size=10))

                # Giriş (Sinyal) Seviyesi
                fig.add_shape(type="line", x0=analiz['signal_date'], y0=analiz['signal_price'], x1=data.index[-1], y1=analiz['signal_price'], 
                             line=dict(color="gray", width=1, dash="dot"))

            # Diğer Katmanlar...
            if f_trend:
                for line in aktif_trendler:
                    fig.add_trace(go.Scatter(x=line['x'], y=line['y'], mode='lines', line=dict(color=line['color'], width=4), name="Trend"))
            if f_seviye:
                for lvl in seviyeler:
                    fig.add_shape(type="line", x0=lvl['date'], y0=lvl['val'], x1=data.index[-1], y1=lvl['val'], line=dict(color=lvl['color'], width=1, dash="dashdot"))
            if f_fibo:
                for lvl, val in fibo_levels.items():
                    fig.add_hline(y=val, line_width=0.8, line_dash="dot", line_color="rgba(255,255,255,0.2)", annotation_text=f"Fibo {lvl}")

            fig.update_layout(template="plotly_dark", height=800, xaxis_rangeslider_visible=False, margin=dict(l=0, r=100, t=10, b=0), yaxis=dict(side="right"))
            st.plotly_chart(fig, use_container_width=True)
with tabs[1]:
    st.header("🎯 Profesyonel Yatırım Radarı")
    st.markdown("---")
    
    # Kullanıcıya neyi taramak istediğini soralım (Opsiyonel ama şık durur)
    islem_tipi = st.radio("Tarama Modu:", ["Yükseliş Fırsatları (AL)", "Düşüş Riskleri (SAT)"], horizontal=True)

    if st.button("🔥 Tüm Piyasaları Derinlemesine Tara"):
        ui_names = get_ui_names()
        piyasalar = {
            "🇹🇷 BIST 100": get_symbol_lists("BIST 100"),
            "₿ Kripto": get_symbol_lists("Kripto Paralar"),
            "🏗️ Emtia": get_symbol_lists("Emtialar (Maden/Enerji)")
        }
        
        for p_adi, s_list in piyasalar.items():
            st.subheader(p_adi)
            with st.spinner(f"{p_adi} taranıyor..."):
                # Radarı çalıştırıyoruz
                sonuclar = piyasa_radari_tara(s_list, ui_names)
                
                # Seçilen moda göre filtrele
                if "AL" in islem_tipi:
                    onayli = [s for s in sonuclar if "AL" in s['durum'] and s['skor'] >= 4]
                    baslik_rengi = "#00FF88" # Yeşil
                    bg_rengi = "rgba(0, 255, 136, 0.1)"
                else:
                    onayli = [s for s in sonuclar if "SAT" in s['durum'] or s['skor'] <= 1]
                    baslik_rengi = "#FF3D00" # Kırmızı
                    bg_rengi = "rgba(255, 61, 0, 0.1)"
                
                if onayli:
                    rows = [onayli[i:i + 2] for i in range(0, len(onayli), 2)]
                    for row in rows:
                        cols = st.columns(2)
                        for idx, item in enumerate(row):
                            with cols[idx]:
                                # Dinamik Renk Belirleme
                                card_color = "#00FF88" if "AL" in item['durum'] else "#FF3D00"
                                card_bg = "rgba(0, 255, 136, 0.05)" if "AL" in item['durum'] else "rgba(255, 61, 0, 0.05)"
                                
                                with st.container(border=True):
                                    st.markdown(f"<h3 style='color:{card_color}; margin-bottom:0;'>{item['display_name']}</h3>", unsafe_allow_html=True)
                                    
                                    if "AL" in item['durum']:
                                        st.success(f"**{item['durum']}**")
                                    else:
                                        st.error(f"**{item['durum']}**")
                                    
                                    # SEVİYE TABLOSU
                                    c1, c2 = st.columns(2)
                                    # Satışta "En İyi Giriş" aslında "En İyi Satış/Short" yeridir
                                    entry_label = "En İyi Giriş" if "AL" in item['durum'] else "Direnç / Satış"
                                    target_label = "Potansiyel Hedef" if "AL" in item['durum'] else "Düşüş Hedefi"
                                    
                                    c1.metric(entry_label, f"{item['en_guclu_alis']:.2f}")
                                    c2.metric(target_label, f"{item['hedef']:.2f}")
                                    
                                    # KAZANÇ / KAYIP KUTUSU
                                    val_text = "Beklenen Kazanç" if "AL" in item['durum'] else "Beklenen Düşüş"
                                    st.markdown(f"""
                                        <div style="background-color:{card_bg}; padding:10px; border-radius:5px; text-align:center; border: 1px solid {card_color};">
                                            <span style="color:{card_color}; font-size:18px;">{val_text}: <b>%{item['kazanc_beklentisi']:.1f}</b></span>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.caption(f"💡 Analiz Notu: {item.get('notlar', 'Veri yok.')}")
                else:
                    st.info(f"{p_adi} piyasasında şu an seçilen kriterde bir durum görünmüyor.")