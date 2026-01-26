import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import ollama
import PyPDF2
import io
import interface
import importlib
import utils

importlib.reload(utils)

from utils import kapsamli_teknik_analiz, dinamik_trend_analizi
from utils import apply_technical_indicators, create_strategy_plot, pdf_metin_cikar, ask_ai_about_pdf, ai_yorum_yap, ai_sohbet_yaniti_uret, ai_genel_degerlendirme

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
            "SNGYO.IS", "TSPOR.IS", "ULUFA.IS", "VAKKO.IS", "YATAS.IS"
        ]
    elif market_type == "Kripto Paralar":
        return [
            "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD",
            "LTC-USD", "BCH-USD", "LINK-USD", "ICP-USD", "ARB-USD", "XLM-USD", "HBAR-USD", "FIL-USD", "VET-USD", "INJ-USD",
            "APT-USD", "PEPE-USD", "RNDR-USD", "QNT-USD", "ALGO-USD", "IMX-USD", "AAVE-USD", "GRT-USD", "MKR-USD", "EGLD-USD",
            "FTM-USD", "THETA-USD", "SAND-USD", "AXS-USD", "NEAR-USD", "CHZ-USD", "LDO-USD"
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
        "LDO-USD": "Lido DAO",

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
        "LOGO.IS": "Logo Yazılım", "SNGYO.IS": "Sinpaş GYO", "VAKKO.IS": "Vakko", "YATAS.IS": "Yataş"
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
    "📊 Tahminleme", 
    "🧠 Stratejiler", 
    "📄 Şirket Analizi", 
    "🤖 AI Finansal Danışman", 
    "🔍 Teknik Analiz" 
])
with tabs[3]:
    st.header("🤖 AI Finansal Danışman")
    
    st.subheader("Bütüncül Analiz Raporu")
    st.info("Tahminleme ve strateji analizlerini birleştirerek kapsamlı bir genel değerlendirme alın.")

    if st.button("Analizleri Birleştir ve Yorumla"):
        tahmin_yorumu = st.session_state.get('tahmin_yorumu', None)
        strateji_yorumu = st.session_state.get('strateji_yorumu', None)

        if not tahmin_yorumu and not strateji_yorumu:
            st.warning("Yorum yapılacak herhangi bir analiz bulunamadı. Lütfen önce 'Tahminleme' veya 'Stratejiler' sekmelerinden analizleri çalıştırın.")
        else:
            with st.spinner("AI, tüm analizleri birleştirerek genel bir değerlendirme hazırlıyor..."):
                genel_degerlendirme, hata = ai_genel_degerlendirme(tahmin_yorumu, strateji_yorumu, selected_symbol)
                if hata:
                    st.error(hata)
                else:
                    st.markdown("---")
                    st.header(f"Genel Değerlendirme: {selected_symbol}")
                    st.success(genel_degerlendirme)
    
    st.divider()

    st.header("Serbest Sohbet")
    st.info("Finansal piyasalar, şirket analizleri veya yatırım stratejileri hakkında sorularınızı sorun.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Mesajınızı buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Düşünüyor..."):
                cevap, hata = ai_sohbet_yaniti_uret(st.session_state.messages)
                if hata:
                    st.error(hata)
                else:
                    st.markdown(cevap)
                    st.session_state.messages.append({"role": "assistant", "content": cevap})

with tabs[2]:
    st.header("📄 Şirket Raporu Analizi")
    
    if 'pdf_icerik' not in st.session_state:
        st.session_state.pdf_icerik = None
    if 'ai_cevap' not in st.session_state:
        st.session_state.ai_cevap = None

    pdf_dosya = st.file_uploader("Bilançoyu veya finansal raporu buraya sürükleyin (PDF)", type=["pdf"])
    
    if pdf_dosya:
        with st.spinner("PDF içeriği okunuyor..."):
            st.session_state.pdf_icerik = pdf_metin_cikar(pdf_dosya)
            st.success("✅ PDF başarıyla okundu ve içeriği hafızaya alındı!")
    
    if st.session_state.pdf_icerik:
        st.info("PDF içeriği hafızada. Şimdi bu içerik hakkında sorular sorabilirsiniz.")
        soru = st.text_input("🧐 Rapor hakkında bir soru sorun:")
        
        if soru:
            with st.spinner("AI analiz ediyor..."):
                cevap, hata = ask_ai_about_pdf(st.session_state.pdf_icerik, soru)
                if hata:
                    st.error(f"AI Hatası: {hata}")
                else:
                    st.session_state.ai_cevap = cevap
    
    if st.session_state.ai_cevap:
        st.markdown("---")
        st.markdown(f"**🤖 AI Cevabı:**\n{st.session_state.ai_cevap}")



with tabs[0]:
    with st.sidebar:
        st.header("🔮 Tahminleme Ayarları")
        model_type = st.selectbox("Tahmin Modeli Seçiniz:", 
                                  ["ARIMA", "ETS", "Holt-Winters", "XGBoost", "LSTM", "RandomForest-XGBoost Hybrid", "HMM Trend Regime"])
        forecast_days = st.slider("Tahmin Edilecek Gün Sayısı:", 5, 60, 15)

        arima_params = {}
        if model_type == "ARIMA":
            arima_params['p'] = st.number_input("AR (p) Değeri:", min_value=0, value=1)
            arima_params['d'] = st.number_input("Fark Düzeyi (d):", min_value=0, value=1)
            arima_params['q'] = st.number_input("MA (q) Değeri:", min_value=0, value=1)

        st.subheader("Özellik Seçimi")
        col1, col2, col3 = st.columns(3)
        indicator_config = {
            "use_rsi": col3.checkbox("📈 RSI"),
            "use_volume": col3.checkbox("📊 Hacim"),
            "use_macd": col2.checkbox("💹 MACD"),
            "use_volatility": col2.checkbox("🌊 Volatilite"),
            "use_momentum": col1.checkbox("⚡ Momentum"),
            "use_stochastic": col1.checkbox("🎯 Stochastic"),
            "use_williams": col1.checkbox("📉 Williams %R")
        }

    if st.button("📊 Tahminle"):
        try:
            with st.spinner(f"{selected_symbol} için geçmiş veriler çekiliyor..."):
                stock_data = load_data(selected_symbol)
            
            if stock_data is None:
                st.error(f"⚠️ **Veri Çekme Hatası:** '{selected_symbol}' için veri alınamadı.")
                st.stop()

            st.success(f"✅ '{selected_symbol}' verisi başarıyla çekildi.")
            ts_data = stock_data['Close'].dropna()
            features = utils.calculate_technical_features(ts_data, stock_data, [7, 14, 30], indicator_config)

            forecast, error_msg = None, None
            with st.spinner(f"{model_type} modeli eğitiliyor..."):
                            forecast, error_msg = None, None
                            
                            # 1. ARIMA Modeli
                            if model_type == "ARIMA":
                                forecast, error_msg = utils.train_arima_model(ts_data, **arima_params, forecast_days=forecast_days)
                            
                            # 2. ETS (Exponential Smoothing) Modeli
                            elif model_type == "ETS":
                                forecast, error_msg = utils.train_ets_model(ts_data, trend_type='add', forecast_days=forecast_days)
                            
                            # 3. Holt-Winters Modeli
                            elif model_type == "Holt-Winters":
                                # sp=5 iş günlerini temsil eder
                                forecast, error_msg = utils.train_holt_winters_model(ts_data, trend='add', seasonal='add', sp=5, forecast_days=forecast_days)
                            
                            # 4. XGBoost Modeli
                            elif model_type == "XGBoost":
                                forecast, error_msg = utils.train_xgboost_model(ts_data, features, forecast_days)
                            
                            # 5. LSTM Modeli
                            elif model_type == "LSTM":
                                forecast, error_msg = utils.train_lstm_model(ts_data, forecast_days)
                            
                            # 6. Random Forest - XGBoost Hybrid Modeli
                            elif model_type == "RandomForest-XGBoost Hybrid":
                                forecast, error_msg = utils.train_hybrid_rf_xgb_model(ts_data, features, forecast_days)
                            
                            # 7. HMM (Hidden Markov Model) Trend Regime Modeli
                            elif model_type == "HMM Trend Regime":
                                forecast, error_msg = utils.train_hmm_model(ts_data, n_components=2, forecast_days=forecast_days)
                        
            if error_msg:
                st.error(f"Hata: {error_msg}")
            elif forecast is not None:
                # Sonuçları session_state'e kaydet
                st.session_state.tahmin_sonucu = {
                    "ts_data": ts_data,
                    "forecast": forecast,
                    "forecast_days": forecast_days,
                    "symbol": selected_symbol
                }
                
                # AI Yorumunu al ve kaydet
                ai_comment, ai_error = utils.get_ai_forecast_analysis(selected_symbol, model_type, ts_data, forecast)
                if ai_error:
                    st.session_state.tahmin_yorumu = f"AI Analizi başarısız: {ai_error}"
                else:
                    st.session_state.tahmin_yorumu = ai_comment

        except Exception as e:
            st.error(f"⚠️ Genel bir hata oluştu: {e}")

    # --- Kayıtlı Tahmin Sonuçlarını Göster ---
    if st.session_state.tahmin_sonucu:
        st.divider()
        st.header(f"📈 {st.session_state.tahmin_sonucu['symbol']} İçin Tahmin Sonuçları")
        utils.display_forecast_results(
            st.session_state.tahmin_sonucu["ts_data"],
            st.session_state.tahmin_sonucu["forecast"],
            st.session_state.tahmin_sonucu["forecast_days"],
            st.session_state.tahmin_sonucu["symbol"]
        )
    
    if st.session_state.tahmin_yorumu:
        st.divider()
        st.subheader("🤖 Yapay Zeka Tahmin Yorumu")
        st.info(st.session_state.tahmin_yorumu)

# tabs[1] içeriği
with tabs[1]:
    with st.sidebar:
        st.header("🧠 Strateji Ayarları")
        strategies = st.multiselect("Strateji Seçimi:", [
            "Turtle Trade", "Moving Average Crossover", "Donchian Channel Breakout", 
            "Bollinger Bands Breakout", "Parabolic SAR", "MACD Trend Tracking"
        ], key="strateji_secimi")

    if st.button("📈 Stratejileri Uygula"):
        if not strategies:
            st.warning("Lütfen en az bir strateji seçiniz.")
        else:
            try:
                with st.spinner(f"{selected_symbol} için strateji verileri hesaplanıyor..."):
                    # 1. Veri Çekme
                    stock_data = load_data(selected_symbol)
                    if stock_data is None:
                        st.error(f"'{selected_symbol}' için veri çekilemedi.")
                        st.stop()

                    # 2. Teknik Göstergeleri Hesapla
                    stock_data_with_indicators = apply_technical_indicators(stock_data.copy())
                    
                    # 3. Grafiği Oluştur
                    fig = create_strategy_plot(stock_data_with_indicators, strategies, selected_symbol)
                    
                    # 4. Sonuçları ve Yorumu Session State'e Kaydet
                    st.session_state.strateji_grafigi = fig
                    
                    ticker = yf.Ticker(selected_symbol)
                    summary = ticker.info.get('longBusinessSummary', 'Bilgi yok.')
                    son_fiyat = stock_data['Close'].iloc[-1]
                    strateji_metni = f"Uygulanan Stratejiler: {', '.join(strategies)}"
                    analiz_sonucu, hata = ai_yorum_yap(selected_symbol, summary, strateji_metni, son_fiyat)
                    
                    if hata:
                        st.session_state.strateji_yorumu = f"AI yorumu alınamadı: {hata}"
                    else:
                        st.session_state.strateji_yorumu = analiz_sonucu

            except Exception as e:
                st.error(f"Strateji analizi sırasında bir hata oluştu: {e}")

    # --- Kayıtlı Strateji Sonuçlarını Göster ---
    if st.session_state.strateji_grafigi:
        st.header(f"🧠 {st.session_state.secilen_sembol} İçin Strateji Sinyalleri")
        st.plotly_chart(st.session_state.strateji_grafigi, use_container_width=True)
    
    if st.session_state.strateji_yorumu:
        st.divider()
        st.subheader("🤖 AI Strateji Analizi")
        st.info(st.session_state.strateji_yorumu)

import importlib
import utils
importlib.reload(utils) 

with tabs[4]:
    st.header(f"🔍 {selected_symbol} - Profesyonel Strateji Paneli")
    
    if st.button("Analizi Yenile ve Görünümü Optimize Et"):
        with st.spinner("Grafik katmanları ve göstergeler yerleştiriliyor..."):
            data = yf.download(selected_symbol, period="2y", interval="1d")
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if not data.empty:
                # Hesaplamalar
                analiz = utils.kapsamli_teknik_analiz(data)
                aktif_trendler = utils.dinamik_trend_analizi(data)
                gecmis_trendler = utils.tarihsel_trend_analizi(data)
                fibo_levels = utils.calculate_fibonacci_levels(data) 

                # --- ÜST PANEL: SİNYAL VE METRİKLER ---
                sign_map = {
                    "GÜÇLÜ AL": {"color": "#00FF88", "icon": "🚀", "bg": "rgba(0, 255, 136, 0.1)"},
                    "AL": {"color": "#00E5FF", "icon": "📈", "bg": "rgba(0, 229, 255, 0.1)"},
                    "TEPKİ ALIMI (RİSKLİ)": {"color": "#FFD600", "icon": "⚡", "bg": "rgba(255, 214, 0, 0.1)"},
                    "NÖTR": {"color": "#B0BEC5", "icon": "⚖️", "bg": "rgba(176, 190, 197, 0.1)"},
                    "SAT": {"color": "#FF9100", "icon": "📉", "bg": "rgba(255, 145, 0, 0.1)"},
                    "GÜÇLÜ SAT": {"color": "#FF3D00", "icon": "⚠️", "bg": "rgba(255, 61, 0, 0.1)"}
                }
                s_info = sign_map.get(analiz['durum'], sign_map["NÖTR"])
                
                # Sinyal Kartı
                st.markdown(f"""
                    <div style="background-color:{s_info['bg']}; padding:15px; border-radius:10px; border-left: 5px solid {s_info['color']}; margin-bottom:20px;">
                        <span style="color:{s_info['color']}; font-size:24px; font-weight:bold;">{s_info['icon']} {analiz['durum']}</span>
                        <span style="color:white; margin-left:20px;">Teknik Skor: <b>{analiz['skor']} / 5</b> | RSI: <b>{analiz['rsi']:.1f}</b></span>
                    </div>
                """, unsafe_allow_html=True)

                # Metrik Satırı
                m = st.columns(4)
                m[0].metric("Anlık Fiyat", f"{analiz['fiyat']:.2f}")
                m[1].metric("🎯 Hedef", f"{analiz['hedef']:.2f}", delta=f"{((analiz['hedef']/analiz['fiyat'])-1)*100:.1f}%")
                m[2].metric("🛑 Stop", f"{analiz['stop']:.2f}", delta=f"{((analiz['stop']/analiz['fiyat'])-1)*100:.1f}%", delta_color="inverse")
                m[3].metric("R/R Oranı", f"{analiz['rr_oran']:.2f}")

                # --- GRAFİK TASARIMI ---
                plot_data = data.tail(720)
                fig = go.Figure()

                fig.add_trace(go.Candlestick(
                    x=plot_data.index, open=plot_data['Open'], high=plot_data['High'],
                    low=plot_data['Low'], close=plot_data['Close'], name="Fiyat",
                    increasing_line_color="#13934C", decreasing_line_color="#C6431B"
                ))

                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Close'].rolling(50).mean(), 
                                         line=dict(color='#FFD600', width=1), name="SMA 50", opacity=0.5))
                fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Close'].rolling(200).mean(), 
                                         line=dict(color='#E53935', width=1.5), name="SMA 200", opacity=0.8))

                for line in gecmis_trendler:
                    fig.add_trace(go.Scatter(x=line['x'], y=line['y'], mode='lines', line=dict(color=line['color'], width=1), opacity=0.3, showlegend=False))
                for line in aktif_trendler:
                    fig.add_trace(go.Scatter(x=line['x'], y=line['y'], mode='lines', line=dict(color=line['color'], width=3.5), name="Aktif Trend"))

                fig.add_hrect(y0=analiz['fiyat'], y1=analiz['hedef'], fillcolor="rgba(0, 255, 136, 0.05)", line_width=0, name="Kâr Bölgesi")
                fig.add_hrect(y0=analiz['stop'], y1=analiz['fiyat'], fillcolor="rgba(255, 61, 0, 0.05)", line_width=0, name="Zarar Bölgesi")

                for lvl, val in fibo_levels.items():
                    fig.add_hline(y=val, line_width=0.5, line_dash="dot", line_color="rgba(255,255,255,0.2)",
                                  annotation_text=f"Fibo {lvl}", annotation_position="right")

                fig.add_hline(y=analiz['hedef'], line_width=2, line_color="#0B822B", line_dash="dash", annotation_text="🎯 HEDEF", annotation_position="right")
                fig.add_hline(y=analiz['stop'], line_width=2, line_color="#551D0C", line_dash="dash", annotation_text="🛑 STOP", annotation_position="right")

                fig.update_layout(
                    template="plotly_dark",
                    height=800,
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=0, r=50, t=30, b=0),
                    yaxis=dict(side="right", gridcolor="rgba(255,255,255,0.05)", tickfont=dict(color="#B0BEC5")),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )

                st.plotly_chart(fig, use_container_width=True)