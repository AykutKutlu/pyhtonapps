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
# Kendi fonksiyonlarınızın bulunduğu dosya
import utils 

# utils.py içindeki strateji fonksiyonlarını doğrudan kullanmak için:
from utils import apply_technical_indicators, create_strategy_plot, pdf_metin_cikar, ask_ai_about_pdf, ai_yorum_yap
def get_symbol_lists(market_type):
    """Piyasa türüne göre sembol listesini döner."""
    if market_type == "BIST 100":
        return [
            "GARAN.IS", "KCHOL.IS", "THYAO.IS", "FROTO.IS", "ISCTR.IS", "BIMAS.IS", "TUPRS.IS", 
            "ENKAI.IS", "ASELS.IS", "AKBNK.IS", "YKBNK.IS", "VAKBN.IS", "TCELL.IS", "SAHOL.IS", 
            "SASA.IS", "TTKOM.IS", "EREGL.IS", "CCOLA.IS", "PGSUS.IS", "SISE.IS", "LMKDC.IS", "ALTNY.IS"
        ]
    else:
        return [
            "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", 
            "AVAX-USD", "DOT-USD", "MATIC-USD", "LTC-USD", "LINK-USD"
        ]
    
st.set_page_config(page_title="Finansal Analiz Pro", layout="wide")

# 2. Temayı Uygula
interface.apply_custom_css()
# --- SAYFA BAŞLIĞI ---
st.title("📈 Hisse & Kripto Tahminleme ve Stratejiler")

# --- SOL TARAFTA SEÇİMLER (Sidebar) ---
with st.sidebar:
    st.header("⚙️ Ayarlar")
    
    # 1. Piyasa ve Sembol Seçimi
    market_type = st.selectbox("📊 Piyasa Seçiniz", ["BIST 100", "Kripto Paralar"])
    symbols = get_symbol_lists(market_type)
    selected_symbol = st.selectbox("📌 Sembol Seçiniz", symbols)

    st.divider()

    # 2. PDF Rapor Analiz Bölümü
    st.subheader("📄 Şirket Raporu Analizi (PDF)")
    pdf_dosya = st.file_uploader("Bilançoyu buraya sürükleyin", type=["pdf"])
    
    if pdf_dosya:
        # Metin çıkarma işlemi (utils'den veya yukarıdaki def'ten geliyor)
        with st.spinner("PDF içeriği analiz ediliyor..."):
            pdf_icerik = pdf_metin_cikar(pdf_dosya) 
            st.success("✅ PDF başarıyla okundu!")
            
            soru = st.text_input("🧐 Bilanço hakkında bir soru sorun:")
            if soru:
                with st.spinner("AI analiz ediyor..."):
                    cevap, hata = ask_ai_about_pdf(pdf_icerik, soru)
                    if hata:
                        st.error(f"AI Hatası: {hata}")
                    else:
                        st.markdown("---")
                        st.markdown(f"**🤖 AI Cevabı:**\n{cevap}")

# --- ANA GÖRÜNÜM (Tabs) ---
tabs = st.tabs(["📊 Tahminleme", "🧠 Stratejiler"])

with tabs[0]:
    with st.sidebar:
        st.header("🔮 Tahminleme Ayarları")
        model_type = st.selectbox("Tahmin Modeli Seçiniz:", 
                                  ["ARIMA", "ETS", "Holt-Winters", "XGBoost", "LSTM", "RandomForest-XGBoost Hybrid", "HMM Trend Regime"])
        forecast_days = st.slider("Tahmin Edilecek Gün Sayısı:", 5, 60, 15)

        # Model Spesifik Ayarlar
        arima_params = {}
        if model_type == "ARIMA":
            arima_params['p'] = st.number_input("AR (p) Değeri:", min_value=0, value=1)
            arima_params['d'] = st.number_input("Fark Düzeyi (d):", min_value=0, value=1)
            arima_params['q'] = st.number_input("MA (q) Değeri:", min_value=0, value=1)

        # İndikatör Seçimleri (Sözlük yapısında topluyoruz)
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
            # 1. Veri Çekme
            stock_data = yf.download(selected_symbol, start="2020-01-01", progress=False)
            if stock_data.empty:
                st.error("⚠️ Veri çekilemedi!")
                st.stop()
            
            ts_data = stock_data['Close'].dropna()

            # 2. Özellikleri Hesaplama (utils'den çağrılıyor)
            features = utils.calculate_technical_features(ts_data, stock_data, [7, 14, 30], indicator_config)

            # 3. Model Çalıştırma Mantığı
            forecast = None
            error_msg = None

            with st.spinner(f"{model_type} modeli eğitiliyor..."):
                if model_type == "ARIMA":
                    st.info("ARIMA: Geçmiş değerlere dayanarak tahmin üretiliyor.")
                    forecast, error_msg = utils.train_arima_model(ts_data, **arima_params, forecast_days=forecast_days)

                elif model_type == "XGBoost":
                    forecast, error_msg = utils.train_xgboost_model(ts_data, features, forecast_days)

                elif model_type == "LSTM":
                    forecast, error_msg = utils.train_lstm_model(ts_data, forecast_days)
                
                # ... Diğer elif blokları (ETS, HMM vb.) buraya gelecek ...

            # 4. Sonuçları Gösterme
            if error_msg:
                st.error(f"Hata: {error_msg}")
            elif forecast is not None:
                # Grafik ve CSV (utils'den çağrılıyor)
                utils.display_forecast_results(ts_data, forecast, forecast_days, selected_symbol)
                
                # AI Analizi (utils'den çağrılıyor)
                st.divider()
                st.subheader(f"🤖 Yapay Zeka {model_type} Analiz Yorumu")
                ai_comment, ai_error = utils.get_ai_forecast_analysis(selected_symbol, model_type, ts_data, forecast)
                if ai_error: st.warning(f"AI Analizi başarısız: {ai_error}")
                else: st.info(ai_comment)

        except Exception as e:
            st.error(f"⚠️ Genel bir hata oluştu: {e}")

# tabs[1] içeriği
with tabs[1]:
    with st.sidebar:
        st.header("🧠 Strateji Ayarları")
        strategies = st.multiselect("Strateji Seçimi:", [
            "Turtle Trade", "Moving Average Crossover", "Donchian Channel Breakout", 
            "Bollinger Bands Breakout", "Parabolic SAR", "MACD Trend Tracking"
        ])

    if st.button("Stratejiyi Göster"):
        if not strategies:
            st.warning("Lütfen en az bir strateji seçiniz.")
            st.stop()
        try:
            # 1. Veri Çekme ve Temizleme
            stock_data = yf.download(selected_symbol, period="720d", interval="1d")

# MultiIndex yapısını düzleştirme
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.droplevel(1) # veya 0, duruma göre
            # (MultiIndex düzeltme ve sütun kontrol kodları buraya gelecek...)
            stock_data.ffill(inplace=True) # Boşlukları ileriye dönük doldur
            stock_data.dropna(inplace=True) # Hala boş kalan satırları sil
            
            # 2. Hesaplamalar
            stock_data = apply_technical_indicators(stock_data)
            
            # 3. Görselleştirme
            fig = create_strategy_plot(stock_data, strategies, selected_symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # 4. AI Analiz Raporu
            st.divider()
            st.subheader(f"🤖 AI {selected_symbol} Strateji Analizi")
            with st.spinner("AI raporu hazırlanıyor..."):
                ticker = yf.Ticker(selected_symbol)
                summary = ticker.info.get('longBusinessSummary', 'Bilgi yok.')
                son_fiyat = stock_data['Close'].iloc[-1]
                strateji_metni = f"Kullanılan Stratejiler: {', '.join(strategies)}"
                
                # Daha önce tanımladığınız AI fonksiyonunu çağırın
                analiz_sonucu = ai_yorum_yap(selected_symbol, summary, strateji_metni, son_fiyat)
                st.info(analiz_sonucu)

        except Exception as e:
            st.error(f"Sistem Hatası: {type(e).__name__} - {str(e)}")
            print(f"Hata Detayı: {e}")