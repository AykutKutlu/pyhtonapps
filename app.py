import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

# Streamlit Arayüzü
st.title("BIST 100 Teknik ve Zaman Serisi Analizi")

# Kullanıcının seçim yapabileceği hisseler
stocks = ["GARAN.IS", "KCHOL.IS", "THYAO.IS", "FROTO.IS", "ISCTR.IS", "BIMAS.IS", "TUPRS.IS", "ENKAI.IS", "ASELS.IS", "AKBNK.IS"]
selected_stocks = st.multiselect("Hisse Seçimi", stocks, default=["GARAN.IS"])

show_rsi = st.checkbox("RSI Göster", value=True)
show_ma = st.checkbox("Hareketli Ortalamalar Göster", value=True)

arima_p = st.number_input("ARIMA p:", min_value=0, value=1)
arima_d = st.number_input("ARIMA d:", min_value=0, value=1)
arima_q = st.number_input("ARIMA q:", min_value=0, value=1)

if st.button("Analizi Çalıştır"):

    for stock in selected_stocks:
        st.subheader(f"{stock} Analizi")
        
        # Veri çekme
        df = yf.download(stock, start="2022-01-01", end="2024-12-31")
        df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
        df["MA50"] = df["Close"].rolling(window=50).mean()
        df["MA200"] = df["Close"].rolling(window=200).mean()
        
        # Teknik Analiz Grafiği
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df["Close"], label="Kapanış Fiyatı", color="black")
        
        if show_rsi:
            ax.plot(df.index, df["RSI"], label="RSI", color="blue")
        if show_ma:
            ax.plot(df.index, df["MA50"], label="50 Günlük MA", color="red")
            ax.plot(df.index, df["MA200"], label="200 Günlük MA", color="green")
        
        ax.set_title(f"{stock} Teknik Analiz Grafiği")
        ax.legend()
        st.pyplot(fig)

        # Zaman Serisi Analizi
        ts_data = df["Close"].dropna()
        
        if len(ts_data) > 30:
            model = ARIMA(ts_data, order=(arima_p, arima_d, arima_q))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=10)

            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(ts_data.index, ts_data, label="Gerçek Veri", color="black")
            ax2.plot(pd.date_range(start=ts_data.index[-1], periods=11, freq="D")[1:], forecast, label="Tahmin", color="red")
            ax2.set_title(f"{stock} Zaman Serisi Tahmini")
            ax2.legend()
            st.pyplot(fig2)

            st.table(pd.DataFrame({"Tarih": pd.date_range(start=ts_data.index[-1], periods=11, freq="D")[1:], "Tahmin": forecast}))
        else:
            st.warning(f"{stock} için yeterli veri yok!")

        # Finansal Stratejiler Tablosu
        st.subheader("Finansal Stratejiler")
        strategy_df = pd.DataFrame({
            "Strateji": ["Kelebek", "Koruma", "Spread"],
            "Açıklama": ["Kâr potansiyelini optimize eden strateji.", "Riskten korunma stratejisi.", "Fiyat farkından yararlanma stratejisi."]
        })
        st.table(strategy_df)

        # Muhasebe Analizi
        avg_volume = df["Volume"].mean()
        high_low_diff = (df["High"] - df["Low"]).mean()
        accounting_df = pd.DataFrame({
            "Metrik": ["Ortalama Hacim", "Günlük Yüksek-Düşük Farkı"],
            "Değer": [avg_volume, high_low_diff]
        })
        st.subheader("Muhasebe Analizi")
        st.table(accounting_df)
