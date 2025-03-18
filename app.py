import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Hisse senedi sembolleri
symbols = [
    "GARAN.IS", "KCHOL.IS", "THYAO.IS", "FROTO.IS", "ISCTR.IS", "BIMAS.IS", "TUPRS.IS", "ENKAI.IS", "ASELS.IS", "AKBNK.IS",
    "YKBNK.IS", "VAKBN.IS", "TCELL.IS", "SAHOL.IS", "SASA.IS", "TTKOM.IS", "EREGL.IS", "CCOLA.IS", "PGSUS.IS", "SISE.IS"
]

st.title("📈 BIST 100 Hisse Tahminleme")

# Kullanıcıdan seçim alma
symbol = st.selectbox("📊 Hisse Seçimi:", symbols)
p = st.number_input("🔢 AR (p) Değeri:", min_value=0, value=1)
d = st.number_input("🔢 Fark Düzeyi (d):", min_value=0, value=1)
q = st.number_input("🔢 MA (q) Değeri:", min_value=0, value=1)
model_type = st.selectbox("📡 Tahmin Modeli Seçiniz:", ["ARIMA", "ETS", "Holt-Winters"])

if st.button("📊 Tahminle"):
    try:
        # Veri çekme
        stock_data = yf.download(symbol, start="2020-01-01", progress=False)
        
        if stock_data.empty:
            st.error("⚠️ Hata: Veri çekilemedi! Lütfen geçerli bir hisse senedi seçin.")
        else:
            stock_data = stock_data['Close'].dropna()

            # Tarih aralığını belirleme
            ts_data = stock_data.asfreq('B').fillna(method='ffill')

            # Model Seçimi
            if model_type == "ARIMA":
                model = ARIMA(ts_data, order=(p, d, q)).fit()
            elif model_type == "ETS":
                model = ExponentialSmoothing(ts_data, trend='add', seasonal=None).fit()
            elif model_type == "Holt-Winters":
                model = ExponentialSmoothing(ts_data, trend='add', seasonal='add', seasonal_periods=5).fit()

            # Tahmin Yapma
            forecast = model.forecast(steps=30)
            forecast_dates = pd.date_range(start=ts_data.index[-1], periods=30, freq='B')
            forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast": forecast})

            # Grafik Çizdirme
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ts_data.index[-60:], ts_data.values[-60:], label="Gerçek Veri", color='blue', linewidth=2)
            ax.plot(forecast_df["Date"], forecast_df["Forecast"], label="Tahmin", color='red', linestyle='dashed', linewidth=2)
            ax.set_title(f"{symbol} Tahmini ({model_type})", fontsize=14)
            ax.legend()
            ax.grid()
            st.pyplot(fig)

            # Tahmin Verilerini Gösterme
            st.subheader("📋 Tahmin Sonuçları")
            st.dataframe(forecast_df.head())

            # Excel'e Aktarma Fonksiyonu
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(forecast_df)
            st.download_button(
                label="📥 Tahminleri İndir",
                data=csv,
                file_name=f"{symbol}_tahminler.csv",
                mime='text/csv',
            )

    except Exception as e:
        st.error(f"⚠️ Bir hata oluştu: {e}")
