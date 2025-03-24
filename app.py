import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.title("📈 BIST 100 Hisse Tahminleme ve Turtle Trade Stratejisi")

page = st.sidebar.selectbox("📌 Sayfa Seçiniz", ["Tahminleme", "Turtle Trade Stratejisi"])

symbols = [
    "GARAN.IS", "KCHOL.IS", "THYAO.IS", "FROTO.IS", "ISCTR.IS", "BIMAS.IS", "TUPRS.IS", "ENKAI.IS", "ASELS.IS", "AKBNK.IS", 
    "YKBNK.IS", "VAKBN.IS", "TCELL.IS", "SAHOL.IS", "SASA.IS", "TTKOM.IS", "EREGL.IS", "CCOLA.IS", "PGSUS.IS", "SISE.IS", 
    "AEFES.IS", "HALKB.IS", "TOASO.IS", "ARCLK.IS", "TAVHL.IS", "ASTOR.IS", "MGROS.IS", "TTRAK.IS", "AGHOL.IS", "OYAKC.IS", 
    "KOZAL.IS", "ENJSA.IS", "BRSAN.IS", "TURSG.IS", "GUBRF.IS", "MPARK.IS", "OTKAR.IS", "BRYAT.IS", "ISMEN.IS", "PETKM.IS", 
    "ULKER.IS", "CLEBI.IS", "DOAS.IS", "AKSEN.IS", "ANSGR.IS", "ALARK.IS", "EKGYO.IS", "TABGD.IS", "RGYAS.IS", "DOHOL.IS", 
    "TSKB.IS", "ENERY.IS", "KONYA.IS", "EGEEN.IS", "AKSA.IS", "CIMSA.IS", "HEKTS.IS", "MAVI.IS", "VESBE.IS", "KONTR.IS", 
    "TKFEN.IS", "BTCIM.IS", "ECILC.IS", "KCAER.IS", "KRDMD.IS", "SOKM.IS", "KOZAA.IS", "SMRTG.IS", "CWENE.IS", "ZOREN.IS", 
    "EUPWR.IS", "REDR.IS", "VESTL.IS", "MIATK.IS", "ALFAS.IS", "GESAN.IS", "OBAM.IS", "AKFYE.IS", "KLSER.IS", "AGROT.IS", 
    "YEOTK.IS", "BINHO1000.IS", "KARSN.IS", "TMSN.IS", "SKBNK.IS", "FENER.IS", "CANTE.IS", "TUKAS.IS", "KTLEV.IS", "ADEL.IS", 
    "BERA.IS", "ODAS.IS", "AKFGY.IS", "GOLTS.IS", "ARDYZ.IS", "BJKAS.IS", "PEKGY.IS", "PAPIL.IS", "LMKDC.IS", "ALTNY.IS"
]
selected_symbol = st.selectbox("📊 Hisse Seçiniz:", symbols)

if page == "Tahminleme":
    model_type = st.selectbox("📡 Tahmin Modeli Seçiniz:", ["ARIMA", "ETS", "Holt-Winters", "XGBoost", "LSTM"])
    forecast_days = st.slider("📆 Tahmin Edilecek Gün Sayısı:", min_value=10, max_value=60, value=30)
    
    if model_type == "ARIMA":
                    p = st.number_input("🔢 AR (p) Değeri:", min_value=0, value=1)
                    d = st.number_input("🔢 Fark Düzeyi (d):", min_value=0, value=1)
                    q = st.number_input("🔢 MA (q) Değeri:", min_value=0, value=1)

    timeframes = [7, 14, 30]
    use_volume = st.checkbox("📊 İşlem Hacmini Kullan")
    use_volatility = st.checkbox("🌊 Volatiliteyi Kullan")
    use_technical = st.checkbox("📈 Teknik Göstergeleri Kullan (RSI, MACD)")
    
    if st.button("📊 Tahminle"):
        try:
            stock_data = yf.download(selected_symbol, start="2020-01-01", progress=False)
            if stock_data.empty:
                st.error("⚠️ Veri çekilemedi! Lütfen geçerli bir hisse senedi seçin.")
            else:
                ts_data = stock_data['Close'].dropna()
                
                features = pd.DataFrame(index=ts_data.index)
                for t in timeframes:
                    features[f'MA_{t}'] = ts_data.rolling(window=t).mean()
                if use_volume:
                    features['Volume'] = stock_data['Volume']
                if use_volatility:
                    features['Volatility'] = stock_data['Close'].pct_change().rolling(10).std()
                if use_technical:
                    features['RSI'] = 100 - (100 / (1 + ts_data.pct_change().rolling(14).mean()))
                features.dropna(inplace=True)
                
                if model_type == "ARIMA":
                    model = ARIMA(ts_data, order=(1, 1, 1)).fit()
                    forecast = model.forecast(steps=forecast_days)
                elif model_type == "ETS":
                    model = ExponentialSmoothing(ts_data, trend='add').fit()
                    forecast = model.forecast(steps=forecast_days)
                elif model_type == "Holt-Winters":
                    model = ExponentialSmoothing(ts_data, trend='add', seasonal='add', seasonal_periods=5).fit()
                    forecast = model.forecast(steps=forecast_days)
                elif model_type == "XGBoost":
                    X = features.iloc[:-forecast_days]
                    y = ts_data.iloc[:-forecast_days]
                    min_len = min(len(features), len(ts_data))
                    X = features.iloc[:min_len]
                    y = ts_data.iloc[:min_len]
                    model = XGBRegressor(objective='reg:squarederror')
                    model.fit(X, y)
                    forecast = model.predict(features.iloc[-forecast_days:])
                elif model_type == "LSTM":
                    X = features.values.reshape(-1, 1, features.shape[1])
                    y = ts_data.values
                    min_len = min(len(features), len(ts_data))
                    X = features.iloc[:min_len].values.reshape(-1, 1, features.shape[1])
                    y = ts_data.iloc[:min_len].values
                    model = Sequential([
                        LSTM(50, activation='relu', input_shape=(1, features.shape[1])),
                        Dense(1)
                    ])
                    model.compile(optimizer='adam', loss='mse')
                    model.fit(X, y, epochs=20, batch_size=16, verbose=0)
                    forecast = model.predict(features.iloc[-forecast_days:].values.reshape(-1, 1, features.shape[1]))
                    forecast = forecast.flatten()
                
                forecast_dates = pd.date_range(start=ts_data.index[-1], periods=forecast_days, freq='B')
                forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(ts_data.index[-60:], ts_data.values[-60:], label="Gerçek Veri", color='blue', linewidth=2)
                ax.plot(forecast_dates, forecast, label="Tahmin", color='red', linestyle='dashed', linewidth=2)
                ax.set_title(f"{selected_symbol} Tahmini ({model_type})", fontsize=14)
                ax.legend()
                ax.grid()
                st.pyplot(fig)
                
                st.subheader("📋 Tahmin Sonuçları")
                st.dataframe(forecast_df.head())
                csv = forecast_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Tahminleri İndir", data=csv, file_name=f"{selected_symbol}_tahminler.csv", mime='text/csv')
        except Exception as e:
            st.error(f"⚠️ Bir hata oluştu: {e}")

elif page == "Turtle Trade Stratejisi":
    symbol = st.selectbox("📊 Hisse Seçimi:", symbols, key="turtle_symbol")
    max_window = st.slider("📈 Max Noktaları İçin Gün Sayısı:", min_value=5, max_value=50, value=20)
    min_window = st.slider("📉 Min Noktaları İçin Gün Sayısı:", min_value=5, max_value=50, value=20)
    
    if st.button("📊 Stratejiyi Göster"):
        try:
            interval = "1d"
            stock_data = yf.download(symbol, period="360d", interval=interval)
            
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.droplevel(0)
            
            expected_cols = ["Open", "High", "Low", "Close", "Volume"]
            if list(stock_data.columns)[:5] == [symbol] * 5:
                stock_data.columns = expected_cols
            
            missing_cols = [col for col in expected_cols if col not in stock_data.columns]
            if missing_cols:
                st.error(f"⚠️ Eksik veri sütunları: {missing_cols}. Veri kaynağında problem olabilir.")
                st.stop()
            
            if stock_data.empty or stock_data.isnull().all().all():
                st.error("⚠️ Veri çekilemedi! Lütfen farklı bir hisse seçin veya veri kaynağını kontrol edin.")
                st.stop()
            
            st.dataframe(stock_data.head())
            
            stock_data.fillna(method="ffill", inplace=True)
            stock_data.fillna(method="bfill", inplace=True)
            
            stock_data['HighMax'] = stock_data['High'].rolling(window=max_window).max()
            stock_data['LowMin'] = stock_data['Low'].rolling(window=min_window).min()
            
            buy_signals = stock_data[stock_data['High'] >= stock_data['HighMax']]
            sell_signals = stock_data[stock_data['Low'] <= stock_data['LowMin']]
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Mum Grafiği'
            ))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['HighMax'], mode='lines', name=f'{max_window} Günlük En Yüksek'))
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['LowMin'], mode='lines', name=f'{min_window} Günlük En Düşük'))
            
            fig.add_trace(go.Scatter(
                x=buy_signals.index, 
                y=buy_signals['High'], 
                mode='markers', 
                marker=dict(color='green', size=10, symbol='triangle-up'), 
                name='📈 Alım Sinyali'
            ))
            
            fig.add_trace(go.Scatter(
                x=sell_signals.index, 
                y=sell_signals['Low'], 
                mode='markers', 
                marker=dict(color='red', size=10, symbol='triangle-down'), 
                name='📉 Satış Sinyali'
            ))
            
            fig.update_layout(title=f"{symbol} Turtle Trade Stratejisi ({'1 Day'})", xaxis_title="Tarih", yaxis_title="Fiyat")
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"⚠️ Bir hata oluştu: {e}")