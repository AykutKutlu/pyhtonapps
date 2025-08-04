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

st.set_page_config(
    page_title="Tahminleme ve Stratejiler",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Arayüzü daha akıcı ve modern yapmak için CSS
st.markdown(
    """
    <style>
    /* Ana arkaplan ve metin renkleri */
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
    }

    /* Başlıklar */
    h1, h2, h3, h4, h5, h6 {
        color: #bb86fc;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Sidebar stilini daha anlaşılır ve modern hale getirme */
    [data-testid="stSidebar"] {
        background-color: #1e1e1e;
        color: #e0e0e0;
        min-width: 350px !important;
        max-width: 400px !important;
        width: 380px !important;
        padding: 2rem 1.5rem;
        border-right: 1px solid #333333;
        box-shadow: 2px 0 5px rgba(0,0,0,0.2);
    }

    /* Sidebar'daki tüm metinlerin rengini beyaz yapma */
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Ana içerik alanının sidebar'a göre ayarlanması */
    @media (min-width: 600px) {
        .main {
            margin-left: 400px;
        }
    }

    /* Butonlar */
    .stButton > button {
        background-color: #6200ee;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        background-color: #3700b3;
        transform: translateY(-2px);
    }

    /* Selectbox ve diğer input alanları */
    .stSelectbox > div > div, .stNumberInput > div > div, .stSlider > div > div {
        background-color: #2c2c2c;
        border-radius: 8px;
        border: 1px solid #3e3e3e;
        color: #e0e0e0;
    }
    .stSelectbox div[role="listbox"] {
        background-color: #2c2c2c;
        border: 1px solid #3e3e3e;
        color: #e0e0e0;
    }
    .stSelectbox div[role="option"] {
        color: #e0e0e0;
    }
    .stSelectbox div[role="option"]:hover {
        background-color: #3e3e3e;
    }

    /* Checkbox'lar */
    .stCheckbox > label {
        color: #e0e0e0;
        font-size: 16px;
    }
    .stCheckbox > label > div[data-testid="stDecoration"] {
        background-color: #3e3e3e;
        border-color: #bb86fc;
    }

    /* Sekmeler (Tabs) */
    .stTabs [data-testid="stTab"] {
        background-color: #2c2c2c;
        color: #b0b0b0;
        font-size: 18px;
        font-weight: 500;
        border-radius: 8px 8px 0 0;
        transition: all 0.3s ease;
    }
    .stTabs [data-testid="stTab"][aria-selected="true"] {
        background-color: #1e1e1e;
        color: #ffffff;
        border-bottom: 3px solid #6200ee;
    }
    
    /* İndirme butonu */
    .stDownloadButton > button {
        background-color: #03dac6;
        color: #000000;
    }
    .stDownloadButton > button:hover {
        background-color: #018786;
    }
    
    /* Geri bildirim kutuları */
    .stSuccess, .stError, .stWarning {
        border-radius: 8px;
    }
    
    /* Mobil uyumluluk */
    @media (max-width: 600px) {
        [data-testid="stSidebar"] {
            min-width: 100% !important;
            max-width: 100% !important;
            width: 100% !important;
            position: relative !important;
        }
        .main {
            margin-left: 0 !important;
        }
        .stButton > button {
            width: 100%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("📈 Hisse & Kripto Tahminleme ve Stratejiler")

# SOL TARAFTA SEÇİMLER (sidebar)
with st.sidebar:
    st.header("⚙️ Ayarlar")

    market_type = st.selectbox("📊 Piyasa Seçiniz", ["BIST 100", "Kripto Paralar"])

    bist_symbols = [
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

    crypto_symbols = [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "MATIC-USD",
        "LTC-USD", "BCH-USD", "LINK-USD", "ICP-USD", "ARB-USD", "XLM-USD", "HBAR-USD", "FIL-USD", "VET-USD", "INJ-USD",
        "APT-USD", "PEPE-USD", "RNDR-USD", "QNT-USD", "ALGO-USD", "IMX-USD", "AAVE-USD", "GRT-USD", "MKR-USD", "EGLD-USD",
        "FTM-USD", "THETA-USD", "SAND-USD", "AXS-USD", "XEC-USD", "KAS-USD", "XTZ-USD", "NEAR-USD", "CHZ-USD", "LDO-USD",
        "CRV-USD", "RUNE-USD", "FLOW-USD", "BSV-USD", "STX-USD", "ENS-USD", "GALA-USD", "KAVA-USD", "MINA-USD", "TWT-USD",
        "FXS-USD", "ZEC-USD", "SUSHI-USD", "ONE-USD", "CVX-USD", "OSMO-USD", "ROSE-USD", "CELO-USD", "GMT-USD", "NEXO-USD",
        "DYDX-USD", "LRC-USD", "AUDIO-USD", "COMP-USD", "YFI-USD", "BICO-USD", "JASMY-USD", "IOST-USD", "ANKR-USD", "ENS-USD",
        "BAL-USD", "API3-USD", "COTI-USD", "BAND-USD", "OCEAN-USD", "GLMR-USD", "KDA-USD", "SPELL-USD", "ELF-USD", "CTSI-USD",
        "BNT-USD", "AGIX-USD", "FET-USD", "ILV-USD", "DASH-USD", "LPT-USD", "MASK-USD", "DGB-USD", "ACA-USD", "SXP-USD",
        "REN-USD", "HNT-USD", "RSR-USD", "XNO-USD", "PERP-USD", "ALPHA-USD", "STORJ-USD", "POND-USD", "ARDR-USD", "RAD-USD",
        "MLN-USD", "CVC-USD", "TLM-USD", "TRU-USD", "STRAX-USD", "GTC-USD", "IDEX-USD", "BOND-USD", "VTHO-USD", "BTS-USD",
        "POLY-USD", "DENT-USD", "UBT-USD", "FORTH-USD", "MC-USD", "SUPER-USD", "POLS-USD", "PNT-USD", "MTL-USD", "GLCH-USD"
    ]

    symbols = bist_symbols if market_type == "BIST 100" else crypto_symbols
    selected_symbol = st.selectbox("📌 Sembol Seçiniz", symbols)

# ANA GÖRÜNÜM (tabs)
tabs = st.tabs(["📊 Tahminleme", "🧠 Stratejiler"])


with tabs[0]:
    with st.sidebar:
        st.header("🔮 Tahminleme Ayarları")
        model_type = st.selectbox("Tahmin Modeli Seçiniz:", ["ARIMA", "ETS", "Holt-Winters", "XGBoost", "LSTM", "RandomForest-XGBoost Hybrid"], key="model_type")
        forecast_days = st.slider("Tahmin Edilecek Gün Sayısı:", min_value=5, max_value=60, value=15)

        if model_type == "ARIMA":
            p = st.number_input("AR (p) Değeri:", min_value=0, value=1)
            d = st.number_input("Fark Düzeyi (d):", min_value=0, value=1)
            q = st.number_input("MA (q) Değeri:", min_value=0, value=1)

        timeframes = [7, 14, 30]

        col1, col2, col3 = st.columns(3)

        with col3:
            use_rsi = st.checkbox("📈 RSI")
            use_volume = st.checkbox("📊 İşlem Hacmini Kullan")

        with col2:
            use_macd = st.checkbox("💹 MACD")
            use_volatility = st.checkbox("🌊 Volatiliteyi Kullan")

        with col1:
            use_momentum = st.checkbox("⚡ Momentum")
            use_stochastic = st.checkbox("🎯 Stochastic Oscillator (%K)")
            use_williams = st.checkbox("📉 Williams %R")

   
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

                if use_rsi:
                    delta = ts_data.diff()
                    gain = pd.Series(np.where(delta > 0, delta, 0).squeeze(), index=ts_data.index)
                    loss = pd.Series(np.where(delta < 0, -delta, 0).squeeze(), index=ts_data.index)
                    avg_gain = pd.Series(gain, index=ts_data.index).rolling(window=14).mean()
                    avg_loss = pd.Series(loss, index=ts_data.index).rolling(window=14).mean()
                    rs = avg_gain / avg_loss
                    features['RSI'] = 100 - (100 / (1 + rs))

                if use_macd:
                    ema_12 = ts_data.ewm(span=12, adjust=False).mean()
                    ema_26 = ts_data.ewm(span=26, adjust=False).mean()
                    macd = ema_12 - ema_26
                    signal = macd.ewm(span=9, adjust=False).mean()
                    features['MACD'] = macd
                    features['Signal'] = signal

                if use_momentum:
                    features['Momentum'] = ts_data - ts_data.shift(4)

                if use_stochastic:
                    lowest_low = ts_data.rolling(14).min()
                    highest_high = ts_data.rolling(14).max()
                    features['Stochastic_K'] = ((ts_data - lowest_low) / (highest_high - lowest_low)) * 100

                if use_williams:
                    highest_high = ts_data.rolling(14).max()
                    lowest_low = ts_data.rolling(14).min()
                    features['Williams_%R'] = -100 * ((highest_high - ts_data) / (highest_high - lowest_low))

                features.dropna(inplace=True)

                if model_type == "ARIMA":
                    model = ARIMA(ts_data, order=(p, d, q)).fit()
                    forecast = model.forecast(steps=forecast_days)
                elif model_type == "ETS":
                    model = ExponentialSmoothing(ts_data, trend='add').fit()
                    forecast = model.forecast(steps=forecast_days)
                elif model_type == "Holt-Winters":
                    model = ExponentialSmoothing(ts_data, trend='add', seasonal='add', seasonal_periods=5).fit()
                    forecast = model.forecast(steps=forecast_days)
                elif model_type == "XGBoost":
                    X = pd.DataFrame(index=features.index)
                    X["Lag_1"] = ts_data.shift(1)
                    X["Lag_2"] = ts_data.shift(2)
                    X["Lag_3"] = ts_data.shift(3)
                    X = pd.concat([X, features], axis=1).dropna()
                    y = ts_data.loc[X.index]

                    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, objective='reg:squarederror')
                    model.fit(X, y)

                    last_row = X.iloc[-1:].copy()
                    forecast = []
                    for _ in range(forecast_days):
                        pred = model.predict(last_row)[0]
                        forecast.append(pred)
                        new_row = last_row.shift(-1, axis=1)
                        new_row.iloc[0, -1] = pred
                        last_row = new_row

                elif model_type == "LSTM":
                    from sklearn.preprocessing import MinMaxScaler

                    scaler = MinMaxScaler(feature_range=(0, 1))
                    ts_scaled = scaler.fit_transform(ts_data.values.reshape(-1, 1))

                    X_train, y_train = [], []
                    lookback = 10
                    for i in range(lookback, len(ts_scaled)):
                        X_train.append(ts_scaled[i - lookback:i, 0])
                        y_train.append(ts_scaled[i, 0])

                    X_train, y_train = np.array(X_train), np.array(y_train)
                    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

                    model = Sequential()
                    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
                    model.add(LSTM(units=50, return_sequences=False))
                    model.add(Dense(units=25))
                    model.add(Dense(units=1))

                    model.compile(optimizer='adam', loss='mean_squared_error')
                    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

                    last_values = ts_scaled[-lookback:].reshape(1, lookback, 1)
                    forecast = []
                    for _ in range(forecast_days):
                        next_pred = model.predict(last_values)[0][0]
                        forecast.append(next_pred)
                        last_values = np.roll(last_values, -1, axis=1)
                        last_values[0, -1, 0] = next_pred

                    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

                elif model_type == "RandomForest-XGBoost Hybrid":
                    X = pd.DataFrame(index=features.index)
                    X["Lag_1"] = ts_data.shift(1)
                    X["Lag_2"] = ts_data.shift(2)
                    X["Lag_3"] = ts_data.shift(3)
                    X = pd.concat([X, features], axis=1).dropna()
                    y = ts_data.loc[X.index]

                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, objective='reg:squarederror')

                    rf.fit(X, y)
                    xgb.fit(X, y)

                    last_row = X.iloc[-1:].copy()
                    forecast = []
                    for _ in range(forecast_days):
                        rf_pred = rf.predict(last_row)[0]
                        xgb_pred = xgb.predict(last_row)[0]
                        hybrid_pred = (rf_pred + xgb_pred) / 2
                        forecast.append(hybrid_pred)
                        new_row = last_row.shift(-1, axis=1)
                        new_row.iloc[0, -1] = hybrid_pred
                        last_row = new_row

                forecast_dates = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
                forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(ts_data.index[-60:], ts_data.values[-60:], label="Gerçek Veri", color='blue', linewidth=2)
                ax.plot(forecast_dates, forecast, label="Tahmin", color='red', linestyle='dashed', linewidth=2)
                ax.legend()
                ax.grid()
                st.pyplot(fig)

                csv = forecast_df.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Tahminleri İndir", data=csv, file_name=f"{selected_symbol}_tahminler.csv", mime='text/csv')

        except Exception as e:
            st.error(f"⚠️ Bir hata oluştu: {e}")


with tabs[1]:
    with st.sidebar:
        st.header("🧠 Strateji Ayarları")
        symbol = selected_symbol
        strategies = st.multiselect("Strateji Seçimi:", [
            "Turtle Trade", "Moving Average Crossover", "Donchian Channel Breakout", 
            "Bollinger Bands Breakout", "Parabolic SAR", "Keltner Channel Breakout", 
            "Ichimoku Cloud", "SuperTrend Indicator", "RSI Trend Strategy", "MACD Trend Tracking"
        ])

    if st.button("Stratejiyi Göster"):
        try:
            interval = "1d"
            stock_data = yf.download(symbol, period="720d", interval=interval)
            
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.droplevel(0)
            
            expected_cols = ["Open", "High", "Low", "Close", "Volume"]
            if list(stock_data.columns)[:5] == [symbol] * 5:
                stock_data.columns = expected_cols
            
            missing_cols = [col for col in expected_cols if col not in stock_data.columns]
            if missing_cols:
                st.error(f"Eksik veri sütunları: {missing_cols}. Veri kaynağında problem olabilir.")
                st.stop()
            
            if stock_data.empty or stock_data.isnull().all().all():
                st.error("Veri çekilemedi! Lütfen farklı bir hisse seçin veya veri kaynağını kontrol edin.")
                st.stop()
            
            stock_data.fillna(method="ffill", inplace=True)
            stock_data.fillna(method="bfill", inplace=True)
            def calculate_parabolic_sar(data, af=0.02, max_af=0.2):
                sar = pd.Series(index=data.index)
                trend = 1
                ep = data['High'][0]
                af_value = af
                sar[0] = data['Low'][0]
                for i in range(1, len(data)):
                    sar[i] = sar[i-1] + af_value * (ep - sar[i-1])
        
                    if trend == 1:
                        if data['Low'][i] < sar[i]:
                            trend = -1
                            sar[i] = ep
                            ep = data['Low'][i]
                            af_value = af
                        else:
                            ep = max(ep, data['High'][i])
                    elif trend == -1:
                        if data['High'][i] > sar[i]:
                            trend = 1
                            sar[i] = ep
                            ep = data['High'][i]
                            af_value = af
                        else:
                            ep = min(ep, data['Low'][i])

                    if af_value < max_af:
                        af_value += af
    
                return sar


            
            stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
            stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()
            stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
            stock_data['EMA_20'] = stock_data['Close'].ewm(span=20, adjust=False).mean()
            stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()
            stock_data['Rolling_STD'] = stock_data['Close'].rolling(window=20).std()
            stock_data['Upper_Band'] = stock_data['SMA_20'] + (stock_data['Rolling_STD'] * 2)
            stock_data['Lower_Band'] = stock_data['SMA_20'] - (stock_data['Rolling_STD'] * 2)
            stock_data['HighMax'] = stock_data['High'].rolling(window=20).max()
            stock_data['LowMin'] = stock_data['Low'].rolling(window=20).min()
            stock_data['RSI'] = 100 - (100 / (1 + stock_data['Close'].pct_change().rolling(14).mean()))
            stock_data['MACD'] = stock_data['Close'].ewm(span=12, adjust=False).mean() - stock_data['Close'].ewm(span=26, adjust=False).mean()
            stock_data['Signal_Line'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
            stock_data['Parabolic_SAR'] = calculate_parabolic_sar(stock_data)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=stock_data.index, open=stock_data['Open'], high=stock_data['High'], 
                low=stock_data['Low'], close=stock_data['Close'], name='Mum Grafiği'))
            
            buy_signals = pd.Series(index=stock_data.index, dtype="float64")
            sell_signals = pd.Series(index=stock_data.index, dtype="float64")
            
            if "Turtle Trade" in strategies or "Donchian Channel Breakout" in strategies:
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['HighMax'], mode='lines', name='Yüksek Nokta'))
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['LowMin'], mode='lines', name='Düşük Nokta'))
                
                buy_signals[stock_data['High'] >= stock_data['HighMax']] = stock_data['High']
                sell_signals[stock_data['Low'] <= stock_data['LowMin']] = stock_data['Low']
                
            if "Moving Average Crossover" in strategies:
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_50'], mode='lines', name='50 Günlük SMA'))
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_200'], mode='lines', name='200 Günlük SMA'))
    
                buy_signals = (stock_data['SMA_50'] > stock_data['SMA_200']) & (stock_data['SMA_50'].shift(1) <= stock_data['SMA_200'].shift(1))
                sell_signals = (stock_data['SMA_50'] < stock_data['SMA_200']) & (stock_data['SMA_50'].shift(1) >= stock_data['SMA_200'].shift(1))
    
                stock_data['Buy_Signal'] = np.where(buy_signals, stock_data['Close'], np.nan)
                stock_data['Sell_Signal'] = np.where(sell_signals, stock_data['Close'], np.nan)
    
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Buy_Signal'], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy Signal'))
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Sell_Signal'], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell Signal'))

            
            if "Bollinger Bands Breakout" in strategies:
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Upper_Band'], mode='lines', name='Üst Band'))
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lower_Band'], mode='lines', name='Alt Band'))
                
                buy_signals[stock_data['Close'] > stock_data['Upper_Band']] = stock_data['Close']
                sell_signals[stock_data['Close'] < stock_data['Lower_Band']] = stock_data['Close']
            if "Parabolic SAR" in strategies:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=stock_data.index, open=stock_data['Open'], high=stock_data['High'],
                    low=stock_data['Low'], close=stock_data['Close'], name='Mum Grafiği'
                ))

                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Parabolic_SAR'], 
                                        mode='markers', marker=dict(color='purple', size=4, symbol='circle'),
                                        name='Parabolic SAR'))

                buy_signal = stock_data['Close'] > stock_data['Parabolic_SAR']
                sell_signal = stock_data['Close'] < stock_data['Parabolic_SAR']

                fig.add_trace(go.Scatter(x=stock_data.index[buy_signal], 
                                        y=stock_data['Close'][buy_signal], 
                                        mode='markers', 
                                        marker=dict(color='green', size=10, symbol='triangle-up'), 
                                        name='Buy Signal'))

                fig.add_trace(go.Scatter(x=stock_data.index[sell_signal], 
                                        y=stock_data['Close'][sell_signal], 
                                        mode='markers', 
                                        marker=dict(color='red', size=10, symbol='triangle-down'), 
                                        name='Sell Signal'))

                fig.update_layout(title=f"{symbol} - Parabolic SAR", xaxis_title="Tarih", yaxis_title="Fiyat")
                st.plotly_chart(fig)

            if "Keltner Channel Breakout" in strategies:
                stock_data['ATR'] = stock_data['High'] - stock_data['Low']
                stock_data['Upper_Keltner'] = stock_data['SMA_20'] + (stock_data['ATR'].rolling(window=20).mean() * 1.5)
                stock_data['Lower_Keltner'] = stock_data['SMA_20'] - (stock_data['ATR'].rolling(window=20).mean() * 1.5)

                buy_signals_keltner = stock_data[stock_data['Close'] > stock_data['Upper_Keltner']]
                sell_signals_keltner = stock_data[stock_data['Close'] < stock_data['Lower_Keltner']]

                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Upper_Keltner'], mode='lines', name='Üst Keltner Kanalı'))
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Lower_Keltner'], mode='lines', name='Alt Keltner Kanalı'))

                fig.add_trace(go.Scatter(
                    x=buy_signals_keltner.index, 
                    y=buy_signals_keltner['Close'], 
                    mode='markers', 
                    marker=dict(color='green', size=10, symbol='triangle-up'), 
                    name='Alım Sinyali'
                ))

                fig.add_trace(go.Scatter(
                    x=sell_signals_keltner.index, 
                    y=sell_signals_keltner['Close'], 
                    mode='markers', 
                    marker=dict(color='red', size=10, symbol='triangle-down'), 
                    name='Satış Sinyali'
                ))

            if "Ichimoku Cloud" in strategies:
                stock_data['Tenkan_Sen'] = (stock_data['High'].rolling(window=9).max() + stock_data['Low'].rolling(window=9).min()) / 2
                stock_data['Kijun_Sen'] = (stock_data['High'].rolling(window=26).max() + stock_data['Low'].rolling(window=26).min()) / 2
                stock_data['Senkou_Span_A'] = ((stock_data['Tenkan_Sen'] + stock_data['Kijun_Sen']) / 2).shift(26)
                stock_data['Senkou_Span_B'] = ((stock_data['High'].rolling(window=52).max() + stock_data['Low'].rolling(window=52).min()) / 2).shift(26)
                stock_data['Chikou_Span'] = stock_data['Close'].shift(-26)

                buy_signals_ichimoku = stock_data[stock_data['Close'] > stock_data['Senkou_Span_A']]
                sell_signals_ichimoku = stock_data[stock_data['Close'] < stock_data['Senkou_Span_B']]

                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Senkou_Span_A'], mode='lines', name='Senkou Span A'))
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Senkou_Span_B'], mode='lines', name='Senkou Span B'))
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Tenkan_Sen'], mode='lines', name='Tenkan Sen'))
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Kijun_Sen'], mode='lines', name='Kijun Sen'))
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Chikou_Span'], mode='lines', name='Chikou Span'))

                fig.add_trace(go.Scatter(
                    x=buy_signals_ichimoku.index, 
                    y=buy_signals_ichimoku['Close'], 
                    mode='markers', 
                    marker=dict(color='green', size=10, symbol='triangle-up'), 
                    name='Alım Sinyali'
                ))

                fig.add_trace(go.Scatter(
                    x=sell_signals_ichimoku.index, 
                    y=sell_signals_ichimoku['Close'], 
                    mode='markers', 
                    marker=dict(color='red', size=10, symbol='triangle-down'), 
                    name='Satış Sinyali'
                ))

            if "SuperTrend Indicator" in strategies:
                atr_period = 14
                multiplier = 2.5
                def Supertrend(df, atr_period, multiplier):
                    high = df['High']
                    low = df['Low']
                    close = df['Close']
    
                    price_diffs = [high - low, 
                                high - close.shift(), 
                                close.shift() - low]
                    true_range = pd.concat(price_diffs, axis=1)
                    true_range = true_range.abs().max(axis=1)
    
                    atr = true_range.ewm(alpha=1/atr_period, min_periods=atr_period).mean()
    
                    hl2 = (high + low) / 2
    
                    final_upperband = hl2 + (multiplier * atr)
                    final_lowerband = hl2 - (multiplier * atr)
    
                    supertrend = [True] * len(df)
    
                    for i in range(1, len(df.index)):
                        curr, prev = i, i-1
        
                        if close[curr] > final_upperband[prev]:
                            supertrend[curr] = True
                        elif close[curr] < final_lowerband[prev]:
                            supertrend[curr] = False
                        else:
                            supertrend[curr] = supertrend[prev]
            
                            if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                                final_lowerband[curr] = final_lowerband[prev]
                            if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                                final_upperband[curr] = final_upperband[prev]

                        if supertrend[curr] == True:
                            final_upperband[curr] = np.nan
                        else:
                            final_lowerband[curr] = np.nan
    
                    return pd.DataFrame({
                        'Supertrend': supertrend,
                        'Final Lowerband': final_lowerband,
                        'Final Upperband': final_upperband
                    }, index=df.index)

                supertrend_data = Supertrend(stock_data, atr_period, multiplier)

                stock_data = stock_data.join(supertrend_data)

                buy_signals_supertrend = stock_data[stock_data['Close'] > stock_data['Final Upperband']]
                sell_signals_supertrend = stock_data[stock_data['Close'] < stock_data['Final Lowerband']]

                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Final Lowerband'], mode='lines', name='SuperTrend Lower'))
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Final Upperband'], mode='lines', name='SuperTrend Upper'))

                fig.add_trace(go.Scatter(
                    x=buy_signals_supertrend.index, 
                    y=buy_signals_supertrend['Close'], 
                    mode='markers', 
                    marker=dict(color='green', size=10, symbol='triangle-up'), 
                    name='Buy Signal'
                ))

                fig.add_trace(go.Scatter(
                    x=sell_signals_supertrend.index, 
                    y=sell_signals_supertrend['Close'], 
                    mode='markers', 
                    marker=dict(color='red', size=10, symbol='triangle-down'), 
                    name='Sell Signal'
                ))



            if "RSI Trend Strategy" in strategies:
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI (14)'))
                
                buy_signals[stock_data['RSI'] < -0.30] = stock_data['Close']
                sell_signals[stock_data['RSI'] > 0.70] = stock_data['Close']
            
            if "MACD Trend Tracking" in strategies:
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name='MACD'))
                fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Signal_Line'], mode='lines', name='Signal Line'))
                
                buy_signals[stock_data['MACD'] > stock_data['Signal_Line']] = stock_data['Close']
                sell_signals[stock_data['MACD'] < stock_data['Signal_Line']] = stock_data['Close']
            
            fig.add_trace(go.Scatter(
                x=buy_signals.dropna().index, 
                y=buy_signals.dropna(), 
                mode='markers', 
                marker=dict(color='green', size=10, symbol='triangle-up'), 
                name='Alım Sinyali'
            ))
            
            fig.add_trace(go.Scatter(
                x=sell_signals.dropna().index, 
                y=sell_signals.dropna(), 
                mode='markers', 
                marker=dict(color='red', size=10, symbol='triangle-down'), 
                name='Satış Sinyali'
            ))

            
            fig.update_layout(title=f"{symbol} - Stratejiler", xaxis_title="Tarih", yaxis_title="Fiyat")
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Hata oluştu: {e}")