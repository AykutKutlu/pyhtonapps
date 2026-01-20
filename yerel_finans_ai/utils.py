import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ollama
import PyPDF2

# Kütüphanenin varlığını kontrol eden değişkeni tanımla
try:
    import PyPDF2
    _pdf_available = True
except ImportError:
    _pdf_available = False
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Derin öğrenme katmanları (LSTM için) - PyTorch tercih edilir, TensorFlow alternatif olarak kullanılır
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _torch_available = True
except Exception:
    _torch_available = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    _tf_available = True
except Exception:
    _tf_available = False

# Gizli Markov Modeli (HMM için)
from hmmlearn.hmm import GaussianHMM

def pdf_metin_cikar(uploaded_file):
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages[:10]:
            content = page.extract_text()
            if content:
                text += content
        return text
    except ImportError:
        return "Hata: PyPDF2 kütüphanesi yüklü değil. Lütfen 'pip install PyPDF2' komutuyla yükleyin."
    except Exception as e:
        return f"PDF okunurken hata oluştu: {e}"

def haberleri_yorumlattir(symbol):
    ticker = yf.Ticker(symbol)
    news = ticker.news[:5]  # Son 5 haberi al
    
    haber_metni = ""
    for item in news:
        haber_metni += f"- {item['title']}\n"
        
    prompt = f"{symbol} hissesi hakkında son haber başlıkları şunlar:\n{haber_metni}\nBu haberlerin hisse fiyatı üzerindeki olası etkisini (pozitif/negatif) yorumla."
    
    response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']

def ai_yorum_yap(symbol, summary, strateji_verisi, son_fiyat):
    prompt = f"""
    Sen uzman bir borsa analistisin. Aşağıdaki verilere dayanarak {symbol} hissesi/varlığı için profesyonel bir yorum yap:
    
    1. ŞİRKET/VARLIK ÖZETİ: {summary}
    2. GÜNCEL FİYAT: {son_fiyat}
    3. UYGULANAN TEKNİK STRATEJİLER VE SİNYALLER: {strateji_verisi}
    
    Analizinde şunlara değin:
    - Şirketin faaliyet alanı ile teknik görünüm uyumlu mu?
    - Sinyaller genel olarak 'Al' mı yoksa 'Sat' mı ağırlıklı?
    - Yatırımcılar hangi seviyelere dikkat etmeli?
    
    Yazını kısa, öz ve profesyonel bir tonda tut. Sonuna 'Bu bir yatırım tavsiyesi değildir.' notunu ekle.
    """
    try:
        response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        return f"Yapay zeka yorumu şu an alınamıyor: {e}"

def ai_tahmin_yorumu(symbol, model_ismi, tahmin_fiyatı, son_fiyat):
    değişim = ((tahmin_fiyatı - son_fiyat) / son_fiyat) * 100
    yon = "Artış" if değişim > 0 else "Düşüş"
    
    prompt = f"""
    Sen kıdemli bir piyasa stratejistisin. {symbol} varlığı için {model_ismi} modeli kullanılarak bir tahmin yapıldı.
    
    Veriler:
    - Güncel Fiyat: {son_fiyat:.2f}
    - 15 Gün Sonraki Tahmin: {tahmin_fiyatı:.2f}
    - Beklenen Değişim: %{değişim:.2f} ({yon})
    
    Görev: 
    Bu istatistiksel tahmini, genel borsa psikolojisi ve teknik analiz prensipleriyle yorumla. 
    Bu modelin yanılma payı olabileceğini hatırlatarak, yatırımcıya bu süreçte hangi indikatörleri (RSI, hacim vb.) takip etmesi gerektiğini söyle.
    """
    
    try:
        response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        return "Tahmin yorumu şu an oluşturulamadı."

def train_arima_model(data, p, d, q, forecast_days):
    """
    ARIMA modelini eğitir ve tahmin sonuçlarını döner.
    """
    try:
        # Model eğitimi
        model = ARIMA(data, order=(p, d, q)).fit()
        # Tahmin yapma
        forecast = model.forecast(steps=forecast_days)
        return forecast, None
    except Exception as e:
        # Hata durumunda hatayı döndür
        return None, str(e)
    
def train_ets_model(data, trend_type='add', forecast_days=30):
    """
    ETS (Exponential Smoothing) modelini eğitir ve tahmin üretir.
    """
    try:
        # Model tanımlama ve eğitme
        model = ExponentialSmoothing(data, trend=trend_type).fit()
        # Tahmin üretme
        forecast = model.forecast(steps=forecast_days)
        return forecast, None
    except Exception as e:
        # Hata durumunda hata mesajını döner
        return None, str(e)
    
def train_holt_winters_model(data, trend='add', seasonal='add', sp=5, forecast_days=30):
    """
    Holt-Winters (Üçlü Üstel Düzeltme) modelini eğitir ve tahmin üretir.
    """
    try:
        # Model tanımlama ve eğitme
        model = ExponentialSmoothing(
            data, 
            trend=trend, 
            seasonal=seasonal, 
            seasonal_periods=sp
        ).fit()
        
        # Tahmin üretme
        forecast = model.forecast(steps=forecast_days)
        return forecast, None
    except Exception as e:
        # Hata durumunda hata mesajını döner
        return None, str(e)
    
def train_xgboost_model(data, extra_features, forecast_days=30):
    """
    XGBoost modelini gecikmeli özelliklerle eğitir ve recursive tahmin yapar.
    """
    try:
        # 1. Özellik Mühendisliği (Lag Features)
        X = pd.DataFrame(index=data.index)
        X["Lag_1"] = data.shift(1)
        X["Lag_2"] = data.shift(2)
        X["Lag_3"] = data.shift(3)
        
        # Dışsal özellikleri ekle ve eksik değerleri temizle
        X = pd.concat([X, extra_features], axis=1).dropna()
        y = data.loc[X.index]

        # 2. Model Tanımlama ve Eğitim
        model = XGBRegressor(
            objective='reg:squarederror',
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X, y)

        # 3. İteratif (Recursive) Tahmin Süreci
        last_row = X.iloc[-1:].copy()
        forecast = []
        
        for _ in range(forecast_days):
            pred = model.predict(last_row)[0]
            forecast.append(pred)
            
            # Gecikmeli değerleri bir sonraki adım için kaydır (Shift)
            # Not: Bu mantık Lag_1, Lag_2, Lag_3 yapınıza göredir
            new_row = last_row.copy()
            new_row["Lag_3"] = new_row["Lag_2"]
            new_row["Lag_2"] = new_row["Lag_1"]
            new_row["Lag_1"] = pred # En yeni tahmin Lag_1 olur
            
            last_row = new_row
            
        return forecast, None
    except Exception as e:
        return None, str(e)
    
def train_hmm_model(data, n_components=2, forecast_days=30):
    """
    HMM modelini eğitir ve mevcut rejime göre gelecek fiyat simülasyonu yapar.
    """
    try:
        # 1. Getiri (Returns) hesaplama ve hazırlama
        returns = data.pct_change().dropna().values.reshape(-1, 1)
        
        # 2. HMM Model Eğitimi
        hmm_model = GaussianHMM(
            n_components=n_components, 
            covariance_type="diag", 
            n_iter=1000, 
            random_state=42
        ).fit(returns)
        
        # 3. Mevcut rejimleri (gizli durumları) tespit etme
        hidden_states = hmm_model.predict(returns)
        last_state = hidden_states[-1]
        
        # 4. Gelecek Simülasyonu
        forecast = []
        current_price = data.iloc[-1]
        
        for _ in range(forecast_days):
            # Mevcut rejimin ortalama ve varyansına göre rastgele getiri örnekle
            mean = hmm_model.means_[last_state]
            covar = np.sqrt(hmm_model.covars_[last_state])
            next_return = np.random.normal(mean, covar)[0]
            
            # Fiyatı güncelle ve listeye ekle
            current_price *= (1 + next_return)
            forecast.append(current_price)
            
        return forecast, None
    except Exception as e:
        return None, str(e)
    
def train_lstm_model(data, forecast_days=30, lookback=50):
    """PyTorch tercihli LSTM. Eğer PyTorch yoksa, TensorFlow (Keras) fallback kullanılır."""
    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1)).astype(np.float32)

        if len(scaled_data) <= lookback:
            return None, f"Yetersiz veri! En az {lookback + 1} veri noktası gerekli."

        # --- PyTorch implementation ---
        if '_torch_available' in globals() and _torch_available:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            class PyLSTM(nn.Module):
                def __init__(self, input_size=1, h1=32, h2=16, dropout=0.1):
                    super().__init__()
                    self.lstm1 = nn.LSTM(input_size, h1, batch_first=True)
                    self.dropout1 = nn.Dropout(dropout)
                    self.lstm2 = nn.LSTM(h1, h2, batch_first=True)
                    self.dropout2 = nn.Dropout(dropout)
                    self.fc = nn.Linear(h2, 1)

                def forward(self, x):
                    out, _ = self.lstm1(x)
                    out = self.dropout1(out)
                    out, _ = self.lstm2(out)
                    out = self.dropout2(out[:, -1, :])
                    return self.fc(out)

            # Prepare training data
            X_train = []
            y_train = []
            for i in range(lookback, len(scaled_data)):
                X_train.append(scaled_data[i - lookback:i, 0])
                y_train.append(scaled_data[i, 0])

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            X_train_t = torch.tensor(X_train).unsqueeze(2).to(device)  # (N, seq, 1)
            y_train_t = torch.tensor(y_train).unsqueeze(1).to(device)  # (N, 1)

            model = PyLSTM().to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            model.train()
            epochs = 20
            batch_size = 32
            n = len(X_train_t)
            for epoch in range(epochs):
                perm = np.random.permutation(n)
                for i in range(0, n, batch_size):
                    idx = perm[i:i+batch_size]
                    bx = X_train_t[idx]
                    by = y_train_t[idx]
                    optimizer.zero_grad()
                    preds = model(bx)
                    loss = criterion(preds, by)
                    loss.backward()
                    optimizer.step()

            # Recursive forecast
            model.eval()
            current_batch = torch.tensor(scaled_data[-lookback:]).unsqueeze(0).unsqueeze(2).to(device)
            forecast_scaled = []
            with torch.no_grad():
                for _ in range(forecast_days):
                    pred = model(current_batch).cpu().numpy().flatten()[0]
                    forecast_scaled.append(pred)
                    # shift and append
                    new_seq = np.append(current_batch.cpu().numpy()[0,:,0][1:], pred)
                    current_batch = torch.tensor(new_seq).reshape(1, lookback, 1).to(device)

            forecast_rescaled = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

            # --- KRİTİK DÜZELTME ---
            last_real_value = float(data.iloc[-1])
            forecast_final = np.insert(forecast_rescaled, 0, last_real_value)

            return forecast_final, None

        # --- TensorFlow fallback (eski davranış) ---
        elif '_tf_available' in globals() and _tf_available:
            X_train, y_train = [], []
            for i in range(lookback, len(scaled_data)):
                X_train.append(scaled_data[i - lookback:i, 0])
                y_train.append(scaled_data[i, 0])

            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

            model = Sequential([
                LSTM(32, return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.1),
                LSTM(16),
                Dropout(0.1),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

            current_batch = scaled_data[-lookback:].reshape(1, lookback, 1)
            forecast_scaled = []

            for _ in range(forecast_days):
                current_pred = model.predict(current_batch, verbose=0)[0]
                forecast_scaled.append(current_pred)
                new_val = current_pred.reshape(1, 1, 1)
                current_batch = np.append(current_batch[:, 1:, :], new_val, axis=1)

            forecast_rescaled = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

            last_real_value = float(data.iloc[-1])
            forecast_final = np.insert(forecast_rescaled, 0, last_real_value)

            return forecast_final, None

        else:
            return None, "Ne PyTorch ne de TensorFlow bulunamadı. Lütfen 'pip install torch' veya 'pip install tensorflow' ile yükleyin."

    except Exception as e:
        return None, str(e)
    
def train_hybrid_rf_xgb_model(data, extra_features, forecast_days=30):
    """
    RandomForest ve XGBoost modellerini birleştirerek hibrit tahmin yapar.
    """
    try:
        # 1. Özellik Mühendisliği (Lag Features)
        X = pd.DataFrame(index=data.index)
        X["Lag_1"] = data.shift(1)
        X["Lag_2"] = data.shift(2)
        X["Lag_3"] = data.shift(3)
        
        # Dışsal özellikleri birleştir ve temizle
        X = pd.concat([X, extra_features], axis=1).dropna()
        y = data.loc[X.index]

        # 2. Modellerin Tanımlanması ve Eğitilmesi
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, 
                                 objective='reg:squarederror', random_state=42)

        rf_model.fit(X, y)
        xgb_model.fit(X, y)

        # 3. İteratif Hibrit Tahmin Süreci
        last_row = X.iloc[-1:].copy()
        forecast = []

        for _ in range(forecast_days):
            # İki modelden de tahmin al
            rf_pred = rf_model.predict(last_row)[0]
            xgb_pred = xgb_model.predict(last_row)[0]
            

                        # Hibrit tahmin (Ortalama)
            hybrid_pred = (rf_pred + xgb_pred) / 2
            forecast.append(hybrid_pred)
            
            # Bir sonraki adım için Lag değerlerini güncelle
            new_row = last_row.copy()
            new_row["Lag_3"] = new_row["Lag_2"]
            new_row["Lag_2"] = new_row["Lag_1"]
            new_row["Lag_1"] = hybrid_pred
            
            last_row = new_row
            
        return forecast, None
    except Exception as e:
        return None, str(e)
    
def display_forecast_results(ts_data, forecast, forecast_days, symbol):
    # Tahmin serisi artık (forecast_days + 1) uzunluğunda olduğu için
    # tarihleri son gerçek tarihten başlatıyoruz.
    forecast_dates = pd.date_range(
        start=ts_data.index[-1],  # Buradaki +1 gün ekleme işlemini sildik
        periods=len(forecast),    # forecast_final uzunluğu kadar tarih üretir
        freq='B'
    )
    
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})

    fig, ax = plt.subplots(figsize=(10, 5))
    # Son 60 günün gerçek verisi
    ax.plot(ts_data.index[-60:], ts_data.values[-60:], label="Gerçek Veri", color='blue', linewidth=2)
    # Tahmin verisi (Artık gerçek verinin son noktasından başlıyor)
    ax.plot(forecast_dates, forecast, label="Tahmin", color='red', linestyle='dashed', linewidth=2)
    
    ax.set_title(f"{symbol} Fiyat Tahmini")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    # CSV İndirme Butonu
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Tahminleri CSV Olarak İndir", 
        data=csv, 
        file_name=f"{symbol}_tahminler.csv", 
        mime='text/csv'
    )
    return forecast_df

def get_ai_forecast_analysis(symbol, model_type, ts_data, forecast):
    """
    Tahmin sonuçlarını Ollama (Llama 3.1) kullanarak analiz eder.
    """
    try:
        # 1. Veri Tiplerini Güvenli Şekilde Dönüştürme
        raw_last_price = ts_data.values[-1]
        son_fiyat = float(raw_last_price.item() if hasattr(raw_last_price, 'item') else raw_last_price)
        
        if isinstance(forecast, (pd.Series, pd.DataFrame)):
            tahmin_fiyat = float(forecast.iloc[-1])
        elif isinstance(forecast, (list, np.ndarray)):
            tahmin_fiyat = float(forecast[-1])
        else:
            tahmin_fiyat = float(forecast)
            
        fark_yuzde = ((tahmin_fiyat - son_fiyat) / son_fiyat) * 100
        yon = "Artış" if fark_yuzde > 0 else "Düşüş"
        
        # 2. Şirket Bilgisi Çekme
        ticker = yf.Ticker(symbol)
        business_summary = ticker.info.get('longBusinessSummary', 'Şirket özeti bulunamadı.')

        # 3. Prompt Hazırlama
        prompt = f"""
        Hisse/Varlık: {symbol}
        Model: {model_type}
        Güncel Fiyat: {son_fiyat:.2f}
        Tahmin Edilen Fiyat (Vade Sonu): {tahmin_fiyat:.2f}
        Beklenen Değişim: %{fark_yuzde:.2f} ({yon})
        Şirket Özeti: {business_summary[:500]}...
        
        Görev: Yukarıdaki teknik tahmin verilerini ve şirket profilini analiz et. 
        Piyasa beklentisini, riskleri ve potansiyeli yorumla. 
        Yanıtın sonunda mutlaka 'Yatırım tavsiyesi değildir' uyarısı ekle.
        """

        # 4. Ollama Çağrısı
        response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content'], None

    except Exception as e:
        return None, str(e)
    
def calculate_parabolic_sar(data, af=0.02, max_af=0.2):
    sar = pd.Series(index=data.index)
    trend, ep, af_value = 1, data['High'][0], af
    sar[0] = data['Low'][0]
    for i in range(1, len(data)):
        sar[i] = sar[i-1] + af_value * (ep - sar[i-1])
        if trend == 1:
            if data['Low'][i] < sar[i]:
                trend, sar[i], ep, af_value = -1, ep, data['Low'][i], af
            else:
                ep = max(ep, data['High'][i])
        else:
            if data['High'][i] > sar[i]:
                trend, sar[i], ep, af_value = 1, ep, data['High'][i], af
            else:
                ep = min(ep, data['Low'][i])
        if af_value < max_af: af_value += af
    return sar

def apply_technical_indicators(df):
    """Tüm teknik göstergeleri hesaplar."""
    df = df.copy()
    # Hareketli Ortalamalar
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    # Kanallar ve Risk
    df['Rolling_STD'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['Rolling_STD'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['Rolling_STD'] * 2)
    df['HighMax'] = df['High'].rolling(window=20).max()
    df['LowMin'] = df['Low'].rolling(window=20).min()
    # Momentum
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Parabolic_SAR'] = calculate_parabolic_sar(df)
    return df

def create_strategy_plot(df, selected_strategies, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Mum Grafiği'))
    
    buy_signals = pd.Series(index=df.index, dtype="float64")
    sell_signals = pd.Series(index=df.index, dtype="float64")

    # Kaplumbağa & Donchian
    if any(s in selected_strategies for s in ["Turtle Trade", "Donchian Channel Breakout"]):
        fig.add_trace(go.Scatter(x=df.index, y=df['HighMax'], name='Kanal Üst', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['LowMin'], name='Kanal Alt', line=dict(dash='dot')))
        buy_signals[df['High'] >= df['HighMax']] = df['High']
        sell_signals[df['Low'] <= df['LowMin']] = df['Low']

    # MA Crossover
    if "Moving Average Crossover" in selected_strategies:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200'))
        buy_idx = (df['SMA_50'] > df['SMA_200']) & (df['SMA_50'].shift(1) <= df['SMA_200'].shift(1))
        sell_idx = (df['SMA_50'] < df['SMA_200']) & (df['SMA_50'].shift(1) >= df['SMA_200'].shift(1))
        buy_signals[buy_idx] = df['Close']
        sell_signals[sell_idx] = df['Close']

    # Sinyalleri Grafiğe Ekle
    fig.add_trace(go.Scatter(x=buy_signals.dropna().index, y=buy_signals.dropna(), mode='markers', 
                             marker=dict(color='green', size=12, symbol='triangle-up'), name='AL'))
    fig.add_trace(go.Scatter(x=sell_signals.dropna().index, y=sell_signals.dropna(), mode='markers', 
                             marker=dict(color='red', size=12, symbol='triangle-down'), name='SAT'))
    
    fig.update_layout(title=f"{symbol} Teknik Analiz", xaxis_rangeslider_visible=False)
    return fig

def calculate_technical_features(ts_data, stock_data, timeframes, config):
    """
    Kullanıcının seçtiği indikatörlere göre özellik (feature) setini hazırlar.
    """
    features = pd.DataFrame(index=ts_data.index)
    
    # Hareketli Ortalamalar (Varsayılan)
    for t in timeframes:
        features[f'MA_{t}'] = ts_data.rolling(window=t).mean()

    if config['use_volume']:
        features['Volume'] = stock_data['Volume']

    if config['use_volatility']:
        features['Volatility'] = ts_data.pct_change().rolling(10).std()

    if config['use_rsi']:
        delta = ts_data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['RSI'] = 100 - (100 / (1 + rs))

    if config['use_macd']:
        ema_12 = ts_data.ewm(span=12, adjust=False).mean()
        ema_26 = ts_data.ewm(span=26, adjust=False).mean()
        features['MACD'] = ema_12 - ema_26
        features['Signal'] = features['MACD'].ewm(span=9, adjust=False).mean()

    if config['use_momentum']:
        features['Momentum'] = ts_data - ts_data.shift(4)

    if config['use_stochastic']:
        low_14 = ts_data.rolling(14).min()
        high_14 = ts_data.rolling(14).max()
        features['Stochastic_K'] = ((ts_data - low_14) / (high_14 - low_14)) * 100

    if config['use_williams']:
        high_14 = ts_data.rolling(14).max()
        low_14 = ts_data.rolling(14).min()
        features['Williams_%R'] = -100 * ((high_14 - ts_data) / (high_14 - low_14))

    return features.dropna()

def get_symbol_lists(market_type):
    """Piyasa türüne göre sembol listesini döner."""
    if market_type == "BIST 100":
        return [
            "GARAN.IS", "KCHOL.IS", "THYAO.IS", "FROTO.IS", "ISCTR.IS", "BIMAS.IS", "TUPRS.IS", 
            "ENKAI.IS", "ASELS.IS", "AKBNK.IS", "YKBNK.IS", "VAKBN.IS", "TCELL.IS", "SAHOL.IS", 
            "SASA.IS", "TTKOM.IS", "EREGL.IS", "CCOLA.IS", "PGSUS.IS", "SISE.IS", # ... listenin devamı
        ]
    else:
        return [
            "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", 
            "AVAX-USD", "DOT-USD", "MATIC-USD", "LTC-USD", "BCH-USD", "LINK-USD", # ... listenin devamı
        ]

def ask_ai_about_pdf(pdf_text, question):
    """PDF içeriği hakkında AI'ya soru sorar."""
    try:
        prompt = f"""
        Aşağıdaki bilanço/rapor metnine dayanarak soruyu cevapla. 
        Sadece sağlanan metindeki bilgileri kullan ve profesyonel bir finansal dil kullan.
        
        Metin Özeti: {pdf_text[:4000]}
        
        Soru: {question}
        """
        response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content'], None
    except Exception as e:
        return None, str(e)


# --- Notlar (README'ye veya uygulama arayüzüne ekleyin) ---
# • Bu projede LSTM için öncelikli backend PyTorch olarak ayarlandı.
# • Eğer PyTorch yüklü değilse, kod TensorFlow/Keras kullanımına geri döner (mevcut davranış korunur).
# • PyTorch kurmak için: pip install torch (veya CUDA destekliyse uygun sürümü seçin).
# • Kaynak: Eğer GPU üzerinde çalıştıracaksanız uygun CUDA sürümü ile torch kurmayı unutmayın.
