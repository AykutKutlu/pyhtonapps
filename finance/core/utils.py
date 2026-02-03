import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import argrelextrema # Tepe ve dip tespiti için şart
import warnings

# Çirkin uyarıları gizlemek için (Opsiyonel)
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np


def kapsamli_teknik_analiz(df):
    """
    Doğrulama Mumlu Sniper Motoru. 
    NameError ve KeyError hatalarına karşı tam korumalı.
    """
    if df is None or len(df) < 50:
        return {"durum": "VERİ YETERSİZ", "skor": 0, "fiyat": 0, "notlar": "Veri yetersiz."}

    df_tech = df.copy()
    close_ser = df_tech['Close']
    
    # --- 1. GÖSTERGE HESAPLAMALARI ---
    df_tech['SMA50'] = close_ser.rolling(50).mean()
    df_tech['SMA200'] = close_ser.rolling(200).mean()
    
    # Bollinger Bantları
    sma20 = close_ser.rolling(20).mean()
    std20 = close_ser.rolling(20).std()
    df_tech['Upper_BB'] = sma20 + (2.1 * std20)
    df_tech['Lower_BB'] = sma20 - (2.1 * std20)

    # MACD & RSI
    exp1 = close_ser.ewm(span=12, adjust=False).mean()
    exp2 = close_ser.ewm(span=26, adjust=False).mean()
    df_tech['MACD'] = exp1 - exp2
    df_tech['Signal_Line'] = df_tech['MACD'].ewm(span=9, adjust=False).mean()
    
    # RSI Hesaplama ve Sabitleme (Hatanın Çözümü Burada)
    delta = close_ser.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df_tech['RSI'] = 100 - (100 / (1 + (gain / loss)))
    
    # Değişkenleri en başta tanımlayalım ki NameError vermesin
    current_price = float(close_ser.iloc[-1])
    rsi_val = float(df_tech['RSI'].iloc[-1])
    sma50_last = float(df_tech['SMA50'].iloc[-1])
    
    # ATR
    tr = pd.concat([df_tech['High']-df_tech['Low'], 
                    (df_tech['High']-close_ser.shift()).abs(), 
                    (df_tech['Low']-close_ser.shift()).abs()], axis=1).max(axis=1)
    df_tech['ATR'] = tr.rolling(window=14).mean()
    current_atr = float(df_tech['ATR'].iloc[-1])
    
    # --- 2. 2-BAR TEYİTLİ SİNYAL ÜRETİMİ ---
    all_signals = []
    df_scan = df_tech.tail(720)
    last_signal_index = -20 

    for i in range(20, len(df_scan)):
        c_price, p_price, pp_price = df_scan['Close'].iloc[i], df_scan['Close'].iloc[i-1], df_scan['Close'].iloc[i-2]
        c_sma, p_sma = df_scan['SMA50'].iloc[i], df_scan['SMA50'].iloc[i-1]
        c_macd, c_sig = df_scan['MACD'].iloc[i], df_scan['Signal_Line'].iloc[i]

        # 🟢 BUY TEYİDİ: 2 Bar kuralı + MACD Onayı
        if (c_price > c_sma and p_price > p_sma) and (pp_price <= df_scan['SMA50'].iloc[i-2]) and (c_macd > c_sig):
            if i - last_signal_index > 15:
                all_signals.append({
                    "type": "BUY", "price": float(c_price), "date": df_scan.index[i],
                    "low": float(df_scan['Low'].iloc[i]), "high": float(df_scan['High'].iloc[i]),
                    "match": True
                })
                last_signal_index = i

        # 🔴 SELL TEYİDİ: 2 Bar kuralı + MACD Onayı
        elif (c_price < c_sma and p_price < p_sma) and (pp_price >= df_scan['SMA50'].iloc[i-2]) and (c_macd < c_sig):
            if i - last_signal_index > 15:
                all_signals.append({
                    "type": "SELL", "price": float(c_price), "date": df_scan.index[i],
                    "low": float(df_scan['Low'].iloc[i]), "high": float(df_scan['High'].iloc[i]),
                    "match": True
                })
                last_signal_index = i

    # --- 3. DURUM TESPİTİ ---
    if all_signals:
        last_sig = all_signals[-1]
        s_type, signal_date, signal_price = last_sig['type'], last_sig['date'], last_sig['price']
        sig_low, sig_high = last_sig['low'], last_sig['high']
    else:
        s_type, signal_date, signal_price = "NEUTRAL", df_tech.index[-1], current_price
        sig_low, sig_high = float(df_tech['Low'].iloc[-1]), float(df_tech['High'].iloc[-1])

    # Seviyeler
    hedef_fiyat = signal_price + (current_atr * 4.5) if s_type == "BUY" else signal_price - (current_atr * 4.5)
    stop_loss = sig_low - (current_atr * 1.5) if s_type == "BUY" else sig_high + (current_atr * 1.5)
    
    sinyal_gecerli = True
    if s_type == "BUY" and current_price >= hedef_fiyat: sinyal_gecerli = False
    elif s_type == "SELL" and current_price <= hedef_fiyat: sinyal_gecerli = False

    # Skorlama
    skor = 0
    analiz_notlari = []
    
    if s_type != "NEUTRAL":
        # RSI kontrolü
        if 30 < rsi_val < 70: 
            skor += 1
            analiz_notlari.append("RSI Dengeli")
        elif rsi_val < 30:
            if s_type == "BUY": skor += 1; analiz_notlari.append("RSI Aşırı Satıldı")
        elif rsi_val > 70:
            if s_type == "SELL": skor += 1; analiz_notlari.append("RSI Aşırı Satıldı")
        
        # Fiyat SMA50 kontrolü
        if (s_type == "BUY" and current_price > sma50_last): 
            skor += 1
            analiz_notlari.append("Fiyat SMA50 Üstünde")
        elif (s_type == "SELL" and current_price < sma50_last): 
            skor += 1
            analiz_notlari.append("Fiyat SMA50 Altında")
        
        # MACD kontrolü
        if (s_type == "BUY" and df_tech['MACD'].iloc[-1] > df_tech['Signal_Line'].iloc[-1]): 
            skor += 1
            analiz_notlari.append("MACD Pozitif")
        elif (s_type == "SELL" and df_tech['MACD'].iloc[-1] < df_tech['Signal_Line'].iloc[-1]): 
            skor += 1
            analiz_notlari.append("MACD Negatif")
        
        # Hacim kontrolü
        if df_tech['Volume'].iloc[-1] > (df_tech['Volume'].tail(20).mean() * 1.2): 
            skor += 1
            analiz_notlari.append("Hacim Güçlü")

    durum = ("🔥 MATCH " if skor >= 3 else "") + (s_type if sinyal_gecerli else "NÖTR")

    # --- 4. RETURN (Hatasız Yapı) ---
    return {
        "fiyat": current_price, 
        "rsi": rsi_val, 
        "skor": skor, 
        "durum": durum,
        "signal_type": s_type if sinyal_gecerli else "NEUTRAL",
        "all_signals": all_signals, 
        "signal_date": signal_date, 
        "signal_price": signal_price,
        "hedef": hedef_fiyat, 
        "stop": stop_loss, 
        "en_guclu_alis": sig_low if s_type == "BUY" else sig_high,
        "kazanc_beklentisi": abs(((hedef_fiyat / current_price) - 1) * 100) if sinyal_gecerli else 0,
        "notlar": ", ".join(analiz_notlari) if analiz_notlari else "Teyit bekleniyor",
        "df": df_tech
    }
def piyasa_radari_tara(sembol_listesi, ui_names):
    sonuclar = []
    if not sembol_listesi: return []

    try:
        data_all = yf.download(sembol_listesi, period="2y", interval="1d", group_by='ticker', progress=False)
    except Exception as e:
        print(f"Veri indirme hatası: {e}")
        return []
    
    for symbol in sembol_listesi:
        try:
            if len(sembol_listesi) > 1:
                df = data_all[symbol].dropna()
            else:
                df = data_all.dropna()
            
            if len(df) < 50: continue
            
            # --- ANALİZ MOTORUNU ÇALIŞTIR ---
            analiz = kapsamli_teknik_analiz(df)
            
            # Temel Bilgileri Ekle
            analiz['display_name'] = ui_names.get(symbol, symbol)
            analiz['symbol'] = symbol
            
            # --- KRİTİK DÜZELTME: FİLTREYİ KALDIRDIK ---
            # Sinyal skoru 0 olsa bile (Nötr/Hedef/İzle), 
            # hepsini listeye ekliyoruz ki UI tarafındaki sekmelerde ayıklayabilelim.
            sonuclar.append(analiz)
                
        except Exception as e:
            print(f"{symbol} tarama hatası: {e}")
            continue
            
    # Sonuçları skora göre sırala (En güçlüler yine üstte kalsın)
    return sorted(sonuclar, key=lambda x: x['skor'], reverse=True)

def calculate_fibonacci_levels(df):
    """
    Son 1 yıllık en yüksek ve en düşük değerlere göre 
    Fibonacci düzeltme seviyelerini hesaplar.
    """
    recent_data = df.tail(252)
    max_price = recent_data['High'].max()
    min_price = recent_data['Low'].min()
    diff = max_price - min_price
    
    levels = {
        "0.0%": max_price,
        "23.6%": max_price - 0.236 * diff,
        "38.2%": max_price - 0.382 * diff,
        "50.0%": max_price - 0.5 * diff,
        "61.8%": max_price - 0.618 * diff,
        "100.0%": min_price
    }
    return levels


def dinamik_trend_analizi(df):
    temp_df = df.copy().tail(120).reset_index()
    
    idx_min = argrelextrema(temp_df.Low.values, np.less_equal, order=7)[0]
    idx_max = argrelextrema(temp_df.High.values, np.greater_equal, order=7)[0]
    
    if len(idx_min) < 3 or len(idx_max) < 3:
        return []

    lines = []
    z = np.polyfit(temp_df.index, temp_df.Close.values, 1)
    mevcut_egim = z[0]
    
    # Kanal genişliğini hata almadan hesapla
    # Bu yöntem listelerin uzunluğuna bakmaz, tüm veri setini kullanır
    safe_offset = (temp_df['High'] - temp_df['Low']).mean() * 2.5 

    # DURUM A: DÜŞÜŞ ŞEMASI
    if mevcut_egim < 0:
        slope_h, intercept_h = np.polyfit(idx_max, temp_df.High.values[idx_max], 1)
        
        # Ana Direnç
        lines.append({
            'type': 'Dinamik Direnç (Düşüş)',
            'color': '#FF3D00',
            'width': 5,
            'x': [temp_df.Date.iloc[idx_max[0]], temp_df.Date.iloc[-1]],
            'y': [slope_h * idx_max[0] + intercept_h, slope_h * (len(temp_df)-1) + intercept_h]
        })
        
        # Gölge Alt Hattı (Hatasız)
        lines.append({
            'type': 'Dinamik Destek (Gölge)',
            'color': 'rgba(255, 61, 0, 0.2)', 
            'width': 1,
            'x': [temp_df.Date.iloc[idx_max[0]], temp_df.Date.iloc[-1]],
            'y': [(slope_h * idx_max[0] + intercept_h) - safe_offset, (slope_h * (len(temp_df)-1) + intercept_h) - safe_offset]
        })

    # DURUM B: YÜKSELİŞ ŞEMASI
    else:
        slope_l, intercept_l = np.polyfit(idx_min, temp_df.Low.values[idx_min], 1)
        
        # Ana Destek
        lines.append({
            'type': 'Dinamik Destek (Yükseliş)',
            'color': '#00FF88',
            'width': 5,
            'x': [temp_df.Date.iloc[idx_min[0]], temp_df.Date.iloc[-1]],
            'y': [slope_l * idx_min[0] + intercept_l, slope_l * (len(temp_df)-1) + intercept_l]
        })
        
        # Gölge Üst Hattı (Hatasız)
        lines.append({
            'type': 'Dinamik Direnç (Gölge)',
            'color': 'rgba(0, 255, 136, 0.2)',
            'width': 1,
            'x': [temp_df.Date.iloc[idx_min[0]], temp_df.Date.iloc[-1]],
            'y': [(slope_l * idx_min[0] + intercept_l) + safe_offset, (slope_l * (len(temp_df)-1) + intercept_l) + safe_offset]
        })

    return lines


def tarihsel_seviye_analizi(df):
    """
    Geçmişteki majör tepe (Mavi Direnç) ve dip (Kırmızı Destek) noktalarını bulur.
    """
    temp_df = df.copy().tail(500).reset_index()
    
    # Majör Dönüş Noktaları (Order=30: Yaklaşık 1.5 aylık en uç noktalar)
    idx_max = argrelextrema(temp_df.High.values, np.greater_equal, order=30)[0]
    idx_min = argrelextrema(temp_df.Low.values, np.less_equal, order=30)[0]
    
    levels = []
    
    # Zirveler (Mavi Dirençler)
    for i in idx_max:
        levels.append({
            'val': temp_df.High.iloc[i],
            'date': temp_df.Date.iloc[i],
            'type': 'Direnç',
            'color': 'rgba(0, 150, 255, 0.7)' # Parlak Mavi
        })
        
    # Dipler (Kırmızı Destekler)
    for i in idx_min:
        levels.append({
            'val': temp_df.Low.iloc[i],
            'date': temp_df.Date.iloc[i],
            'type': 'Destek',
            'color': 'rgba(255, 145, 0, 0.8)'
        })
        
    return levels

##########strateji motorları##########

def osilatör_analizi(df):
    """RSI, MFI gibi osilatörleri hesaplar."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return {'rsi': rsi.iloc[-1]}
def hacim_profili_hesapla(df):
    """Fiyat seviyelerine göre biriken hacmi (VAP) simüle eder."""
    # Son 100 mumun kapanış verisini al
    recent_data = df['Close'].tail(100)
    
    # 20 fiyat dilimi oluştur ve her dilime düşen mum sayısını hesapla
    bins = pd.cut(recent_data, bins=20)
    v_profile = bins.value_counts().sort_index()
    
    # HATA ÖNLEYİCİ: Sözlüğü manuel döngü ile kuruyoruz
    cleaned_results = {}
    for interval, count in v_profile.items():
        # interval.mid fiyat seviyesini, count ise o seviyedeki yoğunluğu temsil eder
        # count değerini açıkça standart Python int tipine çeviriyoruz
        cleaned_results[float(interval.mid)] = int(count)
        
    return cleaned_results
def gelismis_strateji_motoru(df):
    """Trend, ATR stop/hedef ve R/R hesaplayan motor."""
    # Verilerin sayısal olduğundan emin olalım
    last_price = float(df['Close'].iloc[-1])
    
    # ATR (Average True Range) Hesaplaması
    high_low = df['High'] - df['Low']
    high_cp = np.abs(df['High'] - df['Close'].shift())
    low_cp = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    
    # SMA 50 ile Trend Belirleme
    sma_50 = float(df['Close'].rolling(50).mean().iloc[-1])
    trend = "BOĞA" if last_price > sma_50 else "AYI"
    
    # Çarpan (Hisse senedi için 2.0 - 2.5 idealdir)
    mult = 2.5 
    
    if trend == "BOĞA":
        signal = "BUY"
        stop = last_price - (atr * mult)
        hedef = last_price + (atr * mult * 2)
    else:
        signal = "SELL"
        stop = last_price + (atr * mult)
        hedef = last_price - (atr * mult * 2)
        
    # Sıfıra bölünme hatası korumalı R/R
    risk = abs(last_price - stop)
    reward = abs(hedef - last_price)
    rr = round(reward / risk, 2) if risk > 0 else 0
    
    return {
        'fiyat': last_price,
        'gunluk_degisim': ((last_price / df['Close'].iloc[-2]) - 1) * 100,
        'trend': trend,
        'trend_skoru': int(min(max(rr * 20, 10), 95)),
        'atr': atr,
        'hedef': hedef,
        'stop': stop,
        'rr_orani': rr,
        'last_signal': signal,
        'signal_price': last_price,
        'signal_date': df.index[-1]
    }
def coklu_strateji_analizi(df):
    df = df.copy()
    c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']
    detay = {}
    ind = {}

    # --- Bollinger Bantları Hesaplaması ---
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    ind['u_bb'] = sma20 + (2 * std20) # Üst Bant
    ind['l_bb'] = sma20 - (2 * std20) # Alt Bant (Hata buradaydı)
    
    # BB Stratejisi
    detay['Bollinger Breakout'] = "BUY" if c.iloc[-1] > ind['u_bb'].iloc[-1] else "SELL" if c.iloc[-1] < ind['l_bb'].iloc[-1] else "NEUTRAL"

    # 1. EMA Crossover (9/21)
    ema9 = c.ewm(span=9).mean()
    ema21 = c.ewm(span=21).mean()
    detay['EMA Cross'] = "BUY" if ema9.iloc[-1] > ema21.iloc[-1] else "SELL"
    ind['ema9'], ind['ema21'] = ema9, ema21

    # 2. Donchian & 3. Turtle Trading (20 & 55 Günlük)
    ind['u_donchian'] = h.rolling(20).max()
    ind['l_donchian'] = l.rolling(20).min()
    detay['Turtle/Donchian'] = "BUY" if c.iloc[-1] > ind['u_donchian'].iloc[-2] else "SELL" if c.iloc[-1] < ind['l_donchian'].iloc[-2] else "NEUTRAL"

    # 4. ADX + Trend Direction
    plus_dm = h.diff().clip(lower=0)
    minus_dm = l.diff().clip(upper=0).abs()
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    detay['ADX Trend'] = "BUY" if plus_di.iloc[-1] > minus_di.iloc[-1] else "SELL"

    # 5. RSI Filtered (Trend Onaylı)
    rsi = 100 - (100 / (1 + (c.diff().clip(lower=0).rolling(14).mean() / c.diff().clip(upper=0).abs().rolling(14).mean())))
    detay['RSI Filtered'] = "BUY" if rsi.iloc[-1] < 40 and c.iloc[-1] > ema21.iloc[-1] else "SELL" if rsi.iloc[-1] > 60 and c.iloc[-1] < ema21.iloc[-1] else "NEUTRAL"

    # 6. MACD Cross (Trend Filtreli)
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    macd_sig = macd.ewm(span=9).mean()
    detay['MACD Trend'] = "BUY" if macd.iloc[-1] > macd_sig.iloc[-1] and c.iloc[-1] > ema21.iloc[-1] else "SELL"

    # 7. Bollinger & 8. ATR Volatility Breakout
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    ind['u_bb'] = sma20 + (2 * std20)
    detay['Volatility Break'] = "BUY" if c.iloc[-1] > ind['u_bb'].iloc[-1] and (tr.iloc[-1] > atr.iloc[-1] * 1.5) else "NEUTRAL"

    # 9. Support-Resistance Break (Swing High/Low)
    s_high = h.shift(1).rolling(20).max()
    s_low = l.shift(1).rolling(20).min()
    detay['S/R Breakout'] = "BUY" if c.iloc[-1] > s_high.iloc[-1] else "SELL" if c.iloc[-1] < s_low.iloc[-1] else "NEUTRAL"

    # 10. VWAP (Reversion + Trend)
    ind['vwap'] = (v * (h + l + c) / 3).cumsum() / v.cumsum()
    detay['VWAP Strategy'] = "BUY" if c.iloc[-1] > ind['vwap'].iloc[-1] else "SELL"

    # 11. Volume Breakout & 12. OBV Trend
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    detay['Volume/OBV'] = "BUY" if v.iloc[-1] > v.rolling(20).mean().iloc[-1] * 2 and obv.iloc[-1] > obv.rolling(10).mean().iloc[-1] else "NEUTRAL"

    # 13. Z-Score Mean Reversion & 14. Pairs Trading Altyapısı
    z_score = (c - sma20) / std20
    detay['Z-Score/Pairs'] = "BUY" if z_score.iloc[-1] < -2.2 else "SELL" if z_score.iloc[-1] > 2.2 else "NEUTRAL"

    # 15. Trend + Momentum & 16. Teknik + Volatilite Hibrit
    detay['Hybrid Engine'] = "BUY" if (plus_di.iloc[-1] > 25 and rsi.iloc[-1] > 50) else "SELL" if (minus_di.iloc[-1] > 25 and rsi.iloc[-1] < 50) else "NEUTRAL"

    buys = list(detay.values()).count("BUY")
    sells = list(detay.values()).count("SELL")
    final = "🚀 GÜÇLÜ AL" if buys > sells + 4 else "🔼 AL" if buys > sells else "🔽 SAT" if sells > buys else "⏺️ NÖTR"

    return {"final_signal": final, "detay": detay, "indicators": ind}

def hibrit_sinyal_motoru(df):
    df = df.copy()
    c, h, l = df['Close'], df['High'], df['Low']
    
    # --- 1. MACD HESAPLAMA ---
    exp1 = c.ewm(span=12, adjust=False).mean()
    exp2 = c.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=9, adjust=False).mean()
    
    # --- 2. TREND KIRILIMI (SMA20/50 Kesişimi veya Dinamik Kanal) ---
    sma50 = c.rolling(50).mean()
    
    sinyaller = []
    
    # Son 100 mumu tara (Okları çizmek için)
    for i in range(len(df)-100, len(df)):
        # AL KOŞULU: Fiyat SMA50 üstünde kapanacak VE MACD yukarı kesmiş olacak
        if (c.iloc[i] > sma50.iloc[i]) and (macd.iloc[i] > signal_line.iloc[i]) and \
           (c.iloc[i-1] <= sma50.iloc[i-1] or macd.iloc[i-1] <= signal_line.iloc[i-1]):
            sinyaller.append({'date': df.index[i], 'price': l.iloc[i], 'type': 'BUY', 'label': '🚀 HİBRİT AL'})
            
        # SAT KOŞULU: Fiyat SMA50 altına inecek VE MACD aşağı kesmiş olacak
        elif (c.iloc[i] < sma50.iloc[i]) and (macd.iloc[i] < signal_line.iloc[i]) and \
             (c.iloc[i-1] >= sma50.iloc[i-1] or macd.iloc[i-1] >= signal_line.iloc[i-1]):
            sinyaller.append({'date': df.index[i], 'price': h.iloc[i], 'type': 'SELL', 'label': '⚠️ HİBRİT SAT'})
            
    return sinyaller    