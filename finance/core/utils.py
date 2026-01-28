import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import argrelextrema # Tepe ve dip tespiti için şart
import warnings

# Çirkin uyarıları gizlemek için (Opsiyonel)
warnings.filterwarnings('ignore')

def kapsamli_teknik_analiz(df):
    """
    720 günlük BUY/SELL sinyallerini tarar. 
    Hedefi görülmüş sinyalleri eler ve en güncel aktif stratejiyi döner.
    """
    if df is None or len(df) < 50:
        return {"durum": "VERİ YETERSİZ", "skor": 0, "fiyat": 0, "notlar": "Veri yetersiz."}

    # --- 1. TEKNİK HESAPLAMALAR ---
    df_tech = df.copy()
    close_ser = df_tech['Close']
    
    # ATR (Oynaklık)
    high_low = df_tech['High'] - df_tech['Low']
    high_close = np.abs(df_tech['High'] - df_tech['Close'].shift())
    low_close = np.abs(df_tech['Low'] - df_tech['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df_tech['ATR'] = np.max(ranges, axis=1).rolling(window=14).mean()
    current_atr = df_tech['ATR'].iloc[-1]
    
    # RSI & Ortalamalar
    delta = close_ser.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df_tech['RSI'] = 100 - (100 / (1 + (gain / loss)))
    rsi_val = df_tech['RSI'].iloc[-1]

    df_tech['SMA20'] = close_ser.rolling(20).mean()
    df_tech['SMA50'] = close_ser.rolling(50).mean()
    df_tech['SMA200'] = close_ser.rolling(200).mean()
    
    hacim_son = df_tech['Volume'].iloc[-1]
    hacim_ort = df_tech['Volume'].tail(360).mean()
    hacim_onayi = hacim_son > (hacim_ort * 1.5)

    # --- 2. ÇİFT YÖNLÜ SİNYAL TARAMASI (720 GÜN) ---
    all_signals = []
    scan_depth = min(len(df_tech), 720)
    df_scan = df_tech.tail(scan_depth)
    
    for i in range(1, len(df_scan)):
        c_close, p_close = df_scan['Close'].iloc[i], df_scan['Close'].iloc[i-1]
        c_sma50, p_sma50 = df_scan['SMA50'].iloc[i], df_scan['SMA50'].iloc[i-1]
        
        if c_close > c_sma50 and p_close <= p_sma50:
            all_signals.append({"type": "BUY", "date": df_scan.index[i], "price": float(c_close), "low": float(df_scan['Low'].iloc[i]), "high": float(df_scan['High'].iloc[i])})
        elif c_close < c_sma50 and p_close >= p_sma50:
            all_signals.append({"type": "SELL", "date": df_scan.index[i], "price": float(c_close), "low": float(df_scan['Low'].iloc[i]), "high": float(df_scan['High'].iloc[i])})

    # --- 3. EN GÜNCEL SİNYAL VE YÖN TAYİNİ ---
    if all_signals:
        last_sig = all_signals[-1]
        s_type, signal_date, signal_price = last_sig['type'], last_sig['date'], last_sig['price']
        sig_low, sig_high = last_sig['low'], last_sig['high']
    else:
        s_type, signal_date, signal_price = "NEUTRAL", df_tech.index[-1], float(close_ser.iloc[-1])
        sig_low, sig_high = float(df_tech['Low'].iloc[-1]), float(df_tech['High'].iloc[-1])

    # --- 4. STRATEJİK SEVİYELER ---
    if s_type == "BUY":
        en_guclu_alis = sig_low + (current_atr * 0.2)
        stop_loss = sig_low - (current_atr * 1.5)
        hedef_fiyat = signal_price + (current_atr * 4.5)
    elif s_type == "SELL":
        en_guclu_alis = sig_high - (current_atr * 0.2)
        stop_loss = sig_high + (current_atr * 1.5)
        hedef_fiyat = signal_price - (current_atr * 4.5)
    else:
        en_guclu_alis, stop_loss, hedef_fiyat = signal_price, signal_price * 0.95, signal_price * 1.05

    # --- 5. HEDEF KONTROLÜ (FİLTRELEME) ---
    current_price = float(close_ser.iloc[-1])
    sinyal_gecerli = True
    analiz_notlari = []

    if s_type == "BUY" and current_price >= hedef_fiyat:
        sinyal_gecerli = False
        analiz_notlari.append("Hedef fiyat zaten geçildi.")
    elif s_type == "SELL" and current_price <= hedef_fiyat:
        sinyal_gecerli = False
        analiz_notlari.append("Düşüş hedefi zaten görüldü.")

    # --- 6. PUANLAMA VE SONUÇ ---
    skor = 0
    if 40 < rsi_val < 60: skor += 2; analiz_notlari.append("RSI İdeal")
    if current_price > df_tech['SMA50'].iloc[-1]: skor += 1
    if hacim_onayi: skor += 1; analiz_notlari.append("Hacim Onayı")

    if not sinyal_gecerli:
        durum = "NÖTR / HEDEF GÖRÜLDÜ"
        skor = 0
    else:
        durum = ("🚀 GÜÇLÜ " if skor >= 4 else "") + ("AL" if s_type == "BUY" else "SAT" if s_type == "SELL" else "İZLE")

    return {
        "fiyat": current_price, "rsi": rsi_val, "skor": skor, "durum": durum,
        "signal_type": s_type if sinyal_gecerli else "NEUTRAL",
        "all_signals": all_signals, "signal_date": signal_date, "signal_price": signal_price,
        "hedef": hedef_fiyat, "stop": stop_loss, "en_guclu_alis": en_guclu_alis,
        "kazanc_beklentisi": abs(((hedef_fiyat / current_price) - 1) * 100) if sinyal_gecerli else 0,
        "rr_oran": abs((hedef_fiyat - current_price) / (current_price - stop_loss)) if current_price != stop_loss else 0,
        "notlar": ", ".join(analiz_notlari) if analiz_notlari else "Göstergeler stabil.",
        "df": df_tech
    }


def piyasa_radari_tara(sembol_listesi, ui_names):
    """
    Verilen listeyi tarar. Hem güçlü AL hem de güçlü SAT sinyallerini 
    filtreleyerek analiz sonuçlarını döner.
    """
    sonuclar = []
    if not sembol_listesi:
        return []

    # Veri indirme derinliği
    try:
        # Progress bar'ı kapattık ki arkada temiz çalışsın
        data_all = yf.download(sembol_listesi, period="1y", interval="1d", group_by='ticker', progress=False)
    except Exception:
        return []
    
    for symbol in sembol_listesi:
        try:
            # Multi-index veri kontrolü (Tek sembol vs Çok sembol durumu)
            if len(sembol_listesi) > 1:
                df = data_all[symbol].dropna()
            else:
                df = data_all.dropna()

            if df.empty or len(df) < 50: # En az 50 mum (SMA200 için veri lazım olsa da 50 makul)
                continue
            
            # Senin yazdığın o meşhur kapsamlı teknik analizi çalıştır
            analiz = kapsamli_teknik_analiz(df)
            
            # UI İsimlendirmesi
            analiz['display_name'] = ui_names.get(symbol, symbol)
            analiz['symbol'] = symbol
            
            # --- ÇİFT YÖNLÜ FİLTRELEME MANTIĞI ---
            # 1. Alım Fırsatları: Skor 3 ve üzeri
            # 2. Satış Riskleri: Skor 1 ve altı (Güçlü Sat sinyalleri)
            if analiz['skor'] >= 2 or analiz['skor'] <= -2: 
                sonuclar.append(analiz)
                
        except Exception:
            continue
            
    # Sıralama: Skorlara göre (4-5 başa, 0-1 sona)
    # Bu sayede radarda en üstte en güçlü "AL"lar, en altta en güçlü "SAT"lar olur
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