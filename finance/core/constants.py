"""
Uygulamada kullanılan global constants ve state key'leri
"""

# === SESSION STATE KEY'LERİ ===
# Tab Navigation
SELECTED_TAB_KEY = "selected_tab"
SWITCH_TO_TAB_KEY = "_switch_to_tab"

# Radar Bridge
SELECTED_SYMBOL_RADAR_KEY = "selected_symbol_radar"
SELECTED_MARKET_RADAR_KEY = "selected_market_radar"

# Analysis State
TAHMIN_SONUCU_KEY = "tahmin_sonucu"
TAHMIN_YORUMU_KEY = "tahmin_yorumu"
STRATEJI_GRAFIGI_KEY = "strateji_grafigi"
STRATEJI_YORUMU_KEY = "strateji_yorumu"
SECILEN_SEMBOL_KEY = "secilen_sembol"

# Chart Data Cache
CHART_DATA_KEY = "chart_data"
LAST_SYMBOL_KEY = "last_symbol"

# Radar Cache
RADAR_CACHE_KEY = "radar_cache"

# === TAB NAMES ===
TAB_ANALIZ_PANELI = "📈 Analiz Paneli"
TAB_YATIRIM_RADARI = "🎯 Yatırım Radarı"
TAB_NAMES = [TAB_ANALIZ_PANELI, TAB_YATIRIM_RADARI]

# === MARKET TYPES ===
MARKET_BIST100 = "BIST 100"
MARKET_KRIPTO = "Kripto Paralar"
MARKET_EMTIA = "Emtialar (Maden/Enerji)"
MARKET_USA = "ABD Hisseleri"
MARKET_OPTIONS = [MARKET_BIST100, MARKET_KRIPTO, MARKET_EMTIA, MARKET_USA]

# === RADAR MARKET MAPPING ===
RADAR_MARKET_MAP = {
    "🇹🇷 BIST 100": MARKET_BIST100,
    "₿ Kripto": MARKET_KRIPTO,
    "🏗️ Emtia": MARKET_EMTIA,
    "🇺🇸 ABD Hisseleri": MARKET_USA
}
