import streamlit as st

def apply_custom_css():
    st.markdown(
        """
        <style>
        /* 1. Üst Barı (Header) ve Toolbar'ı Karartma */
        header[data-testid="stHeader"] {
            background-color: #0E1117 !important;
            border-bottom: 1px solid #30363D;
        }
        
        /* Sağ üstteki Deploy/Menü butonlarının olduğu alanı temizleme */
        header[data-testid="stHeader"]::before {
            background-color: #0E1117 !important;
        }

        /* 2. Global Arka Plan */
        .stApp {
            background-color: #0E1117;
            color: #FFFFFF;
        }

        /* 3. Sidebar Düzenlemeleri */
        [data-testid="stSidebar"] {
            background-color: #161B22 !important;
            border-right: 1px solid #30363D;
        }

        /* 4. Kutuların İçindeki Yazıları Net Siyah Yapma */
        /* Selectbox, TextInput, NumberInput ve File Uploader kutuları */
        div[data-baseweb="select"] > div, 
        div[data-baseweb="input"] > div,
        input, 
        .stSelectbox div[role="button"],
        [data-testid="stFileUploadDropzone"] {
            background-color: #FFFFFF !important; 
            color: #000000 !important;             
            border-radius: 8px !important;
            font-weight: 500 !important;
        }

        /* Liste elemanlarının (açılır menü) siyah olması */
        div[data-baseweb="popover"] ul, 
        div[data-baseweb="popover"] li {
            background-color: #FFFFFF !important;
            color: #000000 !important;
        }

        /* 5. Butonlar - Modern Görünüm */
        .stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #9B51E0 0%, #3081ED 100%);
            color: white !important;
            border: none;
            padding: 0.6rem 1rem;
            border-radius: 10px;
            font-weight: 600;
        }
        
        /* 6. Tabs (Sekmeler) */
        .stTabs [data-testid="stTab"][aria-selected="true"] {
            color: #9B51E0 !important;
            border-bottom: 2px solid #9B51E0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )