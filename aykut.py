# CSS: Scroll & Butonlar
import streamlit as st
from PIL import Image

# Sayfa ayarları
st.set_page_config(page_title="Aykut Kutlu | Personal Website", layout="wide")

# CSS stilleri
st.markdown("""
    <style>
    html {
        scroll-behavior: smooth;
    }
    .section {
        padding-top: 60px;
    }
    h2 {
        border-bottom: 2px solid #eee;
        padding-bottom: 10px;
        margin-top: 60px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar: Profil fotoğrafı ve nav butonları
with st.sidebar:
    cols = st.columns([1, 2, 1])
    with cols[1]:
        st.image("Aykut Kutlu.jpg", width=180)

    # Altında yazılar
    st.markdown(
        """
        <div style="text-align: center; margin-top: 10px;">
            <p style="margin: 0; font-size: 20px; font-weight: bold;">Aykut Kutlu</p>
            <p style="margin: 0; font-size: 16px; color: gray;">Data Analyst</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("### 📂 Navigation")
    st.markdown("""
    - [About Me](#about-me)
    - [Experience](#experience)
    - [Education](#education)
    - [Projects](#projects)
    - [Contact](#contact)
    """, unsafe_allow_html=True)

# Başlık
st.title("👋 Hi, I'm Aykut Kutlu")


# About Me
st.markdown("""<div class="section" id="about-me"></div>""", unsafe_allow_html=True)
st.header("🧑‍💻 About Me")
st.markdown("""
I’m a 31-year-old data enthusiast with a strong foundation in economics and a passion for leveraging data to drive insights and strategic decision-making. 
My expertise lies in data science, financial analysis, and algorithmic trading, where I apply advanced analytical techniques to real-world challenges. 
I actively develop and deploy end-to-end analytical applications and forecasting models using Python, R, Streamlit, and Shiny. 
I also build mobile and backend solutions with Flutter and FastAPI, bridging data with accessible, interactive tools.
My work blends technical precision with a deep curiosity for patterns, performance, and practical impact.

I hold a Bachelor’s degree in Economics from Hacettepe University, where I built a strong foundation in statistical thinking and economic modeling. 
Currently, I serve as a Data Analysis Specialist at the Turkish Red Crescent (KIZILAYKART), where I focus on projects involving Minimum Expenditure Basket calculations, vulnerability profiling, and time series forecasting to support humanitarian operations and strategic planning.

In my daily work, I use tools like Python, R, SPSS, and Excel to build scalable workflows, perform robust data analysis, and produce automated reporting solutions. 
My diverse project portfolio includes forecasting platforms, technical trading systems, and interactive dashboards designed to deliver insights to both technical and non-technical stakeholders.

Beyond my technical capabilities, my passion for theater contributes to my creativity and communication skills, helping me think from multiple perspectives and collaborate effectively within multidisciplinary teams. 
I’m constantly learning and striving to turn complex data into meaningful, impactful outcomes.
""")

# Experience
st.markdown("""<div class="section" id="experience"></div>""", unsafe_allow_html=True)
st.header("💼 Experience")
st.subheader("Data Analysis Specialist – Turkish Red Crescent KIZILAYKART (04/2023 – Present)")
st.markdown("""
- Conduct in-depth data analysis and generate comprehensive reports to support data-driven decision-making processes.  
- Perform Minimum Expenditure Basket (MEB) calculations to assess costs of essential goods.  
- Carry out vulnerability profiling and targeting studies using statistical methods.  
- Apply time series forecasting, regression, and decision tree models.  
- Work with Python, R, SPSS; automate data pipelines and reporting tools.  
- Collaborate with interdisciplinary teams for effective humanitarian planning.  
""")

# Education
st.markdown("""<div class="section" id="education"></div>""", unsafe_allow_html=True)
st.header("🎓 Education")
st.markdown("""
- **Hacettepe University** – B.A. in Economics (09/2018 – 06/2022) | GPA: 2.84/4.00  
- **Mehmet Emin Resulzade Anatolian High School** (09/2008 – 06/2011)  
""")

# Projects
st.markdown("""<div class="section" id="projects"></div>""", unsafe_allow_html=True)
st.header("📊 Projects")
st.markdown("""
- 📈 **[Financial Technical Analysis App](https://financial-analysis1.streamlit.app/):** Forecasting with ARIMA, XGBoost, LSTM models  
- 🐢 **[Basic Analysis App](https://anewliz-app.streamlit.app/):** Technical analysis  
- 🛒 **[Basic Financial Analysis](https://aykut.shinyapps.io/finansal_ts/):** Financial Analysis with Basic Time Series  
- 📊 **[MEB Calculation Tool](https://aykut.shinyapps.io/MEB_Turk_Kizilay/):** MEB Calculation with TURKSTAT Data  
""")

# Contact
st.markdown("""<div class="section" id="contact"></div>""", unsafe_allow_html=True)
st.header("📫 Contact")
st.markdown("""
- 📧 Email: aykutkutlu811@gmail.com  
- 💼 LinkedIn: [linkedin.com/in/aykutkutlu](https://www.linkedin.com/in/aykut-kutlu-562867236/)  
- 🐙 GitHub: [github.com/simple1one](https://github.com/simple1one)  
""")

# Footer
st.markdown("---")
st.markdown("© 2025 Aykut Kutlu - Personal Website")