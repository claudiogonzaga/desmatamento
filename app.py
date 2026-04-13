"""
ABM de Desmatamento Ilegal e Dissuasão via ACP Ambiental
Entrypoint Streamlit multipágina.
"""

import streamlit as st

st.set_page_config(
    page_title="ABM Desmatamento & ACP",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS global — tema escuro
st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: #e0e0e0;
        font-family: Georgia, serif;
    }
    .stSidebar {
        background-color: #1a1a1a;
    }
    h1, h2, h3, h4 {
        font-family: Georgia, serif;
        color: #e0e0e0;
    }
    .stSlider label, .stSlider p, .stSlider div {
        font-family: Georgia, serif;
        color: #e0e0e0 !important;
    }
    .stSidebar label, .stSidebar p, .stSidebar span,
    .stSidebar .stMarkdown, .stSidebar [data-testid="stWidgetLabel"] {
        color: #e0e0e0 !important;
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4 {
        color: #f0f0f0 !important;
    }
    .stNumberInput label {
        color: #e0e0e0 !important;
    }
    /* Métricas st.metric — forçar texto visível */
    [data-testid="stMetricLabel"] {
        color: #cccccc !important;
    }
    [data-testid="stMetricLabel"] label,
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] div {
        color: #cccccc !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    [data-testid="stMetricValue"] div {
        color: #ffffff !important;
    }
    [data-testid="stMetricDelta"] {
        color: inherit !important;
    }
    /* Fallback: qualquer texto dentro de metric */
    [data-testid="metric-container"] {
        color: #ffffff !important;
    }
    [data-testid="metric-container"] label {
        color: #cccccc !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 1.8rem !important;
    }
</style>
""", unsafe_allow_html=True)


# Navegação multipágina
home_page = st.Page("views/home.py", title="Simulação", icon="🗺️", default=True)
comparacao_page = st.Page("views/comparacao.py", title="Comparação", icon="⚖️")
resumo_page = st.Page("views/resumo.py", title="Resumo", icon="📊")

pg = st.navigation([home_page, comparacao_page, resumo_page])
pg.run()
