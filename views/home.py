"""
Página principal de simulação — sliders + 3 painéis visuais + botão Play.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import streamlit as st
import numpy as np
from model import DeforestationModel
from viz import plot_grid_map, plot_trajectory, plot_metrics_bar


st.title("Simulação — Desmatamento e Dissuasão via ACP")
st.caption(
    "Modelo baseado em agentes: teoria econômica do crime (Becker, 1968) "
    "e modelo de dissuasão de Schmitt (2015), adaptado para a via judicial (ACP)."
)

# ── Sidebar: parâmetros ─────────────────────────────────────────────
with st.sidebar:
    st.header("Parâmetros do Modelo")

    st.subheader("Configuração geral")
    n_properties = st.slider("Nº propriedades", 10, 80, 40)
    grid_size = st.slider("Grid (NxN)", 50, 150, 100, step=10)
    n_years = st.slider("Anos simulados", 5, 30, 15)
    seed = st.number_input("Seed (reprodutibilidade)", value=42, step=1)

    st.subheader("Fiscalização administrativa")
    Pd = st.slider("Prob. detecção (Pd)", 0.0, 1.0, 0.55, 0.05)
    Pa = st.slider("Prob. autuação (Pa)", 0.0, 1.0, 0.30, 0.05)

    st.subheader("ACP ambiental")
    P_inq = st.slider("Prob. inquérito MP (P_inq)", 0.0, 0.30, 0.08, 0.01)
    P_acp = st.slider("Prob. ACP ajuizada (P_acp)", 0.0, 1.0, 0.40, 0.05)
    P_cond = st.slider("Prob. condenação (P_cond)", 0.0, 1.0, 0.65, 0.05)
    P_exec = st.slider("Prob. execução (P_exec)", 0.0, 1.0, 0.30, 0.05)
    V_ACP = st.slider("Valor ACP (R$/ha)", 1_000, 100_000, 15_000, 1_000)
    t_acp = st.slider("Tempo ACP (anos)", 2, 15, 7)

    st.subheader("Economia")
    Gp = st.slider("Preço pecuária (R$/ha/ano)", 50, 1_500, 300, 25)
    Ga = st.slider("Preço soja (R$/ha/ano)", 100, 2_000, 500, 25)
    Gt = st.slider("Valor terra desmatada (R$/ha)", 1_000, 15_000, 5_000, 500)


# ── Cache da simulação ──────────────────────────────────────────────
@st.cache_data(show_spinner="Rodando simulação...")
def run_simulation(params_tuple):
    params = dict(params_tuple)
    model = DeforestationModel(params)
    model.run()
    return model.history


params = {
    "n_properties": n_properties,
    "grid_size": grid_size,
    "n_years": n_years,
    "seed": int(seed),
    "Pd": Pd, "Pa": Pa,
    "P_inq": P_inq, "P_acp": P_acp, "P_cond": P_cond, "P_exec": P_exec,
    "V_ACP": float(V_ACP), "t_acp": float(t_acp),
    "Gp": float(Gp), "Ga": float(Ga), "Gt": float(Gt),
}

params_tuple = tuple(sorted(params.items()))
history = run_simulation(params_tuple)

if not history:
    st.error("A simulação não gerou resultados.")
    st.stop()

# ── Controles: slider de ano + botão Play ───────────────────────────
max_year = len(history) - 1

# Inicializar estado do play
if "playing" not in st.session_state:
    st.session_state.playing = False
if "play_year" not in st.session_state:
    st.session_state.play_year = 0

ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 6])
with ctrl_col1:
    play_btn = st.button("▶ Play", use_container_width=True)
with ctrl_col2:
    stop_btn = st.button("⏹ Stop", use_container_width=True)

if play_btn:
    st.session_state.playing = True
    st.session_state.play_year = 0
if stop_btn:
    st.session_state.playing = False

# Se play está ativo, auto-avançar
if st.session_state.playing:
    selected_year = st.session_state.play_year
    if selected_year >= max_year:
        st.session_state.playing = False
        selected_year = max_year
else:
    selected_year = st.slider(
        "Ano da simulação", 0, max_year, max_year,
        help="Navegue pela evolução do município ao longo do tempo."
    )

snapshot = history[selected_year]

# ── Métricas rápidas ────────────────────────────────────────────────
st.markdown(f"### Ano {snapshot['year']} de {max_year}")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Floresta", f"{snapshot['pct_forest']:.1f}%",
              delta=f"{snapshot['pct_forest'] - history[0]['pct_forest']:.1f}pp")
with col2:
    st.metric("Pastagem", f"{snapshot['pct_pasture']:.1f}%")
with col3:
    st.metric("Soja", f"{snapshot['pct_soy']:.1f}%")
with col4:
    st.metric("Desmatado no ano", f"{snapshot['annual_converted']} ha")
with col5:
    C = snapshot["C"]
    st.metric("C (comportamento)", f"R$ {C:,.0f}/ha",
              delta="Compensa" if C > 0 else "Não compensa",
              delta_color="inverse")

# ── Indicadores de dissuasão (legíveis) ─────────────────────────────
d1, d2, d3, d4 = st.columns(4)
with d1:
    st.markdown(f"**VD admin:** R$ {snapshot['VD_admin']:,.2f}/ha")
with d2:
    st.markdown(f"**VD ACP:** R$ {snapshot['VD_acp']:,.2f}/ha")
with d3:
    st.markdown(f"**VD total:** R$ {snapshot['VD_total']:,.2f}/ha")
with d4:
    st.markdown(f"**VE:** R$ {snapshot['VE']:,.0f}/ha")

# ── 3 Painéis ───────────────────────────────────────────────────────
st.markdown("---")

# Painel 1: Mapa
st.subheader("Mapa do Município")
map_placeholder = st.empty()
fig_map = plot_grid_map(
    snapshot["grid"],
    boundaries=snapshot.get("boundaries"),
    figsize=(10, 10),
    dpi=150,
)
map_placeholder.pyplot(fig_map, use_container_width=True)

# Painel 2: Trajetória
st.subheader("Trajetória Temporal")
fig_traj = plot_trajectory(history, figsize=(12, 6))
st.pyplot(fig_traj, use_container_width=True)

# Painel 3: Métricas do ano
st.subheader("Métricas Detalhadas")
fig_bar = plot_metrics_bar(snapshot, figsize=(12, 4.5))
st.pyplot(fig_bar, use_container_width=True)

# ── Tabela de fiscalização ──────────────────────────────────────────
with st.expander("Dados de fiscalização por ano"):
    import pandas as pd
    rows = []
    for h in history:
        rows.append({
            "Ano": h["year"],
            "Desmatado (ha)": h["annual_converted"],
            "Autuações": h["annual_infractions"],
            "Embargos": h["annual_embargoes"],
            "ACPs": h["annual_acps"],
            "VD admin (R$/ha)": round(h["VD_admin"], 2),
            "VD ACP (R$/ha)": round(h["VD_acp"], 2),
            "VD total (R$/ha)": round(h["VD_total"], 2),
            "VE (R$/ha)": round(h["VE"], 2),
            "C (R$/ha)": round(h["C"], 2),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

# ── Lógica de auto-play (rerun) ─────────────────────────────────────
if st.session_state.playing and selected_year < max_year:
    time.sleep(0.8)
    st.session_state.play_year = selected_year + 1
    st.rerun()
