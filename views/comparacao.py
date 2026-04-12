"""
Página de comparação de cenários — ex: sem ACP vs. com ACP forte.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from model import DeforestationModel
from viz import plot_comparison, plot_grid_map


st.title("Comparação de Cenários")
st.caption("Compare dois cenários lado a lado para avaliar o efeito da ACP ambiental.")


# ── Cache ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Rodando cenário...")
def run_scenario(params_tuple):
    params = dict(params_tuple)
    model = DeforestationModel(params)
    model.run()
    return model.history


# ── Cenários pré-definidos ──────────────────────────────────────────
PRESETS = {
    "Personalizado": {},
    "Sem ACP (apenas IBAMA)": {
        "P_inq": 0.0, "P_acp": 0.0, "P_cond": 0.0, "P_exec": 0.0, "V_ACP": 0.0,
    },
    "ACP moderada (status quo)": {
        "P_inq": 0.08, "P_acp": 0.40, "P_cond": 0.65, "P_exec": 0.30,
        "V_ACP": 15000.0, "t_acp": 7.0,
    },
    "ACP forte (MP fortalecido)": {
        "P_inq": 0.20, "P_acp": 0.60, "P_cond": 0.80, "P_exec": 0.50,
        "V_ACP": 50000.0, "t_acp": 4.0,
    },
    "Fiscalização total fortalecida": {
        "Pd": 0.80, "Pa": 0.50,
        "P_inq": 0.20, "P_acp": 0.60, "P_cond": 0.80, "P_exec": 0.50,
        "V_ACP": 50000.0, "t_acp": 4.0,
    },
}

# ── Configuração compartilhada ──────────────────────────────────────
with st.sidebar:
    st.header("Configuração compartilhada")
    grid_size = st.slider("Grid", 50, 150, 80, step=10, key="comp_grid")
    n_properties = st.slider("Nº propriedades", 10, 80, 30, key="comp_props")
    n_years = st.slider("Anos simulados", 5, 30, 15, key="comp_years")
    seed = st.number_input("Seed", value=42, step=1, key="comp_seed")

base_params = {
    "grid_size": grid_size,
    "n_properties": n_properties,
    "n_years": n_years,
    "seed": int(seed),
}

# ── Seleção de cenários ─────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Cenário A")
    preset_a = st.selectbox("Preset", list(PRESETS.keys()), index=1, key="preset_a")

    params_a = dict(base_params)
    if preset_a != "Personalizado":
        params_a.update(PRESETS[preset_a])
    else:
        with st.expander("Parâmetros Cenário A"):
            params_a["Pd"] = st.slider("Pd", 0.0, 1.0, 0.55, 0.05, key="a_Pd")
            params_a["Pa"] = st.slider("Pa", 0.0, 1.0, 0.30, 0.05, key="a_Pa")
            params_a["P_inq"] = st.slider("P_inq", 0.0, 0.30, 0.08, 0.01, key="a_Pinq")
            params_a["P_acp"] = st.slider("P_acp", 0.0, 1.0, 0.40, 0.05, key="a_Pacp")
            params_a["P_cond"] = st.slider("P_cond", 0.0, 1.0, 0.65, 0.05, key="a_Pcond")
            params_a["P_exec"] = st.slider("P_exec", 0.0, 1.0, 0.30, 0.05, key="a_Pexec")
            params_a["V_ACP"] = float(st.slider("V_ACP", 0, 100000, 15000, 1000, key="a_VACP"))
            params_a["t_acp"] = float(st.slider("t_acp", 2, 15, 7, key="a_tacp"))

with col_b:
    st.subheader("Cenário B")
    preset_b = st.selectbox("Preset", list(PRESETS.keys()), index=3, key="preset_b")

    params_b = dict(base_params)
    if preset_b != "Personalizado":
        params_b.update(PRESETS[preset_b])
    else:
        with st.expander("Parâmetros Cenário B"):
            params_b["Pd"] = st.slider("Pd", 0.0, 1.0, 0.55, 0.05, key="b_Pd")
            params_b["Pa"] = st.slider("Pa", 0.0, 1.0, 0.30, 0.05, key="b_Pa")
            params_b["P_inq"] = st.slider("P_inq", 0.0, 0.30, 0.08, 0.01, key="b_Pinq")
            params_b["P_acp"] = st.slider("P_acp", 0.0, 1.0, 0.40, 0.05, key="b_Pacp")
            params_b["P_cond"] = st.slider("P_cond", 0.0, 1.0, 0.65, 0.05, key="b_Pcond")
            params_b["P_exec"] = st.slider("P_exec", 0.0, 1.0, 0.30, 0.05, key="b_Pexec")
            params_b["V_ACP"] = float(st.slider("V_ACP", 0, 100000, 15000, 1000, key="b_VACP"))
            params_b["t_acp"] = float(st.slider("t_acp", 2, 15, 7, key="b_tacp"))


# ── Rodar simulações ────────────────────────────────────────────────
tuple_a = tuple(sorted(params_a.items()))
tuple_b = tuple(sorted(params_b.items()))

history_a = run_scenario(tuple_a)
history_b = run_scenario(tuple_b)

if not history_a or not history_b:
    st.error("Erro na simulação.")
    st.stop()

# ── Resultados comparativos ─────────────────────────────────────────
st.markdown("---")

# Métricas finais
final_a = history_a[-1]
final_b = history_b[-1]

st.subheader("Resultado Final")
mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.metric("Floresta (A)", f"{final_a['pct_forest']:.1f}%")
    st.metric("Floresta (B)", f"{final_b['pct_forest']:.1f}%",
              delta=f"{final_b['pct_forest'] - final_a['pct_forest']:+.1f}%")
with mc2:
    total_desm_a = sum(h["annual_converted"] for h in history_a)
    total_desm_b = sum(h["annual_converted"] for h in history_b)
    st.metric("Total desmatado (A)", f"{total_desm_a} ha")
    st.metric("Total desmatado (B)", f"{total_desm_b} ha",
              delta=f"{total_desm_b - total_desm_a:+d} ha",
              delta_color="inverse")
with mc3:
    total_acps_a = sum(h["annual_acps"] for h in history_a)
    total_acps_b = sum(h["annual_acps"] for h in history_b)
    st.metric("ACPs (A)", total_acps_a)
    st.metric("ACPs (B)", total_acps_b)
with mc4:
    st.metric("C final (A)", f"R$ {final_a['C']:,.0f}/ha")
    st.metric("C final (B)", f"R$ {final_b['C']:,.0f}/ha")

# Gráfico comparativo
st.subheader("Comparação Temporal")
label_a = preset_a if preset_a != "Personalizado" else "Cenário A"
label_b = preset_b if preset_b != "Personalizado" else "Cenário B"
fig_comp = plot_comparison(history_a, history_b, label_a, label_b, figsize=(14, 8))
st.pyplot(fig_comp, use_container_width=True)

# Mapas finais lado a lado
st.subheader("Mapas Finais")
map_a, map_b = st.columns(2)
with map_a:
    st.caption(f"**{label_a}** — Ano {final_a['year']}")
    fig_ma = plot_grid_map(final_a["grid"], final_a.get("boundaries"), figsize=(6, 6))
    st.pyplot(fig_ma, use_container_width=True)
with map_b:
    st.caption(f"**{label_b}** — Ano {final_b['year']}")
    fig_mb = plot_grid_map(final_b["grid"], final_b.get("boundaries"), figsize=(6, 6))
    st.pyplot(fig_mb, use_container_width=True)

# Tabela comparativa
with st.expander("Tabela comparativa detalhada"):
    rows = []
    for i in range(max(len(history_a), len(history_b))):
        row = {"Ano": i}
        if i < len(history_a):
            row["Floresta A (%)"] = round(history_a[i]["pct_forest"], 1)
            row["Desmatado A (ha)"] = history_a[i]["annual_converted"]
            row["ACPs A"] = history_a[i]["annual_acps"]
            row["C A (R$/ha)"] = round(history_a[i]["C"], 0)
        if i < len(history_b):
            row["Floresta B (%)"] = round(history_b[i]["pct_forest"], 1)
            row["Desmatado B (ha)"] = history_b[i]["annual_converted"]
            row["ACPs B"] = history_b[i]["annual_acps"]
            row["C B (R$/ha)"] = round(history_b[i]["C"], 0)
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
