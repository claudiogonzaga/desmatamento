"""
Página de resumo analítico — parâmetros de Schmitt, cálculos detalhados.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
from math import exp
from model import DeforestationModel


st.title("Resumo Analítico")
st.caption(
    "Cálculo detalhado da dissuasão administrativa (Schmitt, 2015) "
    "e da dissuasão via ACP ambiental."
)

# ── Parâmetros padrão do modelo ─────────────────────────────────────
P = DeforestationModel.DEFAULT_PARAMS

st.markdown("---")

# ── 1. Via administrativa ───────────────────────────────────────────
st.header("1. Dissuasão Administrativa (Schmitt, 2015)")

st.markdown("""
**Fórmula:**
$$VD_{admin} = P_d \\times P_a \\times P_j \\times P_c \\times P_p \\times (S + V_e + V_a) \\times e^{-r \\times t}$$
""")

admin_data = {
    "Parâmetro": [
        "Pd — Prob. detecção (DETER/PRODES)",
        "Pa — Prob. autuação",
        "Pj — Prob. julgamento 1ª instância",
        "Pc — Prob. confirmação da autuação",
        "Pp — Prob. pagamento da multa",
        "S — Multa (R$/ha)",
        "Ve — Valor embargo (lucro cessante)",
        "Va — Valor bens apreendidos (R$)",
        "r — Taxa Selic",
        "t — Tempo médio julgamento (anos)",
    ],
    "Schmitt (2015)": [
        "0,45", "0,24", "0,26", "0,90", "0,10",
        "R$ 5.000", "R$ 200/ha/ano", "R$ 15.185,22",
        "0,10", "2,90",
    ],
    "Atualizado (2024–25)": [
        f"{P['Pd']:.2f}", f"{P['Pa']:.2f}", f"{P['Pj']:.2f}",
        f"{P['Pc']:.2f}", f"{P['Pp']:.2f}",
        f"R$ {P['S']:,.0f}", f"R$ {P['Ve']:,.0f}/ha/ano",
        f"R$ {P['Va']:,.2f}",
        f"{P['r']:.4f}", f"{P['t_admin']:.2f}",
    ],
}
st.dataframe(pd.DataFrame(admin_data), use_container_width=True, hide_index=True)

# Cálculo
prob_chain_admin = P["Pd"] * P["Pa"] * P["Pj"] * P["Pc"] * P["Pp"]
sanction_value = P["S"] + P["Ve"] + P["Va"]
discount_admin = exp(-P["r"] * P["t_admin"])
VD_admin = prob_chain_admin * sanction_value * discount_admin

st.markdown("**Cálculo passo a passo:**")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    - Cadeia de probabilidades: {P['Pd']:.2f} × {P['Pa']:.2f} × {P['Pj']:.2f} × {P['Pc']:.2f} × {P['Pp']:.2f} = **{prob_chain_admin:.6f}**
    - Valor da sanção: R$ {P['S']:,.0f} + R$ {P['Ve']:,.0f} + R$ {P['Va']:,.2f} = **R$ {sanction_value:,.2f}**
    """)
with col2:
    st.markdown(f"""
    - Fator desconto: e^(-{P['r']:.4f} × {P['t_admin']:.2f}) = **{discount_admin:.4f}**
    - **VD_admin = R$ {VD_admin:,.2f}/ha**
    """)

st.info(f"**Resultado:** A dissuasão administrativa é de **R$ {VD_admin:,.2f} por hectare**.")

# ── 2. Via judicial (ACP) ───────────────────────────────────────────
st.markdown("---")
st.header("2. Dissuasão via ACP Ambiental (modelo proposto)")

st.markdown("""
**Fórmula:**
$$VD_{ACP} = P_d \\times P_{inq} \\times P_{acp} \\times P_{cond} \\times P_{exec} \\times V_{ACP} \\times e^{-r \\times t_{ACP}}$$
""")

acp_data = {
    "Parâmetro": [
        "Pd — Prob. detecção",
        "P_inq — Prob. inquérito civil (MP)",
        "P_acp — Prob. ACP ajuizada",
        "P_cond — Prob. condenação judicial",
        "P_exec — Prob. execução da sentença",
        "V_ACP — Valor demandado (R$/ha)",
        "r — Taxa Selic",
        "t_ACP — Tempo até execução (anos)",
    ],
    "Estimativa": [
        f"{P['Pd']:.2f}",
        f"{P['P_inq']:.2f}",
        f"{P['P_acp']:.2f}",
        f"{P['P_cond']:.2f}",
        f"{P['P_exec']:.2f}",
        f"R$ {P['V_ACP']:,.0f}",
        f"{P['r']:.4f}",
        f"{P['t_acp']:.1f}",
    ],
    "Faixa plausível": [
        "0,45–0,80",
        "0,05–0,15",
        "0,30–0,60",
        "0,60–0,80",
        "0,20–0,50",
        "R$ 1.000–100.000",
        "0,10–0,15",
        "5–10",
    ],
}
st.dataframe(pd.DataFrame(acp_data), use_container_width=True, hide_index=True)

prob_chain_acp = P["Pd"] * P["P_inq"] * P["P_acp"] * P["P_cond"] * P["P_exec"]
discount_acp = exp(-P["r"] * P["t_acp"])
VD_acp = prob_chain_acp * P["V_ACP"] * discount_acp

st.markdown("**Cálculo passo a passo:**")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    - Cadeia de probabilidades: {P['Pd']:.2f} × {P['P_inq']:.2f} × {P['P_acp']:.2f} × {P['P_cond']:.2f} × {P['P_exec']:.2f} = **{prob_chain_acp:.6f}**
    - Valor ACP: **R$ {P['V_ACP']:,.0f}/ha**
    """)
with col2:
    st.markdown(f"""
    - Fator desconto: e^(-{P['r']:.4f} × {P['t_acp']:.1f}) = **{discount_acp:.4f}**
    - **VD_ACP = R$ {VD_acp:,.2f}/ha**
    """)

st.info(f"**Resultado:** A dissuasão via ACP é de **R$ {VD_acp:,.2f} por hectare**.")

# ── 3. Dissuasão total ──────────────────────────────────────────────
st.markdown("---")
st.header("3. Dissuasão Total")

VD_total = VD_admin + VD_acp

st.markdown(f"""
$$VD_{{total}} = VD_{{admin}} + VD_{{ACP}} = R\\$ {VD_admin:,.2f} + R\\$ {VD_acp:,.2f} = \\mathbf{{R\\$ {VD_total:,.2f}/ha}}$$
""")

# ── 4. Vantagem econômica ───────────────────────────────────────────
st.markdown("---")
st.header("4. Vantagem Econômica do Desmatamento (VE)")

st.markdown("""
**Fórmula:**
$$VE = G_f + (G_{uso} \\times C_p) + G_t$$
""")

ve_data = {
    "Parâmetro": [
        "Gf — Ganho exploração florestal ilegal (R$/ha)",
        "Gp — Ganho pecuária (R$/ha/ano)",
        "Ga — Ganho soja (R$/ha/ano)",
        "Gt — Valorização da terra (R$/ha)",
        "Cp — Coeficiente de prescrição (anos)",
        "c — Custo do desmatamento (R$/ha)",
    ],
    "Schmitt (2015)": [
        "R$ 2.000", "R$ 200", "R$ 700", "R$ 4.000", "5", "R$ 200",
    ],
    "Atualizado": [
        f"R$ {P['Gf']:,.0f}",
        f"R$ {P['Gp']:,.0f}",
        f"R$ {P['Ga']:,.0f}",
        f"R$ {P['Gt']:,.0f}",
        f"{P['Cp']}",
        f"R$ {P['c_desmat']:,.0f}",
    ],
}
st.dataframe(pd.DataFrame(ve_data), use_container_width=True, hide_index=True)

VE_pec = P["Gf"] + (P["Gp"] * P["Cp"]) + P["Gt"]
VE_soja = P["Gf"] + (P["Ga"] * P["Cp"]) + P["Gt"]

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    **Pecuária:**
    - VE = {P['Gf']:,.0f} + ({P['Gp']:,.0f} × {P['Cp']}) + {P['Gt']:,.0f}
    - **VE = R$ {VE_pec:,.0f}/ha**
    """)
with col2:
    st.markdown(f"""
    **Soja:**
    - VE = {P['Gf']:,.0f} + ({P['Ga']:,.0f} × {P['Cp']}) + {P['Gt']:,.0f}
    - **VE = R$ {VE_soja:,.0f}/ha**
    """)

# ── 5. Comportamento ────────────────────────────────────────────────
st.markdown("---")
st.header("5. Comportamento (C = VE - VD - c)")

C_pec = VE_pec - (VD_total + P["c_desmat"])
C_soja = VE_soja - (VD_total + P["c_desmat"])

summary = {
    "Variável": [
        "VD admin (R$/ha)", "VD ACP (R$/ha)", "VD total (R$/ha)",
        "VE pecuária (R$/ha)", "VE soja (R$/ha)",
        "Custo desmatamento (R$/ha)",
        "C pecuária (R$/ha)", "C soja (R$/ha)",
    ],
    "Valor": [
        f"R$ {VD_admin:,.2f}", f"R$ {VD_acp:,.2f}", f"R$ {VD_total:,.2f}",
        f"R$ {VE_pec:,.0f}", f"R$ {VE_soja:,.0f}",
        f"R$ {P['c_desmat']:,.0f}",
        f"R$ {C_pec:,.2f}", f"R$ {C_soja:,.2f}",
    ],
    "Interpretação": [
        "", "", "",
        "", "", "",
        "COMPENSA" if C_pec > 0 else "NÃO COMPENSA",
        "COMPENSA" if C_soja > 0 else "NÃO COMPENSA",
    ],
}
st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

if C_pec > 0:
    st.error(
        f"**Pecuária:** C = R$ {C_pec:,.2f}/ha > 0 — desmatar compensa economicamente. "
        f"A dissuasão total (R$ {VD_total:,.2f}/ha) é insuficiente frente "
        f"à vantagem econômica (R$ {VE_pec:,.0f}/ha)."
    )
else:
    st.success(
        f"**Pecuária:** C = R$ {C_pec:,.2f}/ha < 0 — desmatar não compensa."
    )

if C_soja > 0:
    st.error(
        f"**Soja:** C = R$ {C_soja:,.2f}/ha > 0 — desmatar compensa economicamente."
    )
else:
    st.success(
        f"**Soja:** C = R$ {C_soja:,.2f}/ha < 0 — desmatar não compensa."
    )

# ── 6. Análise de sensibilidade: qual V_ACP torna C ≤ 0? ──────────
st.markdown("---")
st.header("6. Análise: Valor de ACP necessário para dissuasão total")

st.markdown("""
Para que C ≤ 0, precisamos que VD_total ≥ VE - c.
Fixando os demais parâmetros, qual o **V_ACP mínimo** que torna o desmatamento irracional?
""")

# VD_total = VD_admin + Pd * P_inq * P_acp * P_cond * P_exec * V_ACP * exp(-r*t)
# Queremos VD_total >= VE - c
# V_ACP_min = (VE - c - VD_admin) / (Pd * P_inq * P_acp * P_cond * P_exec * exp(-r*t))

target_pec = VE_pec - P["c_desmat"]
target_soja = VE_soja - P["c_desmat"]

denom = prob_chain_acp * discount_acp
if denom > 0:
    V_ACP_min_pec = max(0, (target_pec - VD_admin) / denom)
    V_ACP_min_soja = max(0, (target_soja - VD_admin) / denom)
else:
    V_ACP_min_pec = float("inf")
    V_ACP_min_soja = float("inf")

col1, col2 = st.columns(2)
with col1:
    if V_ACP_min_pec < 1e9:
        st.metric("V_ACP mínimo (pecuária)", f"R$ {V_ACP_min_pec:,.0f}/ha")
    else:
        st.warning("Cadeia de probabilidades é zero — ACP sem efeito.")
with col2:
    if V_ACP_min_soja < 1e9:
        st.metric("V_ACP mínimo (soja)", f"R$ {V_ACP_min_soja:,.0f}/ha")
    else:
        st.warning("Cadeia de probabilidades é zero — ACP sem efeito.")

st.markdown("""
**Nota:** Estes valores assumem que todos os demais parâmetros permanecem constantes.
Na prática, aumentos nas probabilidades de inquérito, condenação e execução
reduzem significativamente o valor de ACP necessário.
""")

# ── 7. Referências ──────────────────────────────────────────────────
st.markdown("---")
st.header("7. Referências")
st.markdown("""
- Becker, G. S. (1968). *Crime and Punishment: An Economic Approach*. Journal of Political Economy, 76(2), 169–217.
- Schmitt, J. (2015). *Crime sem castigo: a efetividade da fiscalização ambiental para o controle do desmatamento ilegal na Amazônia*. Tese de doutorado, Universidade de Brasília.
- Börner, J. et al. (2009). *Direct conservation payments in the Brazilian Amazon*. Ecological Economics, 69(6), 1272–1282.
- IBAMA (2025). Dados de fiscalização ambiental na Amazônia Legal.
- INPE/PRODES (2025). Taxa de desmatamento na Amazônia.
- IMEA (2025). Custos de produção da soja em Mato Grosso.
- Cepea/Esalq. Margens operacionais da pecuária.
""")
