import json
import joblib
import pandas as pd
import streamlit as st
import tensorflow as tf
import sklearn.compose._column_transformer as _ct
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass
    _ct._RemainderColsList = _RemainderColsList
# ---------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------
st.set_page_config(
    page_title="SP // Triagem de Ocorrências (MLP)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Estilo customizado: Terminal de Inteligência Urbana
# ---------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #0b0f17;
    color: #e2e8f0;
}

.stApp {
    background: radial-gradient(circle at 15% 10%, #131c2e 0%, #0b0f17 55%);
}

/* Cabeçalho técnico */
.top-banner {
    border-left: 4px solid #38bdf8;
    background: rgba(15, 23, 42, 0.85);
    border-top: 1px solid #1e293b;
    border-right: 1px solid #1e293b;
    border-bottom: 1px solid #1e293b;
    padding: 1.4rem 1.8rem;
    border-radius: 6px;
    margin-bottom: 1.5rem;
}

.tag-sistema {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    color: #38bdf8;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}

.titulo-principal {
    font-size: 1.85rem;
    font-weight: 700;
    color: #f8fafc;
    margin: 0;
    line-height: 1.2;
}

.subtitulo {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 0.45rem;
}

/* Painéis */
.painel-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 1.4rem;
}

.label-secao {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #64748b;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.5rem;
    margin-bottom: 1.1rem;
}

/* Resultado */
.veredito-box {
    border-radius: 8px;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
    border: 1px solid;
}

.veredito-roubo {
    background: rgba(239, 68, 68, 0.08);
    border-color: rgba(239, 68, 68, 0.45);
}

.veredito-furto {
    background: rgba(56, 189, 248, 0.08);
    border-color: rgba(56, 189, 248, 0.45);
}

.veredito-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.veredito-classe {
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0.2rem 0 0.4rem 0;
}

/* Barra dupla de espectro */
.espectro-container {
    margin: 1.2rem 0;
}

.espectro-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    margin-bottom: 0.4rem;
}

.espectro-barra {
    width: 100%;
    height: 14px;
    background-color: #1e293b;
    border-radius: 999px;
    overflow: hidden;
    display: flex;
}

.barra-furto {
    background: linear-gradient(90deg, #0284c7, #38bdf8);
    height: 100%;
    transition: width 0.4s ease;
}

.barra-roubo {
    background: linear-gradient(90deg, #f97316, #ef4444);
    height: 100%;
    transition: width 0.4s ease;
}

/* Ficha técnica e aviso */
.ficha-item {
    display: flex;
    justify-content: space-between;
    padding: 0.45rem 0;
    border-bottom: 1px dashed #1e293b;
    font-size: 0.88rem;
}

.ficha-chave {
    color: #94a3b8;
    font-family: 'IBM Plex Mono', monospace;
}

.ficha-valor {
    color: #f1f5f9;
    font-weight: 600;
}

.nota-metodologica {
    margin-top: 1.2rem;
    padding: 0.9rem 1rem;
    background: rgba(30, 41, 59, 0.55);
    border-left: 3px solid #f59e0b;
    border-radius: 4px;
    font-size: 0.84rem;
    color: #cbd5e1;
    line-height: 1.45;
}

/* Botão */
div.stButton > button {
    width: 100%;
    background: #38bdf8;
    color: #0b0f17;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
    border: none;
    padding: 0.7rem 1rem;
    border-radius: 6px;
    margin-top: 0.6rem;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background: #7dd3fc;
    color: #0b0f17;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Carregamento em cache dos artefatos treinados
# ---------------------------------------------------------
@st.cache_resource
def carregar_artefatos():
    modelo = tf.keras.models.load_model("artefatos/modelo_mlp.keras")
    preprocessador = joblib.load("artefatos/preprocessador.joblib")
    with open("artefatos/opcoes_formulario.json", "r", encoding="utf-8") as f:
        opcoes = json.load(f)
    with open("artefatos/resumo_modelo.json", "r", encoding="utf-8") as f:
        resumo = json.load(f)
    return modelo, preprocessador, opcoes, resumo

modelo, preprocessador, opcoes, resumo = carregar_artefatos()

# ---------------------------------------------------------
# Cabeçalho
# ---------------------------------------------------------
st.markdown(f"""
<div class="top-banner">
    <div class="tag-sistema">SÃO PAULO // BASE SSP-SP 2025 // REDE NEURAL MLP ({resumo['arquitetura']})</div>
    <h1 class="titulo-principal">Triagem de Perfil de Ocorrência: Roubo vs. Furto</h1>
    <div class="subtitulo">
        Classificação baseada exclusivamente no contexto espacial e temporal do registro policial 
        (subtração de aparelhos celulares na capital).
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Layout principal em 2 colunas assimétricas
# ---------------------------------------------------------
col_form, col_res = st.columns([1.05, 1.15], gap="large")

with col_form:
    st.markdown('<div class="painel-card">', unsafe_allow_html=True)
    st.markdown('<div class="label-secao">01 // Parâmetros da Ocorrência</div>', unsafe_allow_html=True)

    # Bairro com opção de digitar outro (o OneHotEncoder trata desconhecidos como 'infrequent')
    lista_bairros = opcoes["bairros"] + ["OUTRO BAIRRO (NÃO LISTADO)"]
    idx_padrao_bairro = lista_bairros.index("PINHEIROS") if "PINHEIROS" in lista_bairros else 0
    bairro_sel = st.selectbox("Bairro da ocorrência (Capital)", lista_bairros, index=idx_padrao_bairro)

    if bairro_sel == "OUTRO BAIRRO (NÃO LISTADO)":
        bairro_final = st.text_input("Informe o nome do bairro", value="NÃO INFORMADO").strip().upper()
        if not bairro_final:
            bairro_final = "NÃO INFORMADO"
    else:
        bairro_final = bairro_sel

    # Tipo de local e subtipo encadeado dinamicamente
    tipos_local = opcoes["tipo_local"]
    idx_via = tipos_local.index("Via Pública") if "Via Pública" in tipos_local else 0
    tipo_local = st.selectbox("Tipo de local", tipos_local, index=idx_via)

    subtipos_disponiveis = opcoes["subtipo_por_tipo"].get(tipo_local, ["NÃO INFORMADO"])
    subtipo_local = st.selectbox("Subtipo de local", subtipos_disponiveis)

    # Dia da semana e período lado a lado
    c1, c2 = st.columns(2)
    with c1:
        dia_semana = st.selectbox("Dia da semana", opcoes["dia_semana"], index=4)  # sexta
    with c2:
        periodo = st.selectbox("Período do dia", opcoes["periodo"], index=3)  # noite

    analisar = st.button("EXECUTAR CLASSIFICAÇÃO")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Inferência e Exibição do Diagnóstico
# ---------------------------------------------------------
entrada = pd.DataFrame([{
    "BAIRRO": bairro_final,
    "DESCR_TIPOLOCAL": tipo_local,
    "DESCR_SUBTIPOLOCAL": subtipo_local,
    "dia_semana": dia_semana,
    "periodo": periodo
}])

colunas_ordem = ["BAIRRO", "DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL", "dia_semana", "periodo"]
x_proc = preprocessador.transform(entrada[colunas_ordem])
prob_roubo = float(modelo.predict(x_proc, verbose=0)[0][0])
prob_furto = 1.0 - prob_roubo

eh_roubo = prob_roubo >= 0.5
classe_prevista = "ROUBO (Art. 157)" if eh_roubo else "FURTO (Art. 155)"
descricao_perfil = (
    "Perfil compatível com subtração mediante abordagem, grave ameaça ou violência."
    if eh_roubo else
    "Perfil compatível com subtração sem abordagem direta ou sem percepção imediata da vítima."
)

with col_res:
    st.markdown('<div class="painel-card">', unsafe_allow_html=True)
    st.markdown('<div class="label-secao">02 // Diagnóstico Estimado pela MLP</div>', unsafe_allow_html=True)

    classe_css = "veredito-roubo" if eh_roubo else "veredito-furto"
    cor_texto = "#f87171" if eh_roubo else "#38bdf8"

    st.markdown(f"""
    <div class="veredito-box {classe_css}">
        <div class="veredito-tag" style="color: {cor_texto};">CLASSIFICAÇÃO PREDOMINANTE</div>
        <div class="veredito-classe" style="color: {cor_texto};">{classe_prevista}</div>
        <div style="color: #cbd5e1; font-size: 0.92rem;">{descricao_perfil}</div>
    </div>
    """, unsafe_allow_html=True)

    # Barra comparativa de probabilidade
    st.markdown(f"""
    <div class="espectro-container">
        <div class="espectro-labels">
            <span style="color: #38bdf8;">FURTO: {prob_furto:.1%}</span>
            <span style="color: #f87171;">ROUBO: {prob_roubo:.1%}</span>
        </div>
        <div class="espectro-barra">
            <div class="barra-furto" style="width: {prob_furto * 100:.2f}%;"></div>
            <div class="barra-roubo" style="width: {prob_roubo * 100:.2f}%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Resumo técnico da arquitetura carregada
    st.markdown('<div class="label-secao" style="margin-top: 1.4rem;">03 // Metadados do Modelo em Produção</div>', unsafe_allow_html=True)
    camadas_str = " → ".join(map(str, resumo["camadas"])) + " → 1 (sigmoid)"
    m_teste = resumo["metricas_teste"]

    st.markdown(f"""
    <div class="ficha-item">
        <span class="ficha-chave">Arquitetura Selecionada</span>
        <span class="ficha-valor">{resumo['arquitetura']} ({camadas_str} | {resumo['ativacao']})</span>
    </div>
    <div class="ficha-item">
        <span class="ficha-chave">Desempenho no Teste (F1 / Acurácia)</span>
        <span class="ficha-valor">{m_teste['f1']:.3f} / {m_teste['acuracia']:.1%}</span>
    </div>
    <div class="ficha-item">
        <span class="ficha-chave">Bairro Avaliado</span>
        <span class="ficha-valor">{bairro_final}</span>
    </div>
    <div class="nota-metodologica">
        <strong>Nota de interpretação:</strong> Este sistema não calcula o risco de um crime acontecer na rua. 
        Dado que uma ocorrência foi registrada com as características selecionadas, o modelo estima a 
        probabilidade estatística de o boletim corresponder a um <em>roubo</em> ou a um <em>furto</em> 
        com base nos padrões históricos da SSP-SP em 2025.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
