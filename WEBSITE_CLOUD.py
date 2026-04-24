# Requisitos:
# pip install streamlit rasterio numpy pandas plotly folium streamlit-folium pillow pyproj matplotlib

import streamlit as st
import rasterio
import folium
import os
import numpy as np
import pandas as pd
from rasterio.transform import rowcol
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from folium import raster_layers
from PIL import Image
import pathlib

from pyproj import Transformer
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from io import BytesIO

# =====================================================================
# CONFIGURAÇÃO INICIAL
# =====================================================================
base_path = pathlib.Path(__file__).parent

classif_folder = base_path
output_vis_folder = base_path

csv_path = "area_macrofitas.csv"
logo_path = "https://raw.githubusercontent.com/gpereiraufsj/Macrofitas_Aquasmart/main/Logo.png"

st.set_page_config(layout="wide", page_title="AQUASMART • Dashboard Científico")

# =====================================================================
# SIDEBAR • LOGO + NAVEGAÇÃO
# =====================================================================
with st.sidebar:
    try:
        st.image(logo_path, use_column_width=True)
    except TypeError:
        st.image(logo_path)

    st.title("AQUASMART")
    st.caption("Dashboard científico • Monitoramento")

    pagina = st.radio(
        "Navegação",
        [
            "📊 Síntese Integrada",
            "🌿 Macrófitas",
            "💧 Qualidade da Água",
            "🧱 Sedimentos & Risco",
        ],
        index=0
    )

# =====================================================================
# HEADER
# =====================================================================
st.markdown("# AQUASMART • Dashboard Científico")
st.caption("Macrófitas • Qualidade da Água • Síntese integrada • Sedimentos e risco ambiental")
st.markdown("---")

# =====================================================================
# FUNÇÕES AUXILIARES (compartilhadas)
# =====================================================================
EPS = 1e-6


def list_water_files(folder: pathlib.Path):
    return sorted([p for p in folder.glob("DATA_*.tif")])


def parse_date_from_filename(p: pathlib.Path) -> str:
    return p.stem.replace("DATA_", "")


def bounds_3857_to_4326(bounds):
    """Converte bounds (left,bottom,right,top) de EPSG:3857 -> EPSG:4326 para usar no Folium."""
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
    lon1, lat1 = transformer.transform(left, bottom)
    lon2, lat2 = transformer.transform(right, top)
    return [[lat1, lon1], [lat2, lon2]]  # [[SW],[NE]]


def get_transformer_to_raster(raster_crs):
    """Transforma lon/lat (EPSG:4326) -> CRS do raster (ex.: EPSG:3857)."""
    if raster_crs is None:
        return None
    epsg = raster_crs.to_epsg() if hasattr(raster_crs, "to_epsg") else None
    if epsg == 4326:
        return None
    return Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)


def read_band(src, idx):
    return src.read(idx).astype("float32")


def compute_ndvi(R, NIR):
    return (NIR - R) / (NIR + R + EPS)


def compute_ndwi(G, NIR):
    return (G - NIR) / (G + NIR + EPS)


def normalize_to_uint8_diag(a, vmin=None, vmax=None):
    """Para NDVI/NDWI (diagnóstico) — autoescala robusta por percentis."""
    a = np.asarray(a, dtype="float32")
    valid = np.isfinite(a)

    if not np.any(valid):
        return np.zeros_like(a, dtype=np.uint8), 0.0, 1.0

    if vmin is None:
        vmin = float(np.nanpercentile(a, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(a, 98))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    x = (a - vmin) / (vmax - vmin)
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8), vmin, vmax


def colormap_rgba(uint8_img, cmap_name="viridis"):
    """Aplica colormap do Matplotlib em uint8 (0–255) e retorna RGBA uint8."""
    uint8_img = np.asarray(uint8_img, dtype=np.uint8)
    cmap = cm.get_cmap(cmap_name)
    x = uint8_img.astype("float32") / 255.0
    rgba = (cmap(x) * 255).astype(np.uint8)
    return rgba


def make_colorbar_image(vmin: float, vmax: float, cmap_name: str, label: str = "") -> Image.Image:
    """Gera uma colorbar (PNG) como PIL Image para exibir no Streamlit."""
    fig, ax = plt.subplots(figsize=(5.2, 0.7))
    fig.subplots_adjust(bottom=0.35, left=0.08, right=0.98, top=0.95)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name)),
        cax=ax,
        orientation="horizontal"
    )
    if label:
        cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=9)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def sample_from_array(src, arr, lon, lat):
    """Amostra arr (mesma grade do raster) no ponto (lon/lat EPSG:4326)."""
    transformer = get_transformer_to_raster(src.crs)
    if transformer:
        x, y = transformer.transform(lon, lat)
    else:
        x, y = lon, lat

    r, c = rowcol(src.transform, x, y)
    if r < 0 or c < 0 or r >= arr.shape[0] or c >= arr.shape[1]:
        return np.nan
    v = arr[r, c]
    return float(v) if np.isfinite(v) else np.nan


# =====================================================================
# MÓDULOS DEMONSTRATIVOS — DADOS GENÉRICOS PARA CONCEPÇÃO DO DASHBOARD
# =====================================================================
@st.cache_data
def make_demo_integrated_data(seed: int = 42):
    """Cria dados simulados para demonstrar um painel integrado do projeto."""
    rng = np.random.default_rng(seed)

    etapas = pd.DataFrame({
        "Etapa": [
            "Diagnóstico ambiental",
            "Sedimentos e dragagem",
            "Risco hidrodinâmico",
            "Macrófitas e qualidade da água",
            "Governança e divulgação",
            "OneHealth",
        ],
        "Conclusão_%": [82, 74, 68, 78, 46, 52],
        "Produtos_previstos": [8, 7, 5, 9, 6, 4],
        "Produtos_entregues": [6, 5, 3, 7, 2, 2],
        "Prioridade": ["Alta", "Alta", "Alta", "Muito alta", "Média", "Média"],
    })
    etapas["Status"] = pd.cut(
        etapas["Conclusão_%"],
        bins=[0, 40, 70, 90, 100],
        labels=["Atenção", "Em andamento", "Avançado", "Concluído"],
        include_lowest=True,
    ).astype(str)

    datas = pd.date_range("2024-02-01", "2026-01-01", freq="MS")
    n = len(datas)
    t = np.arange(n)
    saz = np.sin(2 * np.pi * (t / 12.0))

    serie = pd.DataFrame({
        "Data": datas,
        "Macrófitas_ha": np.clip(38 + 10 * saz + 0.55 * t + rng.normal(0, 3.0, n), 5, None),
        "Clorofila_ugL": np.clip(72 + 18 * saz + rng.normal(0, 9, n), 10, None),
        "Turbidez_NTU": np.clip(9 + 2.8 * saz + rng.normal(0, 1.6, n), 1, None),
        "Sedimentacao_g_m2_mes": np.clip(430 + 75 * saz + 8 * t + rng.normal(0, 35, n), 120, None),
        "Risco_hidrologico_0_100": np.clip(42 + 16 * saz + 1.2 * t + rng.normal(0, 6, n), 0, 100),
        "Engajamento_stakeholders": np.clip(20 + 2.4 * t + rng.normal(0, 4, n), 0, 100),
    })
    return etapas, serie


@st.cache_data
def make_demo_sediment_data(seed: int = 7):
    """Cria pontos simulados de sedimento/batimetria para mapa e gráficos."""
    rng = np.random.default_rng(seed)
    nomes = [
        "R1 Margem", "R1 Centro", "R2 Margem", "R2 Centro", "R3 Centro",
        "R4 Margem", "R4 Centro", "R5 Margem", "R5 Centro", "Tributário",
    ]
    lat0, lon0 = -20.022, -44.105
    df = pd.DataFrame({
        "Ponto": nomes,
        "Latitude": lat0 + rng.normal(0, 0.010, len(nomes)),
        "Longitude": lon0 + rng.normal(0, 0.014, len(nomes)),
        "P_total_mgkg": rng.uniform(650, 2300, len(nomes)),
        "N_total_%": rng.uniform(0.12, 0.85, len(nomes)),
        "MO_%": rng.uniform(4.5, 19.0, len(nomes)),
        "As_mgkg": rng.uniform(3.0, 38.0, len(nomes)),
        "Pb_mgkg": rng.uniform(8.0, 92.0, len(nomes)),
        "Taxa_sed_g_m2_mes": rng.uniform(180, 850, len(nomes)),
        "Perda_volume_%": rng.uniform(8, 36, len(nomes)),
    })

    def minmax(s):
        return (s - s.min()) / (s.max() - s.min() + EPS)

    df["Score_prioridade"] = (
        0.25 * minmax(df["P_total_mgkg"]) +
        0.20 * minmax(df["MO_%"]) +
        0.20 * minmax(df["As_mgkg"]) +
        0.15 * minmax(df["Pb_mgkg"]) +
        0.20 * minmax(df["Taxa_sed_g_m2_mes"])
    ) * 100

    df["Classe_risco"] = pd.cut(
        df["Score_prioridade"],
        bins=[0, 35, 65, 100],
        labels=["Baixo", "Médio", "Alto"],
        include_lowest=True,
    ).astype(str)
    return df.sort_values("Score_prioridade", ascending=False)


@st.cache_data
def make_demo_scenario_data():
    anos = np.array([2024, 2030, 2040, 2050])
    cenarios = []
    base = 12.7
    specs = {
        "Referência": [-0.00, -0.55, -1.35, -2.30],
        "Controle de erosão": [0.00, -0.30, -0.75, -1.20],
        "Dragagem + NBS": [0.00, 0.95, 0.65, 0.25],
        "Pressão urbana": [0.00, -0.85, -2.10, -3.35],
    }
    for cenario, delta in specs.items():
        for ano, d in zip(anos, delta):
            vol = base + d
            risco = np.clip(100 - (vol / base) * 72 + (ano - 2024) * 0.45, 0, 100)
            cenarios.append({
                "Ano": int(ano),
                "Cenário": cenario,
                "Volume_útil_milhões_m3": round(vol, 2),
                "Índice_risco_0_100": round(float(risco), 1),
            })
    return pd.DataFrame(cenarios)


def page_sintese_integrada():
    st.subheader("📊 Síntese Integrada do Projeto")
    st.caption(
        "Módulo demonstrativo com dados simulados. A ideia é transformar o relatório técnico em um painel de acompanhamento científico, "
        "com visão por etapa, indicadores ambientais e alertas de gestão."
    )

    etapas, serie = make_demo_integrated_data()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filtros da síntese")
    status_sel = st.sidebar.multiselect(
        "Status das etapas:",
        sorted(etapas["Status"].unique()),
        default=sorted(etapas["Status"].unique()),
        key="sintese_status",
    )
    etapas_f = etapas[etapas["Status"].isin(status_sel)].copy()

    media_conclusao = etapas["Conclusão_%"].mean()
    produtos_previstos = int(etapas["Produtos_previstos"].sum())
    produtos_entregues = int(etapas["Produtos_entregues"].sum())
    indice_alerta = float(serie["Risco_hidrologico_0_100"].iloc[-1])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conclusão média", f"{media_conclusao:.1f}%")
    c2.metric("Produtos entregues", f"{produtos_entregues}/{produtos_previstos}")
    c3.metric("Índice de risco", f"{indice_alerta:.1f}/100")
    c4.metric("Meses monitorados", f"{serie['Data'].nunique()}")

    col_a, col_b = st.columns([1.2, 1.0])
    with col_a:
        fig = px.bar(
            etapas_f,
            x="Conclusão_%",
            y="Etapa",
            orientation="h",
            color="Status",
            text="Conclusão_%",
            title="Andamento demonstrativo por etapa do projeto",
            labels={"Conclusão_%": "Conclusão (%)"},
        )
        fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_range=[0, 105])
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        radar = pd.DataFrame({
            "Dimensão": [
                "Qualidade da água", "Macrófitas", "Sedimentos",
                "Risco hidrológico", "Governança", "OneHealth",
            ],
            "Índice": [72, 78, 66, 61, 48, 55],
        })
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar["Índice"],
            theta=radar["Dimensão"],
            fill="toself",
            name="Condição/avanço",
        ))
        fig_radar.update_layout(
            title="Radar sintético de maturidade ambiental",
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("### Séries integradas demonstrativas")
    vars_demo = st.multiselect(
        "Indicadores para visualizar:",
        [c for c in serie.columns if c != "Data"],
        default=["Macrófitas_ha", "Clorofila_ugL", "Sedimentacao_g_m2_mes", "Risco_hidrologico_0_100"],
        key="sintese_vars",
    )
    if vars_demo:
        long = serie.melt(id_vars="Data", value_vars=vars_demo, var_name="Indicador", value_name="Valor")
        fig_ts = px.line(long, x="Data", y="Valor", color="Indicador", markers=True, title="Evolução temporal integrada")
        st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("### Matriz exploratória entre indicadores")
    corr = serie.drop(columns="Data").corr(numeric_only=True)
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        aspect="auto",
        title="Correlação entre indicadores simulados",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    with st.expander("Tabela demonstrativa das etapas"):
        st.dataframe(etapas, use_container_width=True)


def page_sedimentos_risco():
    st.subheader("🧱 Sedimentos, Batimetria & Risco Ambiental")
    st.caption(
        "Módulo demonstrativo com dados genéricos. Foi pensado para integrar assoreamento, qualidade do sedimento, "
        "priorização de dragagem e cenários de gestão do reservatório."
    )

    df = make_demo_sediment_data()
    cen = make_demo_scenario_data()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filtros de sedimento")
    risco_sel = st.sidebar.multiselect(
        "Classe de risco:",
        ["Baixo", "Médio", "Alto"],
        default=["Baixo", "Médio", "Alto"],
        key="sed_risco",
    )
    df_f = df[df["Classe_risco"].isin(risco_sel)].copy()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pontos avaliados", f"{len(df_f)}")
    m2.metric("Prioridade média", f"{df_f['Score_prioridade'].mean():.1f}/100" if not df_f.empty else "-")
    m3.metric("P total médio", f"{df_f['P_total_mgkg'].mean():.0f} mg/kg" if not df_f.empty else "-")
    m4.metric("Taxa sed. média", f"{df_f['Taxa_sed_g_m2_mes'].mean():.0f} g/m²/mês" if not df_f.empty else "-")

    col_map, col_bar = st.columns([1.15, 1.0])
    with col_map:
        st.markdown("### Mapa de pontos prioritários")
        fmap = folium.Map(location=[-20.022, -44.105], zoom_start=13, tiles="OpenStreetMap")
        color_map = {"Baixo": "green", "Médio": "orange", "Alto": "red"}
        for _, row in df_f.iterrows():
            popup = (
                f"<b>{row['Ponto']}</b><br>"
                f"Risco: {row['Classe_risco']}<br>"
                f"Score: {row['Score_prioridade']:.1f}/100<br>"
                f"P total: {row['P_total_mgkg']:.0f} mg/kg<br>"
                f"Taxa sed.: {row['Taxa_sed_g_m2_mes']:.0f} g/m²/mês"
            )
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=6 + row["Score_prioridade"] / 12,
                popup=popup,
                color=color_map.get(row["Classe_risco"], "blue"),
                fill=True,
                fill_opacity=0.72,
            ).add_to(fmap)
        st_folium(fmap, width=760, height=520, key="sedimentos_mapa")

    with col_bar:
        st.markdown("### Ranking de intervenção")
        fig_rank = px.bar(
            df_f.sort_values("Score_prioridade"),
            x="Score_prioridade",
            y="Ponto",
            orientation="h",
            color="Classe_risco",
            text="Score_prioridade",
            title="Prioridade demonstrativa para manejo/dragagem",
            labels={"Score_prioridade": "Score de prioridade"},
        )
        fig_rank.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_rank.update_layout(xaxis_range=[0, 105])
        st.plotly_chart(fig_rank, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig_scatter = px.scatter(
            df_f,
            x="P_total_mgkg",
            y="As_mgkg",
            size="Taxa_sed_g_m2_mes",
            color="Classe_risco",
            hover_name="Ponto",
            title="Nutrientes, metal/metaloide e sedimentação",
            labels={
                "P_total_mgkg": "Fósforo total no sedimento (mg/kg)",
                "As_mgkg": "As (mg/kg)",
            },
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        fig_box = px.box(
            df_f,
            x="Classe_risco",
            y="Perda_volume_%",
            points="all",
            title="Perda de volume simulada por classe de risco",
            labels={"Perda_volume_%": "Perda de volume (%)"},
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### Cenários demonstrativos de gestão")
    col3, col4 = st.columns(2)
    with col3:
        fig_vol = px.line(
            cen,
            x="Ano",
            y="Volume_útil_milhões_m3",
            color="Cenário",
            markers=True,
            title="Projeção conceitual de volume útil",
            labels={"Volume_útil_milhões_m3": "Volume útil (milhões m³)"},
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    with col4:
        fig_risk = px.line(
            cen,
            x="Ano",
            y="Índice_risco_0_100",
            color="Cenário",
            markers=True,
            title="Índice conceitual de risco ambiental/hidrológico",
            labels={"Índice_risco_0_100": "Índice de risco (0–100)"},
        )
        fig_risk.update_yaxes(range=[0, 100])
        st.plotly_chart(fig_risk, use_container_width=True)

    with st.expander("Tabelas demonstrativas"):
        st.markdown("**Pontos simulados de sedimento**")
        st.dataframe(df, use_container_width=True)
        st.markdown("**Cenários simulados**")
        st.dataframe(cen, use_container_width=True)




# =====================================================================
# PÁGINA 1 — MACRÓFITAS
# =====================================================================
if pagina == "📊 Síntese Integrada":
    page_sintese_integrada()

elif pagina == "🌿 Macrófitas":
    st.subheader("🌿 Monitoramento de Macrófitas")

    df_area = pd.read_csv(csv_path)
    df_area["Data"] = pd.to_datetime(df_area["Data"])

    df_area["Area_ha"] = df_area["Area_m2"] / 10_000
    if "Area_smooth" in df_area.columns:
        df_area["Area_smooth_ha"] = df_area["Area_smooth"] / 10_000

    min_date, max_date = df_area["Data"].min(), df_area["Data"].max()

    start_date, end_date = st.sidebar.date_input(
        "📆 Intervalo de datas:",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )

    st.subheader("📆 Análise Mensal de Área Média")
    df_area["Mês"] = df_area["Data"].dt.month
    mensal = df_area.groupby("Mês").mean(numeric_only=True).reset_index()

    fig_mensal = px.bar(
        mensal,
        x="Mês",
        y=mensal["Area_m2"] / 10_000,
        labels={"y": "Área Média (ha)"},
        title="Área Média de Macrófitas por Mês",
        text_auto=".2s"
    )
    st.plotly_chart(fig_mensal, use_container_width=True)
    st.markdown("---")

    filtradas = df_area[
        (df_area["Data"] >= pd.to_datetime(start_date)) &
        (df_area["Data"] <= pd.to_datetime(end_date))
    ]

    total_ha = float(filtradas["Area_ha"].sum())
    max_ha = float(filtradas["Area_ha"].max())
    data_max = filtradas.loc[filtradas["Area_ha"].idxmax(), "Data"].strftime("%Y-%m-%d")
    mean_ha = filtradas.groupby(filtradas["Data"].dt.year)["Area_ha"].mean()

    st.markdown("### 📌 Estatísticas do Período Selecionado")
    col1, col2, col3 = st.columns(3)
    col1.metric("🌱 Área Total", f"{total_ha:,.2f} ha")
    col2.metric("📈 Máxima", f"{max_ha:,.2f} ha", data_max)
    col3.metric("📊 Média Anual", f"{float(mean_ha.mean()):,.2f} ha")

    fig_area = px.line(
        filtradas,
        x="Data",
        y=["Area_ha", "Area_smooth_ha"] if "Area_smooth_ha" in filtradas.columns else ["Area_ha"],
        markers=True,
        labels={"value": "Área (ha)", "variable": "Tipo"},
        title="Evolução da Área de Macrófitas (ha)"
    )
    st.plotly_chart(fig_area, use_container_width=True)

    classif_files = sorted([
        f for f in os.listdir(classif_folder)
        if f.startswith("classificado_macrofitas") and f.endswith(".tif")
    ])
    dates = [f.replace("classificado_macrofitas_", "").replace(".tif", "") for f in classif_files]

    selected_date = st.selectbox("📅 Selecione a data da imagem:", dates)
    file_selected = os.path.join(classif_folder, f"classificado_macrofitas_{selected_date}.tif")

    col_mapa, col_grafico = st.columns([1, 1])

    with col_mapa:
        st.subheader("🗺️ Mapa Classificado - Clique para ver a evolução temporal")
        with rasterio.open(file_selected) as src:
            img = src.read(1)
            bounds = src.bounds

        m = folium.Map(
            location=[(bounds.top + bounds.bottom) / 2, (bounds.left + bounds.right) / 2],
            zoom_start=13
        )
        overlay_img = np.where(img == 1, 255, 0).astype(np.uint8)

        raster_layers.ImageOverlay(
            image=overlay_img,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            colormap=lambda x: (0, 1, 0, x / 255),
            opacity=0.6,
        ).add_to(m)

        folium.LayerControl().add_to(m)
        click_data = st_folium(m, width=600, height=450)

    with col_grafico:
        st.subheader("📊 Presença de Macrófitas no ponto clicado")
        if click_data and click_data.get("last_clicked"):
            lon = click_data["last_clicked"]["lng"]
            lat = click_data["last_clicked"]["lat"]
            st.success(f"Coordenada: ({lat:.5f}, {lon:.5f})")

            resultados = []
            for f in classif_files:
                dt = f.replace("classificado_macrofitas_", "").replace(".tif", "")
                path = os.path.join(classif_folder, f)
                with rasterio.open(path) as src:
                    try:
                        rr, cc = rowcol(src.transform, lon, lat)
                        val = src.read(1)[rr, cc]
                        resultados.append({"Data": dt, "Macrofita": int(val)})
                    except Exception:
                        resultados.append({"Data": dt, "Macrofita": np.nan})

            df_ponto = pd.DataFrame(resultados)
            df_ponto["Data"] = pd.to_datetime(df_ponto["Data"])
            df_ponto = df_ponto.sort_values("Data")

            fig2 = px.line(
                df_ponto,
                x="Data",
                y="Macrofita",
                markers=True,
                title="Presença de Macrófitas (1=sim, 0=não)"
            )
            fig2.update_yaxes(dtick=1, range=[-0.1, 1.1])
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Clique em um ponto no mapa para ver a série temporal.")

    st.subheader("📷 Visualização: RGB | NDVI | Classificação")
    fig_path = os.path.join(output_vis_folder, f"fig_macrofitas_{selected_date}.png")
    if os.path.exists(fig_path):
        st.image(Image.open(fig_path), use_column_width=True)
    else:
        st.warning(f"Imagem não encontrada: {fig_path}")

    st.subheader("📅 Comparação entre Anos")
    years = sorted(df_area["Data"].dt.year.unique())
    year1 = st.selectbox("Ano 1:", years, index=3 if len(years) > 3 else 0)
    year2 = st.selectbox("Ano 2:", years, index=len(years) - 1)

    df_y1 = df_area[df_area["Data"].dt.year == year1].groupby(df_area["Data"].dt.month).mean(numeric_only=True)
    df_y2 = df_area[df_area["Data"].dt.year == year2].groupby(df_area["Data"].dt.month).mean(numeric_only=True)

    fig_comp = px.line(title=f"Comparação Anual: {year1} vs {year2}")
    fig_comp.add_scatter(x=df_y1.index, y=df_y1["Area_ha"], name=f"{year1}", mode="lines+markers")
    fig_comp.add_scatter(x=df_y2.index, y=df_y2["Area_ha"], name=f"{year2}", mode="lines+markers")
    fig_comp.update_layout(xaxis_title="Mês", yaxis_title="Área (ha)")
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")
    st.caption("Versão científica interativa • Desenvolvido com 💚 para o Projeto AQUASMART")


# =====================================================================
# PÁGINA 2 — QUALIDADE DA ÁGUA
# =====================================================================
elif pagina == "💧 Qualidade da Água":
    st.subheader("💧 Qualidade da Água")
    st.caption(
        "Derivado de DATA_*.tif (EPSG:3857) • filtro: remove macrófitas (NDVI > 0.5) • "
        "pixels zerados ocultos • NDWI apenas diagnóstico."
    )

    NDVI_MACROFITAS_THR = 0.50

    VAR_SPECS = {
        "chlor_a":     {"label": "Clorofila-a", "unit": "µg/L", "vmin": 15.0, "vmax": 140.0},
        "turbidity":   {"label": "Turbidez",    "unit": "NTU",  "vmin": 2.5,  "vmax": 20.0},
        "phycocyanin": {"label": "Fitocianina", "unit": "µg/L", "vmin": 2.5,  "vmax": 22.0},
        "secchi":      {"label": "Secchi",      "unit": "cm",   "vmin": 20.0, "vmax": 100.0},
    }

    def robust_scale_to_range(proxy, vmin_out, vmax_out, scale_mask):
        """
        Escala robusta p2–p98 usando APENAS pixels de scale_mask=True.
        Mantém NaN fora do scale_mask.
        """
        proxy = proxy.astype("float32")
    
        # pixels usados para estimar p2/p98
        m = np.isfinite(proxy) & scale_mask
        if not np.any(m):
            return np.full_like(proxy, np.nan, dtype="float32")
    
        p2 = float(np.nanpercentile(proxy[m], 2))
        p98 = float(np.nanpercentile(proxy[m], 98))
    
        # fallback caso a distribuição esteja “colapsada”
        if (not np.isfinite(p2)) or (not np.isfinite(p98)) or (p98 <= p2 + 1e-12):
            p2 = float(np.nanmin(proxy[m]))
            p98 = float(np.nanmax(proxy[m]))
            if (not np.isfinite(p2)) or (not np.isfinite(p98)) or (p98 <= p2 + 1e-12):
                return np.full_like(proxy, np.nan, dtype="float32")
    
        proxy01 = (proxy - p2) / (p98 - p2 + EPS)
        proxy01 = np.clip(proxy01, 0, 1)
    
        out = (vmin_out + proxy01 * (vmax_out - vmin_out)).astype("float32")
        out[~scale_mask] = np.nan  # garante NaN fora da máscara
        return out

    def compute_water_variable_scaled(B, G, R, NIR, var_key: str, valid_mask):
        spec = VAR_SPECS[var_key]
        vmin, vmax = float(spec["vmin"]), float(spec["vmax"])

        if var_key == "chlor_a":
            proxy = (NIR / (R + EPS))
            return robust_scale_to_range(proxy, vmin, vmax, valid_mask)

        if var_key == "phycocyanin":
            proxy = (R / (G + EPS))
            return robust_scale_to_range(proxy, vmin, vmax, valid_mask)

        if var_key == "turbidity":
            proxy = R / (B + G + EPS)
            return robust_scale_to_range(proxy, vmin, vmax, valid_mask)

        if var_key == "secchi":
            proxy_turb = R / (B + G + EPS)
            turb_nt = robust_scale_to_range(proxy_turb, 2.5, 20.0, valid_mask)  # NTU
            sec01 = (turb_nt - 2.5) / (20.0 - 2.5 + EPS)
            out = 100.0 - np.clip(sec01, 0, 1) * (100.0 - 20.0)
            out = np.clip(out, vmin, vmax).astype("float32")
            out[~valid_mask] = np.nan
            return out

        raise ValueError("Variável desconhecida.")

    def compute_filtered_var_and_indices(tif_file: pathlib.Path, var_key: str):
        with rasterio.open(tif_file) as src:
            if src.count < 4:
                raise ValueError("DATA_*.tif precisa ter 4 bandas (B, G, R, NIR).")

            B = read_band(src, 1)
            G = read_band(src, 2)
            R = read_band(src, 3)
            NIR = read_band(src, 4)

            ndvi = compute_ndvi(R, NIR)
            ndwi = compute_ndwi(G, NIR)

            # 1) máscara nodata oficial do raster
            # dataset_mask: 0 = nodata, 255 = válido (em geral)
            ds_mask = src.dataset_mask()
            nodata_mask = (ds_mask == 0)

            # 2) remova pixels “problemáticos” (qualquer banda zero)
            # (isso resolve MUITO para turbidez e fitocianina)
            any_zero = (B == 0) | (G == 0) | (R == 0) | (NIR == 0)

            # 3) máscara final de validade
            valid_mask = (
                (~nodata_mask) &
                (~any_zero) &
                np.isfinite(ndvi) &
                (ndvi <= NDVI_MACROFITAS_THR)
            )
            
            # compute variável já usando valid_mask na escala
            var_scaled = compute_water_variable_scaled(B, G, R, NIR, var_key, valid_mask)
            var_filt = var_scaled  # já vem NaN fora da máscara

            meta = {
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "folium_bounds": bounds_3857_to_4326(src.bounds),
            }
            return var_filt, ndvi, ndwi, meta

    water_files = list_water_files(base_path)
    if not water_files:
        st.warning("Nenhum arquivo encontrado com padrão DATA_*.tif na raiz do repositório.")
        st.stop()

    water_dates = [parse_date_from_filename(p) for p in water_files]

    var_map = {
        "Clorofila-a": "chlor_a",
        "Fitocianina": "phycocyanin",
        "Turbidez": "turbidity",
        "Secchi": "secchi",
    }

    c1, c2, c3, c4, c5 = st.columns([1.4, 1.5, 1.0, 1.3, 1.2])
    DEFAULT_VAR_LABEL = "Clorofila-a"
    DEFAULT_DATE = "2025-06-01"

    var_keys = list(var_map.keys())
    default_var_index = var_keys.index(DEFAULT_VAR_LABEL) if DEFAULT_VAR_LABEL in var_keys else 0

    default_date_index = water_dates.index(DEFAULT_DATE) if DEFAULT_DATE in water_dates else (len(water_dates) - 1)

    with c1:
        var_label = st.selectbox("Variável:", var_keys, index=default_var_index)
    with c2:
        selected_date = st.selectbox("Data (imagem):", water_dates, index=default_date_index)
    with c3:
        cmap_name = st.selectbox("Colormap:", ["viridis", "cividis", "plasma", "inferno", "magma"], index=0)
    with c4:
        use_mean_map = st.checkbox("Mapa médio (média de todas as datas)", value=False)
    with c5:
        show_point_clim = st.checkbox("Climatologia mensal do ponto", value=True)

    gamma = st.slider("Contraste do mapa (gamma)", 0.40, 2.00, 0.85, 0.05)
    use_internal_stretch = st.checkbox("Aumentar contraste (p2–p98 dentro da escala fixa)", value=True)

    var_key = var_map[var_label]
    spec = VAR_SPECS[var_key]
    vmin_fixed, vmax_fixed = float(spec["vmin"]), float(spec["vmax"])
    unit = spec["unit"]
    label_unit = f"{spec['label']} ({unit})"

    tif_path = base_path / f"DATA_{selected_date}.tif"

    try:
        var_A, ndvi_A, ndwi_A, meta_A = compute_filtered_var_and_indices(tif_path, var_key)
    except Exception as e:
        st.error(f"Erro ao processar {tif_path.name}: {e}")
        st.stop()

    #@st.cache_data(show_spinner=True)
    def compute_mean_raster(_var_key: str, _vmin: float, _vmax: float):
        sum_arr = None
        cnt_arr = None
        meta_ref = None

        for p in water_files:
            var_f, _, _, meta = compute_filtered_var_and_indices(p, _var_key)

            var_f = np.where(np.isfinite(var_f), var_f, np.nan)
            valid = np.isfinite(var_f)

            if sum_arr is None:
                sum_arr = np.zeros_like(var_f, dtype=np.float64)
                cnt_arr = np.zeros_like(var_f, dtype=np.uint32)
                meta_ref = meta

            if var_f.shape != sum_arr.shape:
                continue

            sum_arr[valid] += var_f[valid].astype(np.float64)
            cnt_arr[valid] += 1

        mean_arr = np.full_like(sum_arr, np.nan, dtype=np.float32)
        ok = cnt_arr > 0
        mean_arr[ok] = (sum_arr[ok] / cnt_arr[ok]).astype(np.float32)
        return mean_arr, meta_ref, cnt_arr

    if use_mean_map:
        map_arr, meta_use, _ = compute_mean_raster(var_key, vmin_fixed, vmax_fixed)
        map_title = f"{label_unit} • MÉDIA (todas as datas)"
    else:
        map_arr = var_A
        meta_use = meta_A
        map_title = f"{label_unit} • {selected_date}"

    vals = map_arr[np.isfinite(map_arr)]
    if vals.size == 0:
        st.warning("Não sobraram pixels válidos após filtro NDVI e remoção de zeros.")
        st.stop()

    st.markdown("### 📊 Estatística espacial (após filtro NDVI + zeros)")
    stats = {
        "n_pixels": int(vals.size),
        "média": float(np.nanmean(vals)),
        "mediana": float(np.nanmedian(vals)),
        "p10": float(np.nanpercentile(vals, 10)),
        "p90": float(np.nanpercentile(vals, 90)),
    }
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Pixels válidos", f"{stats['n_pixels']:,}")
    s2.metric("Média", f"{stats['média']:.2f} {unit}")
    s3.metric("Mediana", f"{stats['mediana']:.2f} {unit}")
    s4.metric("p10–p90", f"{stats['p10']:.2f} – {stats['p90']:.2f} {unit}")

    if use_internal_stretch:
        p2 = float(np.nanpercentile(vals, 2))
        p98 = float(np.nanpercentile(vals, 98))
        vmin_vis = max(vmin_fixed, p2)
        vmax_vis = min(vmax_fixed, p98)
        if vmax_vis <= vmin_vis:
            vmin_vis, vmax_vis = vmin_fixed, vmax_fixed
    else:
        vmin_vis, vmax_vis = vmin_fixed, vmax_fixed

    st.markdown("### 🗺️ Mapa interativo (escala fixa + contraste)")
    
    # -----------------------------------------------------------------
    # VISUALIZAÇÃO: alpha por pixels válidos + stretch robusto (p2–p98)
    # -----------------------------------------------------------------
    valid_draw = np.isfinite(map_arr)

    vals_vis = map_arr[valid_draw]
    p2 = float(np.nanpercentile(vals_vis, 2))
    p98 = float(np.nanpercentile(vals_vis, 98))

    # fallback se colapsar
    if (not np.isfinite(p2)) or (not np.isfinite(p98)) or (p98 <= p2 + 1e-12):
        p2 = float(np.nanmin(vals_vis))
        p98 = float(np.nanmax(vals_vis))
        if p98 <= p2 + 1e-12:
            p2, p98 = float(vmin_fixed), float(vmax_fixed)

    # clip APENAS para visual (não limita na escala fixa)
    map_clip = np.clip(map_arr, p2, p98)

    norm = np.zeros_like(map_arr, dtype=np.float32)
    norm[valid_draw] = (map_clip[valid_draw] - p2) / (p98 - p2 + EPS)
    norm[valid_draw] = np.clip(norm[valid_draw], 0, 1)
    norm[valid_draw] = norm[valid_draw] ** float(gamma)

    img_u8 = np.zeros_like(map_arr, dtype=np.uint8)
    img_u8[valid_draw] = (norm[valid_draw] * 255.0).astype(np.uint8)

    rgba = colormap_rgba(img_u8, cmap_name=cmap_name)

    # alpha: mostra todo pixel válido
    rgba[..., 3] = 0
    rgba[valid_draw, 3] = 255
    

    folium_bounds = meta_use["folium_bounds"]
    center_lat = (folium_bounds[0][0] + folium_bounds[1][0]) / 2
    center_lon = (folium_bounds[0][1] + folium_bounds[1][1]) / 2

    m = folium.Map(location=[center_lat, center_lon], tiles="OpenStreetMap", zoom_control=True)
    raster_layers.ImageOverlay(
        image=rgba,
        bounds=folium_bounds,
        opacity=0.92,
        interactive=True,
        zindex=1
    ).add_to(m)
    m.fit_bounds(folium_bounds)

    legend_html = f"""
    <div style="
        position: fixed; bottom: 30px; left: 30px; width: 520px; z-index: 9999;
        background-color: white; padding: 10px; border: 1px solid #999; border-radius: 6px;
        font-size: 12px;">
        <b>{map_title}</b><br/>
        escala fixa: [{vmin_fixed:.2f}, {vmax_fixed:.2f}] {unit}<br/>
        janela visual (p2–p98): [{p2:.2f}, {p98:.2f}] {unit}<br/>
        filtro: NDVI ≤ {NDVI_MACROFITAS_THR:.2f} (remove macrófitas) • pixels zerados: ocultos<br/>
        colormap: {cmap_name} • gamma: {gamma:.2f}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    click = st_folium(m, width=1200, height=700)

    cb_img = make_colorbar_image(vmin=vmin_fixed, vmax=vmax_fixed, cmap_name=cmap_name, label=label_unit)
    st.image(cb_img, use_column_width=False)
    DEFAULT_LAT = -20.02610
    DEFAULT_LON = -44.10684
    st.markdown("---")
    st.markdown("### 📈 Série temporal no ponto clicado")

    # se o usuário clicar, usa o clique; senão usa o ponto padrão
    if click and click.get("last_clicked"):
        lon = click["last_clicked"]["lng"]
        lat = click["last_clicked"]["lat"]
    else:
        lon = DEFAULT_LON
        lat = DEFAULT_LAT

    st.success(f"Coordenada (EPSG:4326): ({lat:.5f}, {lon:.5f})")

    series = []
    for p in water_files:
        dt_str = parse_date_from_filename(p)
        try:
            var_f, _, _, _ = compute_filtered_var_and_indices(p, var_key)
            with rasterio.open(p) as src:
                val = sample_from_array(src, var_f, lon, lat)
            series.append({"Data": dt_str, "Valor": val})
        except Exception:
            series.append({"Data": dt_str, "Valor": np.nan})

    df_ts = pd.DataFrame(series)
    df_ts["Data"] = pd.to_datetime(df_ts["Data"])
    df_ts = df_ts.sort_values("Data")

    fig_ts = px.line(
        df_ts,
        x="Data",
        y="Valor",
        markers=True,
        title=f"Série temporal — {label_unit}",
        labels={"Valor": label_unit}
    )
    fig_ts.update_yaxes(range=[vmin_fixed, vmax_fixed])
    st.plotly_chart(fig_ts, use_container_width=True)

    if show_point_clim:
        st.markdown("### 📆 Climatologia mensal do ponto (média por mês)")
        df_ts["Mes"] = df_ts["Data"].dt.month
        clim_pt = df_ts.groupby("Mes")["Valor"].mean().reset_index()

        fig_clim = px.line(
            clim_pt,
            x="Mes",
            y="Valor",
            markers=True,
            title=f"Climatologia mensal no ponto — {label_unit}",
            labels={"Valor": label_unit, "Mes": "Mês"}
        )
        fig_clim.update_layout(xaxis=dict(dtick=1))
        fig_clim.update_yaxes(range=[vmin_fixed, vmax_fixed])
        st.plotly_chart(fig_clim, use_container_width=True)

    with st.expander("Tabela (série no ponto)"):
        st.dataframe(df_ts, use_container_width=True)
        

    st.markdown("---")
    st.markdown("### 🧪 Diagnóstico (NDVI e NDWI) — data selecionada")

    with st.expander("Ver NDVI e NDWI (mapas)"):
        cA, cB = st.columns(2)
        with cA:
            ndvi_u8, ndvi_min, ndvi_max = normalize_to_uint8_diag(ndvi_A)
            st.caption(f"NDVI • escala [{ndvi_min:.3f}, {ndvi_max:.3f}]")
            st.image(colormap_rgba(ndvi_u8, "viridis"), use_column_width=True)
        with cB:
            ndwi_u8, ndwi_min, ndwi_max = normalize_to_uint8_diag(ndwi_A)
            st.caption(f"NDWI • escala [{ndwi_min:.3f}, {ndwi_max:.3f}]")
            st.image(colormap_rgba(ndwi_u8, "cividis"), use_column_width=True)

    st.caption(
        "Qualidade da Água • filtro: NDVI ≤ 0.5 (remove macrófitas). "
        "Pixels zerados ocultos. NDWI exibido apenas para diagnóstico."
    )

# =====================================================================
# PÁGINA 4 — SEDIMENTOS, BATIMETRIA & RISCO
# =====================================================================
elif pagina == "🧱 Sedimentos & Risco":
    page_sedimentos_risco()
