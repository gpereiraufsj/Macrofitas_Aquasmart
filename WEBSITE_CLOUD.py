# Requisitos:
# pip install streamlit rasterio numpy pandas plotly folium geopandas streamlit-folium pillow

import streamlit as st
import rasterio
import folium
import os
import numpy as np
import pandas as pd
from pathlib import Path
from rasterio.transform import rowcol
from streamlit_folium import st_folium
import plotly.express as px
from folium import raster_layers
from PIL import Image

# =====================================================================
# CONFIGURAÇÃO (PORTÁVEL: Windows + Streamlit Cloud)
# =====================================================================
APP_DIR = Path(__file__).resolve().parent

# ✅ Recomendo organizar assim no repositório:
# data/area_macrofitas.csv
# data/classificados/*.tif
# output_vis/fig_macrofitas_YYYY-MM-DD.png
# assets/logo.png

DATA_DIR = APP_DIR / "data"
CLASSIF_DIR = DATA_DIR / "classificados"
OUTPUT_VIS_DIR = APP_DIR / "output_vis"
ASSETS_DIR = APP_DIR / "assets"

csv_path = DATA_DIR / "area_macrofitas.csv"
logo_path = ASSETS_DIR / "logo.png"

st.set_page_config(layout="wide", page_title="AQUASMART • Dashboard Científico")

# =====================================================================
# SIDEBAR • LOGO + NAVEGAÇÃO
# =====================================================================
with st.sidebar:
    # Logo com fallback (não derruba o app)
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=True)
    else:
        st.markdown("### AQUASMART")
        st.caption("Logo não encontrado em: assets/logo.png")

    st.markdown("## Dashboard")
    st.caption("Monitoramento • Macrófitas e Qualidade da Água")

    pagina = st.radio(
        "Navegação",
        ["🌿 Macrófitas", "💧 Qualidade da Água"],
        index=0
    )
    st.markdown("---")

# =====================================================================
# HEADER PRINCIPAL
# =====================================================================
st.markdown("# AQUASMART • Dashboard Científico")
st.caption("Série temporal, mapa interativo e análises em hectares (ha).")

# =====================================================================
# PÁGINA 1 — MACRÓFITAS
# =====================================================================
if pagina == "🌿 Macrófitas":

    st.markdown("## 🌿 Monitoramento de Macrófitas")

    # ---------------------------
    # CARREGAR CSV (com validação)
    # ---------------------------
    if not csv_path.exists():
        st.error(f"CSV não encontrado: {csv_path}")
        st.info("Coloque o arquivo em `data/area_macrofitas.csv` no seu repositório.")
        st.stop()

    df_area = pd.read_csv(csv_path)
    if "Data" not in df_area.columns or "Area_m2" not in df_area.columns:
        st.error("O CSV precisa conter as colunas: 'Data' e 'Area_m2'.")
        st.stop()

    df_area["Data"] = pd.to_datetime(df_area["Data"])

    # Converter m² -> ha
    df_area["Area_ha"] = df_area["Area_m2"] / 10_000
    if "Area_smooth" in df_area.columns:
        df_area["Area_smooth_ha"] = df_area["Area_smooth"] / 10_000

    min_date, max_date = df_area["Data"].min(), df_area["Data"].max()

    # ---------------------------
    # FILTRO DE DATA (SIDEBAR)
    # ---------------------------
    with st.sidebar:
        st.markdown("### Filtros • Macrófitas")
        start_date, end_date = st.date_input(
            "📆 Intervalo de datas",
            [min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

    filtradas = df_area[
        (df_area["Data"] >= pd.to_datetime(start_date)) &
        (df_area["Data"] <= pd.to_datetime(end_date))
    ].copy()

    # ---------------------------
    # KPIs
    # ---------------------------
    st.markdown("### Indicadores do período selecionado")

    if len(filtradas) == 0:
        st.warning("Nenhum dado no intervalo selecionado.")
        st.stop()

    total_ha = filtradas["Area_ha"].sum()
    max_ha = filtradas["Area_ha"].max()
    data_max = filtradas.loc[filtradas["Area_ha"].idxmax(), "Data"].strftime("%Y-%m-%d")
    mean_ha = filtradas.groupby(filtradas["Data"].dt.year)["Area_ha"].mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("🌱 Área Total", f"{total_ha:,.2f} ha")
    c2.metric("📈 Máxima", f"{max_ha:,.2f} ha", data_max)
    c3.metric("📊 Média Anual", f"{mean_ha.mean():,.2f} ha")

    st.markdown("---")

    # ---------------------------
    # SÉRIE TEMPORAL (ha)
    # ---------------------------
    st.markdown("### Série temporal da área (ha)")

    y_cols = ["Area_ha"]
    if "Area_smooth_ha" in filtradas.columns:
        y_cols.append("Area_smooth_ha")

    fig_area = px.line(
        filtradas,
        x="Data",
        y=y_cols,
        markers=True,
        labels={"value": "Área (ha)", "variable": "Tipo"},
        title="Evolução da Área de Macrófitas (ha)"
    )
    st.plotly_chart(fig_area, use_container_width=True)

    st.markdown("---")

    # ---------------------------
    # LISTAR TIFs CLASSIFICADOS (com validação)
    # ---------------------------
    if not CLASSIF_DIR.exists():
        st.error(f"Pasta de TIFs não encontrada: {CLASSIF_DIR}")
        st.info("Coloque os arquivos .tif em `data/classificados/` no repositório.")
        st.stop()

    classif_files = sorted([p for p in CLASSIF_DIR.glob("classificado_macrofitas_*.tif")])
    if len(classif_files) == 0:
        st.error("Nenhum arquivo encontrado com padrão: classificado_macrofitas_*.tif")
        st.stop()

    # Extrair datas do nome
    dates = [p.stem.replace("classificado_macrofitas_", "") for p in classif_files]

    selected_date = st.selectbox("📅 Selecione a data da imagem:", dates, index=len(dates) - 1)
    file_selected = CLASSIF_DIR / f"classificado_macrofitas_{selected_date}.tif"

    # ---------------------------
    # MAPA + PONTO
    # ---------------------------
    col_mapa, col_grafico = st.columns([1, 1], gap="large")

    with col_mapa:
        st.markdown("### 🗺️ Mapa classificado (clique para amostrar)")

        with rasterio.open(file_selected) as src:
            img = src.read(1)
            bounds = src.bounds

        # ATENÇÃO: isto assume que bounds estão em lat/lon (EPSG:4326).
        # Se seus TIFs estiverem em UTM (ex.: SIRGAS2000 / UTM), o folium ficará errado.
        # Se esse for o seu caso, me diga o EPSG e eu corrijo a reprojeção.
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
        click_data = st_folium(m, width=650, height=470)

    with col_grafico:
        st.markdown("### 📌 Série no ponto clicado")

        if click_data and click_data.get("last_clicked"):
            lon = click_data["last_clicked"]["lng"]
            lat = click_data["last_clicked"]["lat"]
            st.success(f"Coordenada: ({lat:.5f}, {lon:.5f})")

            resultados = []
            for p in classif_files:
                dt = p.stem.replace("classificado_macrofitas_", "")

                with rasterio.open(p) as src:
                    try:
                        row, col = rowcol(src.transform, lon, lat)
                        val = src.read(1)[row, col]
                        resultados.append({"Data": dt, "Macrofita": int(val)})
                    except:
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

    st.markdown("---")

    # ---------------------------
    # FIGURA ESTÁTICA (opcional)
    # ---------------------------
    st.markdown("### 📷 RGB | NDVI | Classificação (figura estática)")

    fig_path = OUTPUT_VIS_DIR / f"fig_macrofitas_{selected_date}.png"
    if fig_path.exists():
        st.image(Image.open(fig_path), use_container_width=True)
    else:
        st.warning(f"Figura não encontrada (opcional): {fig_path}")

    st.markdown("---")

    # ---------------------------
    # COMPARAÇÃO ENTRE ANOS (ha)
    # ---------------------------
    st.markdown("### 📅 Comparação entre anos (média mensal em ha)")

    years = sorted(df_area["Data"].dt.year.unique())
    if len(years) < 1:
        st.warning("Sem anos disponíveis no CSV.")
        st.stop()

    default_y1 = min(3, len(years) - 1)
    default_y2 = len(years) - 1

    ycol1, ycol2 = st.columns(2)
    with ycol1:
        year1 = st.selectbox("Ano 1:", years, index=default_y1)
    with ycol2:
        year2 = st.selectbox("Ano 2:", years, index=default_y2)

    # Agrupar por mês dentro do ano (média)
    d1 = df_area[df_area["Data"].dt.year == year1].copy()
    d2 = df_area[df_area["Data"].dt.year == year2].copy()

    df_y1 = d1.groupby(d1["Data"].dt.month)["Area_ha"].mean()
    df_y2 = d2.groupby(d2["Data"].dt.month)["Area_ha"].mean()

    fig_comp = px.line(title=f"Comparação Anual: {year1} vs {year2}")
    fig_comp.add_scatter(x=df_y1.index, y=df_y1.values, name=f"{year1}", mode="lines+markers")
    fig_comp.add_scatter(x=df_y2.index, y=df_y2.values, name=f"{year2}", mode="lines+markers")
    fig_comp.update_layout(xaxis_title="Mês", yaxis_title="Área (ha)")
    st.plotly_chart(fig_comp, use_container_width=True)

# =====================================================================
# PÁGINA 2 — QUALIDADE DA ÁGUA (PLACEHOLDER)
# =====================================================================
else:
    st.markdown("## 💧 Qualidade da Água")
    st.caption("Área reservada para indicadores físico-químicos e biológicos (placeholder).")

    # KPIs placeholder
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Turbidez", "—")
    c2.metric("Clorofila-a", "—")
    c3.metric("OD", "—")
    c4.metric("pH", "—")

    st.markdown("---")
    st.markdown("### Séries temporais (placeholder)")
    st.info("Você vai adicionar aqui os gráficos/indicadores depois.")
    st.empty()

    st.markdown("---")
    st.markdown("### Mapas/estações de amostragem (placeholder)")
    st.empty()

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.caption("AQUASMART • Dashboard científico interativo • Área em hectares (ha)")
