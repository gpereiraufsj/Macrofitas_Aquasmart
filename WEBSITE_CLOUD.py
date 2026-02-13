# Requisitos:
# pip install streamlit rasterio numpy pandas plotly folium geopandas streamlit-folium pillow

import streamlit as st
import rasterio
import folium
import os
import numpy as np
import pandas as pd
from rasterio.transform import rowcol
from streamlit_folium import st_folium
import plotly.express as px
from folium import raster_layers
from PIL import Image

# =====================================================================
# CONFIGURAÇÃO INICIAL
# =====================================================================
classif_folder = "E:/UFMG/AQUASMART/BANCO_IMAGENS/saida_SIRGAS2000"
output_vis_folder = "E:/UFMG/AQUASMART/BANCO_IMAGENS/output_vis"
csv_path = os.path.join(classif_folder, "area_macrofitas.csv")
logo_path = "E:/UFMG/AQUASMART/WEBPAGE/Logo Aquasmart - RGB 02.png"

st.set_page_config(layout="wide", page_title="AQUASMART • Dashboard Científico")

# =====================================================================
# SIDEBAR • NAVEGAÇÃO
# =====================================================================
with st.sidebar:
    st.image(logo_path, use_container_width=True)
    st.markdown("## AQUASMART")
    st.caption("Dashboard científico • Monitoramento")

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

    # ---------------------------
    # CARREGAR CSV E PREPARAR (ha)
    # ---------------------------
    df_area = pd.read_csv(csv_path)
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
    total_ha = filtradas["Area_ha"].sum()
    max_ha = filtradas["Area_ha"].max()
    data_max = (
        filtradas.loc[filtradas["Area_ha"].idxmax(), "Data"].strftime("%Y-%m-%d")
        if len(filtradas) else "-"
    )
    mean_ha = filtradas.groupby(filtradas["Data"].dt.year)["Area_ha"].mean() if len(filtradas) else pd.Series(dtype=float)

    st.markdown("## 🌿 Monitoramento de Macrófitas")
    st.markdown("### Indicadores do período selecionado")

    c1, c2, c3 = st.columns(3)
    c1.metric("🌱 Área Total", f"{total_ha:,.2f} ha")
    c2.metric("📈 Máxima", f"{max_ha:,.2f} ha" if len(filtradas) else "-", data_max if len(filtradas) else None)
    c3.metric("📊 Média Anual", f"{mean_ha.mean():,.2f} ha" if len(mean_ha) else "-")

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
    # SELEÇÃO DE DATA (MAPA)
    # ---------------------------
    classif_files = sorted([
        f for f in os.listdir(classif_folder)
        if f.startswith("classificado_macrofitas") and f.endswith(".tif")
    ])

    dates = [
        f.replace("classificado_macrofitas_", "").replace(".tif", "")
        for f in classif_files
    ]

    selected_date = st.selectbox("📅 Selecione a data da imagem:", dates)
    file_selected = os.path.join(classif_folder, f"classificado_macrofitas_{selected_date}.tif")

    # ---------------------------
    # MAPA + PONTO (lado a lado)
    # ---------------------------
    col_mapa, col_grafico = st.columns([1, 1], gap="large")

    with col_mapa:
        st.markdown("### 🗺️ Mapa classificado (clique para amostrar)")
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
        click_data = st_folium(m, width=650, height=470)

    with col_grafico:
        st.markdown("### 📌 Série no ponto clicado")
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
    # FIGURA ESTÁTICA
    # ---------------------------
    st.markdown("### 📷 RGB | NDVI | Classificação (figura estática)")
    fig_path = os.path.join(output_vis_folder, f"fig_macrofitas_{selected_date}.png")
    if os.path.exists(fig_path):
        st.image(Image.open(fig_path), use_container_width=True)
    else:
        st.warning(f"Imagem não encontrada: {fig_path}")

    st.markdown("---")

    # ---------------------------
    # COMPARAÇÃO ENTRE ANOS (ha)
    # ---------------------------
    st.markdown("### 📅 Comparação entre anos (média mensal em ha)")
    years = sorted(df_area["Data"].dt.year.unique())

    # Índices seguros
    default_y1 = 0 if len(years) == 0 else min(3, len(years) - 1)
    default_y2 = 0 if len(years) == 0 else (len(years) - 1)

    ycol1, ycol2 = st.columns(2)
    with ycol1:
        year1 = st.selectbox("Ano 1:", years, index=default_y1)
    with ycol2:
        year2 = st.selectbox("Ano 2:", years, index=default_y2)

    df_y1 = df_area[df_area["Data"].dt.year == year1].groupby(df_area["Data"].dt.month).mean(numeric_only=True)
    df_y2 = df_area[df_area["Data"].dt.year == year2].groupby(df_area["Data"].dt.month).mean(numeric_only=True)

    # Garantir uso de ha na comparação
    if "Area_ha" not in df_y1.columns:
        df_y1["Area_ha"] = df_y1["Area_m2"] / 10_000
    if "Area_ha" not in df_y2.columns:
        df_y2["Area_ha"] = df_y2["Area_m2"] / 10_000

    fig_comp = px.line(title=f"Comparação Anual: {year1} vs {year2}")
    fig_comp.add_scatter(x=df_y1.index, y=df_y1["Area_ha"], name=f"{year1}", mode="lines+markers")
    fig_comp.add_scatter(x=df_y2.index, y=df_y2["Area_ha"], name=f"{year2}", mode="lines+markers")
    fig_comp.update_layout(xaxis_title="Mês", yaxis_title="Área (ha)")
    st.plotly_chart(fig_comp, use_container_width=True)

# =====================================================================
# PÁGINA 2 — QUALIDADE DA ÁGUA (PLACEHOLDER)
# =====================================================================
elif pagina == "💧 Qualidade da Água":
    st.markdown("## 💧 Qualidade da Água")
    st.caption("Área reservada para indicadores físico-químicos e biológicos (placeholder).")

    st.info("📌 Aqui você vai inserir os indicadores e gráficos de qualidade da água (ex.: turbidez, clorofila-a, OD, pH, temperatura, condutividade etc.).")

    # Placeholder visual (layout científico)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Turbidez", "—")
    c2.metric("Clorofila-a", "—")
    c3.metric("OD", "—")
    c4.metric("pH", "—")

    st.markdown("---")
    st.markdown("### Séries temporais (placeholder)")
    st.empty()

    st.markdown("---")
    st.markdown("### Mapas/estações de amostragem (placeholder)")
    st.empty()

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.caption("AQUASMART • Dashboard científico interativo • Área em hectares (ha)")
