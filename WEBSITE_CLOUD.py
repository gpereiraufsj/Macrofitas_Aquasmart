# Requisitos: pip install streamlit rasterio numpy pandas plotly folium geopandas streamlit-folium

import streamlit as st
import rasterio
import folium
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.transform import rowcol
from streamlit_folium import st_folium
import plotly.express as px
from folium import raster_layers
from PIL import Image
from io import BytesIO
import base64
import imageio
import pathlib

# =====================================================================
# CONFIGURAÇÃO INICIAL
# =====================================================================
base_path = pathlib.Path(__file__).parent

classif_folder = base_path #/ "saida_SIRGAS2000"
output_vis_folder = base_path #/ "output_vis"
#csv_path = classif_folder / "area_macrofitas.csv"
csv_path = "area_macrofitas.csv"
logo_path = "https://raw.githubusercontent.com/gpereiraufsj/Macrofitas_Aquasmart/main/Logo.png"

st.set_page_config(layout="wide", page_title="Macrofitas Monitoramento")

with st.sidebar:
    st.image(logo_path, use_column_width=True)
    st.title("🌿 Macrofitas")

# =====================================================================
# SELECIONAR INTERVALO DE DATAS
# =====================================================================
df_area = pd.read_csv(csv_path)
df_area['Data'] = pd.to_datetime(df_area['Data'])

# 🔹 Converter para hectares
df_area["Area_ha"] = df_area["Area_m2"] / 10_000
if "Area_smooth" in df_area.columns:
    df_area["Area_smooth_ha"] = df_area["Area_smooth"] / 10_000

min_date, max_date = df_area['Data'].min(), df_area['Data'].max()

start_date, end_date = st.sidebar.date_input(
    "📆 Intervalo de datas:", [min_date, max_date], min_value=min_date, max_value=max_date)

# =====================================================================
# COMPARAÇÃO MENSAL
# =====================================================================
st.subheader("📆 Análise Mensal de Área Média")
df_area['Mês'] = df_area['Data'].dt.month
mensal = df_area.groupby('Mês').mean(numeric_only=True).reset_index()

fig_mensal = px.bar(mensal, x="Mês", y=mensal['Area_m2'] / 10000, labels={"y": "Área Média (ha)"},
                    title="Área Média de Macrófitas por Mês", text_auto='.2s')
st.plotly_chart(fig_mensal, use_container_width=True)

st.markdown("---")
st.caption("Versão científica interativa • Desenvolvido com 💚 para o Projeto AQUASMART")



filtradas = df_area[(df_area['Data'] >= pd.to_datetime(start_date)) & (df_area['Data'] <= pd.to_datetime(end_date))]

# =====================================================================
# ESTATÍSTICAS
# =====================================================================
total_ha = filtradas["Area_ha"].sum()
max_ha = filtradas["Area_ha"].max()
data_max = filtradas.loc[filtradas["Area_ha"].idxmax(), "Data"].strftime("%Y-%m-%d")
mean_ha = filtradas.groupby(filtradas['Data'].dt.year)["Area_ha"].mean()

st.markdown("### 📌 Estatísticas do Período Selecionado")
col1, col2, col3 = st.columns(3)
col1.metric("🌱 Área Total", f"{total_ha:,.2f} ha")
col2.metric("📈 Máxima", f"{max_ha:,.2f} ha", data_max)
col3.metric("📊 Média Anual", f"{mean_ha.mean():,.2f} ha")

# =====================================================================
# GRÁFICO TEMPORAL
# =====================================================================
fig_area = px.line(
    filtradas,
    x="Data",
    y=["Area_ha", "Area_smooth_ha"] if "Area_smooth_ha" in filtradas.columns else ["Area_ha"],
    markers=True,
    labels={"value": "Área (ha)", "variable": "Tipo"},
    title="Evolução da Área de Macrófitas (ha)"
)
st.plotly_chart(fig_area, use_container_width=True)

# =====================================================================
# SELECIONAR UMA DATA PARA MAPA
# =====================================================================
classif_files = sorted([f for f in os.listdir(classif_folder) if f.startswith("classificado_macrofitas") and f.endswith(".tif")])
dates = [f.replace("classificado_macrofitas_", "").replace(".tif", "") for f in classif_files]

selected_date = st.selectbox("📅 Selecione a data da imagem:", dates)
file_selected = os.path.join(classif_folder, f"classificado_macrofitas_{selected_date}.tif")

# =====================================================================
# COLUNAS PARA MAPA + GRÁFICO PONTO
# =====================================================================
col_mapa, col_grafico = st.columns([1, 1])

with col_mapa:
    st.subheader("🗺️ Mapa Classificado - Clique para ver a evolução temporal")
    with rasterio.open(file_selected) as src:
        img = src.read(1)
        bounds = src.bounds

    m = folium.Map(location=[(bounds.top + bounds.bottom)/2, (bounds.left + bounds.right)/2], zoom_start=13)
    overlay_img = np.where(img == 1, 255, 0).astype(np.uint8)
    raster_layers.ImageOverlay(
        image=overlay_img,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        colormap=lambda x: (0, 1, 0, x/255),
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
                    row, col = rowcol(src.transform, lon, lat)
                    val = src.read(1)[row, col]
                    resultados.append({"Data": dt, "Macrofita": int(val)})
                except:
                    resultados.append({"Data": dt, "Macrofita": np.nan})

        df_ponto = pd.DataFrame(resultados)
        df_ponto["Data"] = pd.to_datetime(df_ponto["Data"])
        df_ponto = df_ponto.sort_values("Data")

        fig2 = px.line(df_ponto, x="Data", y="Macrofita", markers=True,
                       title="Presença de Macrófitas (1=sim, 0=não)")
        fig2.update_yaxes(dtick=1, range=[-0.1, 1.1])
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Clique em um ponto no mapa para ver a série temporal.")

# =====================================================================
# FIGURAS ESTÁTICAS (RGB/NDVI/CLASSIF)
# =====================================================================
st.subheader("📷 Visualização: RGB | NDVI | Classificação")
fig_path = os.path.join(output_vis_folder, f"fig_macrofitas_{selected_date}.png")
if os.path.exists(fig_path):
    st.image(Image.open(fig_path), use_column_width=True)
else:
    st.warning(f"Imagem não encontrada: {fig_path}")

# =====================================================================
# COMPARAÇÃO ENTRE ANOS
# =====================================================================
st.subheader("📅 Comparação entre Anos")
years = sorted(df_area['Data'].dt.year.unique())
year1 = st.selectbox("Ano 1:", years, index=3)
year2 = st.selectbox("Ano 2:", years, index=len(years)-1)

df_y1 = df_area[df_area['Data'].dt.year == year1].groupby(df_area['Data'].dt.month).mean(numeric_only=True)
df_y2 = df_area[df_area['Data'].dt.year == year2].groupby(df_area['Data'].dt.month).mean(numeric_only=True)

fig_comp = px.line(title=f"Comparação Anual: {year1} vs {year2}")
fig_comp.add_scatter(x=df_y1.index, y=df_y1['Area_ha'], name=f"{year1}", mode="lines+markers")
fig_comp.add_scatter(x=df_y2.index, y=df_y2['Area_ha'], name=f"{year2}", mode="lines+markers")
fig_comp.update_layout(xaxis_title="Mês", yaxis_title="Área (ha)")
st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")
st.caption("Versão científica interativa • Desenvolvido com 💚 para o Projeto AQUASMART")







