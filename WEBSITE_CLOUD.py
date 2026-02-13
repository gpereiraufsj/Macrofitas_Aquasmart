# Requisitos:
# pip install streamlit rasterio numpy pandas plotly folium geopandas streamlit-folium pillow imageio pyproj matplotlib

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
import pathlib

from pyproj import Transformer
import matplotlib.cm as cm
from io import BytesIO

from shapely.ops import unary_union
from rasterio.features import geometry_mask

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
        ["🌿 Macrófitas", "💧 Qualidade da Água"],
        index=0
    )

# =====================================================================
# HEADER
# =====================================================================
st.markdown("# AQUASMART • Dashboard Científico")
st.caption("Macrófitas (ha) • Qualidade da Água (exemplos com máscara água + filtro NDVI)")
st.markdown("---")

# =====================================================================
# FUNÇÕES AUXILIARES — QUALIDADE DA ÁGUA
# =====================================================================
EPS = 1e-6

def list_water_files(folder: pathlib.Path) -> list[pathlib.Path]:
    return sorted([p for p in folder.glob("DATA_*.tif")])

def parse_date_from_filename(p: pathlib.Path) -> str:
    return p.stem.replace("DATA_", "")

def bounds_3857_to_4326(bounds):
    """Converte bounds (left,bottom,right,top) de EPSG:3857 -> EPSG:4326 para usar no Folium."""
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
    lon1, lat1 = transformer.transform(left, bottom)
    lon2, lat2 = transformer.transform(right, top)
    # Folium quer [[southWest],[northEast]]
    return [[lat1, lon1], [lat2, lon2]]

def get_transformer_to_raster(raster_crs):
    if raster_crs is None:
        return None
    epsg = raster_crs.to_epsg() if hasattr(raster_crs, "to_epsg") else None
    if epsg == 4326:
        return None
    return Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)

def read_band(src, idx):
    return src.read(idx).astype("float32")

def compute_ndvi(B, G, R, NIR):
    return (NIR - R) / (NIR + R + EPS)

def compute_ndwi(G, NIR):
    # NDWI (McFeeters) adaptado: (G - NIR)/(G + NIR)
    return (G - NIR) / (G + NIR + EPS)

def compute_masks(B, G, R, NIR, ndwi_thr: float, ndvi_veg_thr: float):
    """
    Água: NDWI > ndwi_thr
    Vegetação/macrófita: NDVI > ndvi_veg_thr
    Manter: água E NÃO vegetação
    """
    ndvi = compute_ndvi(B, G, R, NIR)
    ndwi = compute_ndwi(G, NIR)

    water_mask = ndwi > ndwi_thr
    veg_mask = ndvi > ndvi_veg_thr

    valid_mask = water_mask & (~veg_mask)
    return valid_mask, ndvi, ndwi

def compute_water_variable(B, G, R, NIR, var_key: str):
    """
    Equações genéricas (exemplo). Trocar depois pelos seus algoritmos.
    """
    if var_key == "chlor_a":
        # Exemplo: proxy baseado em NIR/R
        out = (NIR / (R + EPS))
    elif var_key == "phycocyanin":
        # Exemplo: proxy baseado em R/G
        out = (R / (G + EPS))
    elif var_key == "turbidity":
        # Exemplo: proxy baseado em R/(B+G)
        out = R / (B + G + EPS)
    elif var_key == "secchi":
        # Exemplo: inverso da turbidez (proxy)
        turb = R / (B + G + EPS)
        out = 1.0 / (turb + EPS)
    else:
        raise ValueError("Variável desconhecida.")
    return out

def normalize_to_uint8(a, vmin=None, vmax=None):
    a = a.copy()
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
    return (x * 255).astype(np.uint8), float(vmin), float(vmax)

def colormap_rgba(uint8_img, cmap_name="viridis"):
    cmap = cm.get_cmap(cmap_name)
    x = uint8_img.astype("float32") / 255.0
    rgba = (cmap(x) * 255).astype(np.uint8)

    # Primeira cor (valor mínimo da escala) transparente
    rgba[uint8_img == 0, 3] = 0

    return rgba

def sample_from_precomputed_array(src, arr, lon, lat):
    """Amostra valor de 'arr' (mesma grade do raster) no ponto clicado."""
    transformer = get_transformer_to_raster(src.crs)
    if transformer:
        x, y = transformer.transform(lon, lat)
    else:
        x, y = lon, lat

    r, c = rowcol(src.transform, x, y)
    if r < 0 or c < 0 or r >= arr.shape[0] or c >= arr.shape[1]:
        return np.nan
    val = arr[r, c]
    return float(val) if np.isfinite(val) else np.nan

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def make_colorbar_image(vmin: float, vmax: float, cmap_name: str, label: str = "") -> Image.Image:
    """Gera uma colorbar (PNG) como PIL Image para exibir no Streamlit."""
    fig, ax = plt.subplots(figsize=(5.0, 0.7))
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

# =====================================================================
# PÁGINA 1 — MACRÓFITAS (mantida igual)
# =====================================================================
if pagina == "🌿 Macrófitas":

    st.subheader("🌿 Monitoramento de Macrófitas")

    df_area = pd.read_csv(csv_path)
    df_area["Data"] = pd.to_datetime(df_area["Data"])

    df_area["Area_ha"] = df_area["Area_m2"] / 10_000
    if "Area_smooth" in df_area.columns:
        df_area["Area_smooth_ha"] = df_area["Area_smooth"] / 10_000

    min_date, max_date = df_area["Data"].min(), df_area["Data"].max()

    start_date, end_date = st.sidebar.date_input(
        "📆 Intervalo de datas:", [min_date, max_date], min_value=min_date, max_value=max_date
    )

    st.subheader("📆 Análise Mensal de Área Média")
    df_area["Mês"] = df_area["Data"].dt.month
    mensal = df_area.groupby("Mês").mean(numeric_only=True).reset_index()

    fig_mensal = px.bar(
        mensal, x="Mês", y=mensal["Area_m2"] / 10_000,
        labels={"y": "Área Média (ha)"},
        title="Área Média de Macrófitas por Mês", text_auto=".2s"
    )
    st.plotly_chart(fig_mensal, use_container_width=True)

    st.markdown("---")

    filtradas = df_area[
        (df_area["Data"] >= pd.to_datetime(start_date)) &
        (df_area["Data"] <= pd.to_datetime(end_date))
    ]

    total_ha = filtradas["Area_ha"].sum()
    max_ha = filtradas["Area_ha"].max()
    data_max = filtradas.loc[filtradas["Area_ha"].idxmax(), "Data"].strftime("%Y-%m-%d")
    mean_ha = filtradas.groupby(filtradas["Data"].dt.year)["Area_ha"].mean()

    st.markdown("### 📌 Estatísticas do Período Selecionado")
    col1, col2, col3 = st.columns(3)
    col1.metric("🌱 Área Total", f"{total_ha:,.2f} ha")
    col2.metric("📈 Máxima", f"{max_ha:,.2f} ha", data_max)
    col3.metric("📊 Média Anual", f"{mean_ha.mean():,.2f} ha")

    fig_area = px.line(
        filtradas,
        x="Data",
        y=["Area_ha", "Area_smooth_ha"] if "Area_smooth_ha" in filtradas.columns else ["Area_ha"],
        markers=True,
        labels={"value": "Área (ha)", "variable": "Tipo"},
        title="Evolução da Área de Macrófitas (ha)"
    )
    st.plotly_chart(fig_area, use_container_width=True)

    classif_files = sorted([f for f in os.listdir(classif_folder) if f.startswith("classificado_macrofitas") and f.endswith(".tif")])
    dates = [f.replace("classificado_macrofitas_", "").replace(".tif", "") for f in classif_files]

    selected_date = st.selectbox("📅 Selecione a data da imagem:", dates)
    file_selected = os.path.join(classif_folder, f"classificado_macrofitas_{selected_date}.tif")

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

    st.subheader("📷 Visualização: RGB | NDVI | Classificação")
    fig_path = os.path.join(output_vis_folder, f"fig_macrofitas_{selected_date}.png")
    if os.path.exists(fig_path):
        st.image(Image.open(fig_path), use_column_width=True)
    else:
        st.warning(f"Imagem não encontrada: {fig_path}")

    st.subheader("📅 Comparação entre Anos")
    years = sorted(df_area["Data"].dt.year.unique())
    year1 = st.selectbox("Ano 1:", years, index=3 if len(years) > 3 else 0)
    year2 = st.selectbox("Ano 2:", years, index=len(years)-1)

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
# PÁGINA 2 — QUALIDADE DA ÁGUA (NDVI>0.5 remove macrófitas; NDWI só diagnóstico)
# =====================================================================
else:
    st.subheader("💧 Qualidade da Água")
    st.caption("Derivado de DATA_*.tif (EPSG:3857) • filtro: remove macrófitas onde NDVI > 0.5 • NDWI apenas diagnóstico.")

    # ----------------------------
    # Parâmetros fixos
    # ----------------------------
    NDVI_MACROFITAS_THR = 0.50  # remove macrófitas/vegetação aquática

    water_files = list_water_files(base_path)
    if len(water_files) == 0:
        st.warning("Nenhum arquivo encontrado com padrão DATA_*.tif na raiz do repositório.")
        st.stop()

    water_dates = [parse_date_from_filename(p) for p in water_files]

    VAR_SPECS = {
        "chlor_a": {
            "label": "Clorofila-a",
            "unit": "µg/L",
            "vmin": 15.0,
            "vmax": 140.0,
        },
        "turbidity": {
            "label": "Turbidez",
            "unit": "NTU",
            "vmin": 2.5,
            "vmax": 20.0,
        },
        "phycocyanin": {
            "label": "Fitocianina",
            "unit": "µg/L",
            "vmin": 2.5,
            "vmax": 22.0,
        },
        "secchi": {
            "label": "Secchi",
            "unit": "cm",
            "vmin": 20.0,
            "vmax": 100.0,
        },
    }

    
    var_map = {
        "Clorofila-a (proxy)": "chlor_a",
        "Fitocianina (proxy)": "phycocyanin",
        "Turbidez (proxy)": "turbidity",
        "Secchi (proxy)": "secchi",
    }

    # ----------------------------
    # Controles
    # ----------------------------
    c1, c2, c3, c4 = st.columns([1.4, 1.4, 1.0, 1.2])
    with c1:
        var_label = st.selectbox("Variável:", list(var_map.keys()), index=0)
    with c2:
        selected_date = st.selectbox("Data (imagem):", water_dates, index=len(water_dates) - 1)
    with c3:
        cmap_name = st.selectbox("Colormap:", ["viridis", "cividis", "plasma", "inferno", "magma"], index=0)
    with c4:
        compare_mode = st.checkbox("Comparar duas datas", value=False)

    var_key = var_map[var_label]
    spec = VAR_SPECS[var_key]
    unit = spec["unit"]
    vmin_fixed = spec["vmin"]
    vmax_fixed = spec["vmax"]
    
    var_label_unit = f"{spec['label']} ({unit})"
   
    tif_path = base_path / f"DATA_{selected_date}.tif"

    # Se comparar: escolher segunda data + tipo
    date_b = None
    diff_type = "Diferença (B - A)"
    if compare_mode:
        cc1, cc2 = st.columns([1.4, 1.6])
        with cc1:
            date_b = st.selectbox("Data B:", water_dates, index=len(water_dates) - 1)
        with cc2:
            diff_type = st.selectbox("Produto:", ["Diferença (B - A)", "Variação % ((B-A)/A)"], index=0)

    # =================================================================
    # Ler A (e B se necessário), aplicar filtro NDVI e computar variável
    # =================================================================
    def compute_filtered_var_and_indices(tif_file: pathlib.Path):
        with rasterio.open(tif_file) as src:
            if src.count < 4:
                raise ValueError("DATA_*.tif precisa ter 4 bandas (B, G, R, NIR).")

            B = read_band(src, 1)
            G = read_band(src, 2)
            R = read_band(src, 3)
            NIR = read_band(src, 4)

            ndvi = compute_ndvi(B, G, R, NIR)
            ndwi = compute_ndwi(G, NIR)

            # ----------------------------
            # Máscara "sem dado" / zerado
            # ----------------------------
            # Regra prática: se TODAS as bandas são 0 no pixel, é área fora/sem informação → ignora.
            nodata0_mask = (B == 0) & (G == 0) & (R == 0) & (NIR == 0)
            
            # (opcional) também pode eliminar pixels com qualquer banda 0, se seu produto usa 0 como borda:
            # nodata0_mask = nodata0_mask | (B == 0) | (G == 0) | (R == 0) | (NIR == 0)
            
            # ----------------------------
            # Filtro macrófitas (NDVI)
            # ----------------------------
            valid_mask = (
                np.isfinite(ndvi) &
                (ndvi <= NDVI_MACROFITAS_THR) &
                (~nodata0_mask)
            )
            
            var_raw = compute_water_variable(B, G, R, NIR, var_key)
            
            # Também remove zeros na variável (caso a equação gere 0 em bordas)
            var_raw = np.where(var_raw == 0, np.nan, var_raw)
            
            var_filt = np.where(valid_mask, var_raw, np.nan)


            folium_bounds = bounds_3857_to_4326(src.bounds)

            meta = {
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "folium_bounds": folium_bounds,
            }
            return var_filt, ndvi, ndwi, meta

    try:
        var_A, ndvi_A, ndwi_A, meta_A = compute_filtered_var_and_indices(tif_path)
    except Exception as e:
        st.error(f"Erro ao processar {tif_path.name}: {e}")
        st.stop()

    # Se comparar:
    map_arr = var_A
    map_title = f"{var_label_unit} • {selected_date}"
    if compare_mode and date_b:
        tif_path_B = base_path / f"DATA_{date_b}.tif"
        try:
            var_B, ndvi_B, ndwi_B, meta_B = compute_filtered_var_and_indices(tif_path_B)
        except Exception as e:
            st.error(f"Erro ao processar DATA_{date_b}.tif: {e}")
            st.stop()

        # Assume mesma grade/extent (se não for, precisa reamostrar)
        if var_B.shape != var_A.shape:
            st.error("As imagens A e B têm shapes diferentes. Para comparar, precisam estar na mesma grade.")
            st.stop()

        if diff_type == "Diferença (B - A)":
            map_arr = var_B - var_A
            map_title = f"{var_label} • Diferença: {date_b} - {selected_date}"
        else:
            map_arr = (var_B - var_A) / (var_A + EPS) * 100.0
            map_title = f"{var_label} • Variação %: {date_b} vs {selected_date}"

    # =================================================================
    # Estatística espacial (somente pixels válidos = NDVI<=0.5)
    # =================================================================
    vals = map_arr[np.isfinite(map_arr)]
    if vals.size == 0:
        st.warning("Após o filtro NDVI (removendo macrófitas), não sobraram pixels válidos para mapear.")
        st.stop()

    stats = {
        "n_pixels": int(vals.size),
        "média": float(np.nanmean(vals)),
        "mediana": float(np.nanmedian(vals)),
        "p10": float(np.nanpercentile(vals, 10)),
        "p25": float(np.nanpercentile(vals, 25)),
        "p75": float(np.nanpercentile(vals, 75)),
        "p90": float(np.nanpercentile(vals, 90)),
        "mín": float(np.nanmin(vals)),
        "máx": float(np.nanmax(vals)),
    }

    st.markdown("### 📊 Estatística espacial (pixels válidos após filtro NDVI)")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("N pixels válidos", f"{stats['n_pixels']:,}")
    s2.metric("Média", f"{stats['média']:.3f}")
    s3.metric("Mediana", f"{stats['mediana']:.3f}")
    s4.metric("p10–p90", f"{stats['p10']:.3f} – {stats['p90']:.3f}")

    # =================================================================
    # Mapa grande + escala
    # =================================================================
    st.markdown("### 🗺️ Mapa interativo (zoom pela extensão do GeoTIFF)")

    # normalização automática para exibição
    img_u8, vmin, vmax = normalize_to_uint8(map_arr, vmin=vmin_fixed, vmax=vmax_fixed)
    rgba = colormap_rgba(img_u8, cmap_name=cmap_name)

    folium_bounds = meta_A["folium_bounds"]
    center_lat = (folium_bounds[0][0] + folium_bounds[1][0]) / 2
    center_lon = (folium_bounds[0][1] + folium_bounds[1][1]) / 2

    m = folium.Map(location=[center_lat, center_lon], tiles="OpenStreetMap", zoom_control=True)

    raster_layers.ImageOverlay(
        image=rgba,
        bounds=folium_bounds,
        opacity=0.85,
        interactive=True,
        zindex=1
    ).add_to(m)

    # Zoom real baseado nos bounds do raster
    m.fit_bounds(folium_bounds)

    # Legenda + escala (min/max)
    legend_html = f"""
    <div style="
        position: fixed; bottom: 30px; left: 30px; width: 320px; z-index: 9999;
        background-color: white; padding: 10px; border: 1px solid #999; border-radius: 6px;
        font-size: 12px;">
        <b>{map_title}</b><br/>
        escala: [{vmin:.2f}, {vmax:.2f}] {unit}<br/>
        filtro: NDVI ≤ {NDVI_MACROFITAS_THR:.2f} (remove macrófitas)<br/>
        colormap: {cmap_name}<br/>
        <span style="color:#666;">(equações genéricas)</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    click = st_folium(m, width=1200, height=700)

    # Colorbar real (escala visual)
    cb_img = make_colorbar_image(vmin=vmin, vmax=vmax, cmap_name=cmap_name, label=var_label_unit)
    st.image(cb_img, use_column_width=False)

    st.markdown("---")

    # =================================================================
    # Série temporal no ponto + curva sazonal (climatologia mensal)
    # =================================================================
    st.markdown("### 📈 Série temporal no ponto clicado (após filtro NDVI)")
    if click and click.get("last_clicked"):
        lon = click["last_clicked"]["lng"]
        lat = click["last_clicked"]["lat"]
        st.success(f"Coordenada (EPSG:4326): ({lat:.5f}, {lon:.5f})")

        series = []
        for p in water_files:
            dt = parse_date_from_filename(p)
            try:
                var_f, ndvi_f, ndwi_f, meta_f = compute_filtered_var_and_indices(p)
                with rasterio.open(p) as src:
                    val = sample_from_precomputed_array(src, var_f, lon, lat)
                series.append({"Data": dt, "Valor": val})
            except:
                series.append({"Data": dt, "Valor": np.nan})

        df_ts = pd.DataFrame(series)
        df_ts["Data"] = pd.to_datetime(df_ts["Data"])
        df_ts = df_ts.sort_values("Data")

        fig_ts = px.line(
            df_ts, x="Data", y="Valor", markers=True,
            title=f"Série temporal — {var_label_unit} (NDVI ≤ {NDVI_MACROFITAS_THR})",
            labels={"Valor": var_label_unit}
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Curva sazonal (climatologia mensal no ponto)
        st.markdown("### 📆 Curva sazonal (média por mês no ponto)")
        df_ts["Mês"] = df_ts["Data"].dt.month
        clim = df_ts.groupby("Mês")["Valor"].mean(numeric_only=True).reset_index()

        fig_clim = px.line(
            clim, x="Mês", y="Valor", markers=True,
            title=f"Série temporal — {var_label_unit} (NDVI ≤ {NDVI_MACROFITAS_THR})",
            labels={"Valor": var_label_unit}
        )
        fig_clim.update_layout(xaxis=dict(dtick=1))
        st.plotly_chart(fig_clim, use_container_width=True)

        with st.expander("Tabela (série no ponto)"):
            st.dataframe(df_ts, use_container_width=True)
    else:
        st.info("Clique em um ponto no mapa para extrair a série temporal e a curva sazonal.")

    st.markdown("---")

    # =================================================================
    # NDVI e NDWI ao final (diagnóstico)
    # =================================================================
    st.markdown("### 🧪 Diagnóstico (NDVI e NDWI) — data selecionada")

    with st.expander("Ver NDVI e NDWI (mapas)"):
        cA, cB = st.columns(2)
        with cA:
            ndvi_u8, ndvi_min, ndvi_max = normalize_to_uint8(ndvi_A)
            st.caption(f"NDVI • escala [{ndvi_min:.3f}, {ndvi_max:.3f}]")
            st.image(colormap_rgba(ndvi_u8, "viridis"), use_column_width=True)
        with cB:
            ndwi_u8, ndwi_min, ndwi_max = normalize_to_uint8(ndwi_A)
            st.caption(f"NDWI • escala [{ndwi_min:.3f}, {ndwi_max:.3f}]")
            st.image(colormap_rgba(ndwi_u8, "cividis"), use_column_width=True)

    st.caption("Qualidade da Água • filtro: NDVI ≤ 0.5 (remove macrófitas). NDWI exibido apenas para diagnóstico.")

















