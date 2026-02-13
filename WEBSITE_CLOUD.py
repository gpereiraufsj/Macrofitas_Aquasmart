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
    return (x * 255).astype(np.uint8), vmin, vmax

def colormap_rgba(uint8_img, cmap_name="viridis"):
    cmap = cm.get_cmap(cmap_name)
    x = uint8_img.astype("float32") / 255.0
    rgba = (cmap(x) * 255).astype(np.uint8)
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

def load_roi_geometry(shp_path: str | pathlib.Path):
    """Carrega geometria (unificada) do shapefile."""
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("Shapefile vazio.")
    geom = unary_union(gdf.geometry)
    return gdf, geom

def roi_mask_for_raster(src, roi_geom_4326):
    """
    Cria máscara booleana (True dentro do ROI) no grid do raster.
    roi_geom_4326: geometria em EPSG:4326.
    """
    # reprojetar ROI para CRS do raster
    gdf_roi = gpd.GeoDataFrame(geometry=[roi_geom_4326], crs="EPSG:4326").to_crs(src.crs)
    geom_raster_crs = gdf_roi.geometry.iloc[0]

    mask_inside = ~geometry_mask(
        [geom_raster_crs],
        transform=src.transform,
        invert=False,
        out_shape=(src.height, src.width),
        all_touched=False  # True se quiser incluir pixels tocados
    )
    # geometry_mask retorna True fora quando invert=False; por isso negamos
    return mask_inside

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
# PÁGINA 2 — QUALIDADE DA ÁGUA (LAZY: só calcula ao clicar; cache para Cloud)
# =====================================================================
else:
    st.subheader("💧 Qualidade da Água")
    st.caption(
        "DATA_*.tif (EPSG:3857) • remove macrófitas (NDVI > 0.5) • "
        "ROI (F_2024.shp) opcional • NDWI somente diagnóstico • otimizado (lazy + cache)."
    )

    NDVI_MACROFITAS_THR = 0.50

    # ---------- ROI ----------
    roi_path = base_path / "F_2024.shp"
    use_roi = roi_path.exists()
    roi_geom_4326 = None

    if use_roi:
        try:
            _gdf_roi, roi_geom_4326 = load_roi_geometry(roi_path)
            st.success("ROI ativo: F_2024.shp")
        except Exception as e:
            st.warning(f"Falha ao carregar ROI. Continuando sem recorte. Detalhe: {e}")
            use_roi = False
            roi_geom_4326 = None
    else:
        st.info("ROI não encontrado (F_2024.shp). Para recortar, coloque o shapefile na raiz do repositório.")

    # ---------- arquivos ----------
    water_files = list_water_files(base_path)
    if len(water_files) == 0:
        st.warning("Nenhum arquivo encontrado com padrão DATA_*.tif na raiz do repositório.")
        st.stop()
    water_dates = [parse_date_from_filename(p) for p in water_files]

    var_map = {
        "Clorofila-a (proxy)": "chlor_a",
        "Fitocianina (proxy)": "phycocyanin",
        "Turbidez (proxy)": "turbidity",
        "Secchi (proxy)": "secchi",
    }

    # ---------- controles ----------
    c1, c2, c3, c4 = st.columns([1.4, 1.4, 1.0, 1.2])
    with c1:
        var_label = st.selectbox("Variável:", list(var_map.keys()), index=0)
    with c2:
        date_a = st.selectbox("Data A:", water_dates, index=len(water_dates) - 1)
    with c3:
        cmap_name = st.selectbox("Colormap:", ["viridis", "cividis", "plasma", "inferno", "magma"], index=0)
    with c4:
        compare_mode = st.checkbox("Comparar duas datas", value=False)

    var_key = var_map[var_label]
    date_b = None
    diff_type = None

    if compare_mode:
        cc1, cc2 = st.columns([1.2, 1.8])
        with cc1:
            date_b = st.selectbox("Data B:", water_dates, index=len(water_dates) - 1)
        with cc2:
            diff_type = st.selectbox("Produto:", ["Diferença (B - A)", "Variação % ((B-A)/A)"], index=0)

    tif_a = base_path / f"DATA_{date_a}.tif"
    tif_b = base_path / f"DATA_{date_b}.tif" if (compare_mode and date_b) else None

    # ---------- BOTÃO: evita travar na carga ----------
    run = st.button("🚀 Gerar mapa e estatísticas", type="primary")

    # =================================================================
    # Funções locais (ponto / ROI / downsample / stats em blocos)
    # =================================================================
    from shapely.geometry import Point

    def point_in_roi_epsg4326(lon, lat):
        if not use_roi or roi_geom_4326 is None:
            return True
        return roi_geom_4326.contains(Point(lon, lat))

    def read_pixel_4bands(src, lon, lat):
        transformer = get_transformer_to_raster(src.crs)
        if transformer:
            x, y = transformer.transform(lon, lat)
        else:
            x, y = lon, lat
        r, c = rowcol(src.transform, x, y)
        if r < 0 or c < 0 or r >= src.height or c >= src.width:
            return None
        w = rasterio.windows.Window(col_off=c, row_off=r, width=1, height=1)
        pix = src.read([1, 2, 3, 4], window=w).astype("float32").reshape(4)
        if src.nodata is not None and np.any(pix == src.nodata):
            return None
        return pix  # [B,G,R,NIR]

    def compute_var_from_pixel(pix, var_key):
        B, G, R, NIR = pix
        ndvi = (NIR - R) / (NIR + R + EPS)
        ndwi = (G - NIR) / (G + NIR + EPS)
        if (not np.isfinite(ndvi)) or (ndvi > NDVI_MACROFITAS_THR):
            return np.nan, float(ndvi) if np.isfinite(ndvi) else np.nan, float(ndwi) if np.isfinite(ndwi) else np.nan

        if var_key == "chlor_a":
            val = NIR / (R + EPS)
        elif var_key == "phycocyanin":
            val = R / (G + EPS)
        elif var_key == "turbidity":
            val = R / (B + G + EPS)
        elif var_key == "secchi":
            turb = R / (B + G + EPS)
            val = 1.0 / (turb + EPS)
        else:
            val = np.nan

        return float(val) if np.isfinite(val) else np.nan, float(ndvi), float(ndwi)

    @st.cache_data(show_spinner=False)
    def get_meta_cached(tif_path_str: str):
        p = pathlib.Path(tif_path_str)
        with rasterio.open(p) as src:
            return {
                "folium_bounds": bounds_3857_to_4326(src.bounds),
                "crs": str(src.crs),
                "shape": (src.height, src.width),
            }

    @st.cache_data(show_spinner=False)
    def downsample_overlay_cached(tif_path_str: str, var_key: str, cmap_name: str, use_roi_flag: bool, ndvi_thr: float):
        tif_path = pathlib.Path(tif_path_str)
        with rasterio.open(tif_path) as src:
            # downsample para ~1100px
            max_width = 1100
            scale = max(1, int(np.ceil(src.width / max_width)))
            out_w = int(np.ceil(src.width / scale))
            out_h = int(np.ceil(src.height / scale))

            B = src.read(1, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear).astype("float32")
            G = src.read(2, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear).astype("float32")
            R = src.read(3, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear).astype("float32")
            NIR = src.read(4, out_shape=(out_h, out_w), resampling=rasterio.enums.Resampling.bilinear).astype("float32")

            ndvi = (NIR - R) / (NIR + R + EPS)
            ndwi = (G - NIR) / (G + NIR + EPS)

            valid = np.isfinite(ndvi) & (ndvi <= ndvi_thr)

            # ROI no downsample (rápido)
            if use_roi_flag and (roi_geom_4326 is not None):
                gdf_tmp = gpd.GeoDataFrame(geometry=[roi_geom_4326], crs="EPSG:4326").to_crs(src.crs)
                geom_r = gdf_tmp.geometry.iloc[0]
                scale_x = src.width / out_w
                scale_y = src.height / out_h
                new_transform = src.transform * src.transform.scale(scale_x, scale_y)

                roi_inside = ~rasterio.features.geometry_mask(
                    [geom_r],
                    out_shape=(out_h, out_w),
                    transform=new_transform,
                    invert=False,
                    all_touched=False
                )
                valid = valid & roi_inside

            if var_key == "chlor_a":
                var = NIR / (R + EPS)
            elif var_key == "phycocyanin":
                var = R / (G + EPS)
            elif var_key == "turbidity":
                var = R / (B + G + EPS)
            elif var_key == "secchi":
                turb = R / (B + G + EPS)
                var = 1.0 / (turb + EPS)
            else:
                var = np.full_like(R, np.nan, dtype="float32")

            var = np.where(valid, var, np.nan)

            img_u8, vmin, vmax = normalize_to_uint8(var)
            rgba = colormap_rgba(img_u8, cmap_name=cmap_name)

            return rgba, float(vmin), float(vmax), ndvi, ndwi

    @st.cache_data(show_spinner=False)
    def spatial_stats_blockwise_cached(tif_path_str: str, var_key: str, use_roi_flag: bool, ndvi_thr: float):
        tif_path = pathlib.Path(tif_path_str)
        sample = []
        SAMPLE_MAX = 200_000

        with rasterio.open(tif_path) as src:
            geom_r = None
            if use_roi_flag and (roi_geom_4326 is not None):
                gdf_tmp = gpd.GeoDataFrame(geometry=[roi_geom_4326], crs="EPSG:4326").to_crs(src.crs)
                geom_r = gdf_tmp.geometry.iloc[0]

            for _, window in src.block_windows(1):
                B = src.read(1, window=window).astype("float32")
                G = src.read(2, window=window).astype("float32")
                R = src.read(3, window=window).astype("float32")
                NIR = src.read(4, window=window).astype("float32")

                ndvi = (NIR - R) / (NIR + R + EPS)
                valid = np.isfinite(ndvi) & (ndvi <= ndvi_thr)

                if geom_r is not None:
                    roi_inside = ~rasterio.features.geometry_mask(
                        [geom_r],
                        out_shape=(window.height, window.width),
                        transform=rasterio.windows.transform(window, src.transform),
                        invert=False,
                        all_touched=False
                    )
                    valid = valid & roi_inside

                if var_key == "chlor_a":
                    var = NIR / (R + EPS)
                elif var_key == "phycocyanin":
                    var = R / (G + EPS)
                elif var_key == "turbidity":
                    var = R / (B + G + EPS)
                elif var_key == "secchi":
                    turb = R / (B + G + EPS)
                    var = 1.0 / (turb + EPS)
                else:
                    var = np.full_like(R, np.nan, dtype="float32")

                vals = var[valid]
                if vals.size:
                    if len(sample) < SAMPLE_MAX:
                        take = vals
                        if take.size + len(sample) > SAMPLE_MAX:
                            take = np.random.choice(vals, SAMPLE_MAX - len(sample), replace=False)
                        sample.extend(take.astype("float32").tolist())

        if len(sample) == 0:
            return None

        arr = np.array(sample, dtype="float32")
        return {
            "n_amostra": int(arr.size),
            "média": float(np.nanmean(arr)),
            "mediana": float(np.nanmedian(arr)),
            "p10": float(np.nanpercentile(arr, 10)),
            "p25": float(np.nanpercentile(arr, 25)),
            "p75": float(np.nanpercentile(arr, 75)),
            "p90": float(np.nanpercentile(arr, 90)),
            "mín": float(np.nanmin(arr)),
            "máx": float(np.nanmax(arr)),
        }

    # =================================================================
    # Só executa pesado se clicar
    # =================================================================
    if not run:
        st.info("Clique em **Gerar mapa e estatísticas** para processar (evita travar na inicialização).")
        st.stop()

    with st.spinner("Processando... (mapa downsample + estatística em blocos)"):
        rgba_a, vmin, vmax, ndvi_ds, ndwi_ds = downsample_overlay_cached(
            str(tif_a), var_key, cmap_name, use_roi, NDVI_MACROFITAS_THR
        )
        stats_a = spatial_stats_blockwise_cached(str(tif_a), var_key, use_roi, NDVI_MACROFITAS_THR)

        meta = get_meta_cached(str(tif_a))
        folium_bounds = meta["folium_bounds"]

    # =================================================================
    # MAPA
    # =================================================================
    st.markdown("### 🗺️ Mapa interativo (rápido)")
    center_lat = (folium_bounds[0][0] + folium_bounds[1][0]) / 2
    center_lon = (folium_bounds[0][1] + folium_bounds[1][1]) / 2

    m = folium.Map(location=[center_lat, center_lon], tiles="OpenStreetMap", zoom_control=True)
    raster_layers.ImageOverlay(
        image=rgba_a,
        bounds=folium_bounds,
        opacity=0.85,
        interactive=True,
        zindex=1
    ).add_to(m)
    m.fit_bounds(folium_bounds)

    legend_html = f"""
    <div style="
        position: fixed; bottom: 30px; left: 30px; width: 360px; z-index: 9999;
        background-color: white; padding: 10px; border: 1px solid #999; border-radius: 6px;
        font-size: 12px;">
        <b>{var_label} • {date_a}</b><br/>
        escala: [{vmin:.3f}, {vmax:.3f}]<br/>
        filtro: NDVI ≤ {NDVI_MACROFITAS_THR:.2f} (remove macrófitas)<br/>
        ROI: {"ATIVO" if use_roi else "inativo"}<br/>
        colormap: {cmap_name}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    click = st_folium(m, width=1200, height=720)

    cb_img = make_colorbar_image(vmin=vmin, vmax=vmax, cmap_name=cmap_name, label=var_label)
    st.image(cb_img, use_column_width=False)

    # =================================================================
    # ESTATÍSTICA
    # =================================================================
    st.markdown("### 📊 Estatística espacial (amostrada, após filtro NDVI + ROI)")
    if stats_a is None:
        st.warning("Sem pixels válidos após filtro/ROI.")
    else:
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("N amostra", f"{stats_a['n_amostra']:,}")
        s2.metric("Média", f"{stats_a['média']:.3f}")
        s3.metric("Mediana", f"{stats_a['mediana']:.3f}")
        s4.metric("p10–p90", f"{stats_a['p10']:.3f} – {stats_a['p90']:.3f}")
        s5.metric("Mín–Máx", f"{stats_a['mín']:.3f} – {stats_a['máx']:.3f}")

    st.markdown("---")

    # =================================================================
    # SÉRIE TEMPORAL + CURVA SAZONAL
    # =================================================================
    st.markdown("### 📈 Série temporal no ponto (pixel a pixel) + curva sazonal")
    if click and click.get("last_clicked"):
        lon = click["last_clicked"]["lng"]
        lat = click["last_clicked"]["lat"]

        if not point_in_roi_epsg4326(lon, lat):
            st.warning("O ponto clicado está fora do ROI (F_2024.shp). Selecione um ponto dentro.")
        else:
            st.success(f"Coordenada (EPSG:4326): ({lat:.5f}, {lon:.5f})")

            series = []
            for p in water_files:
                dt = parse_date_from_filename(p)
                try:
                    with rasterio.open(p) as src:
                        pix = read_pixel_4bands(src, lon, lat)
                        if pix is None:
                            series.append({"Data": dt, "Valor": np.nan, "NDVI": np.nan, "NDWI": np.nan})
                            continue
                        val, ndvi_p, ndwi_p = compute_var_from_pixel(pix, var_key)
                        series.append({"Data": dt, "Valor": val, "NDVI": ndvi_p, "NDWI": ndwi_p})
                except:
                    series.append({"Data": dt, "Valor": np.nan, "NDVI": np.nan, "NDWI": np.nan})

            df_ts = pd.DataFrame(series)
            df_ts["Data"] = pd.to_datetime(df_ts["Data"])
            df_ts = df_ts.sort_values("Data")

            fig_ts = px.line(
                df_ts, x="Data", y="Valor", markers=True,
                title=f"Série temporal — {var_label} (NDVI ≤ {NDVI_MACROFITAS_THR})",
                labels={"Valor": var_label}
            )
            st.plotly_chart(fig_ts, use_container_width=True)

            if compare_mode and date_b is not None:
                vA = df_ts.loc[df_ts["Data"] == pd.to_datetime(date_a), "Valor"]
                vB = df_ts.loc[df_ts["Data"] == pd.to_datetime(date_b), "Valor"]
                if len(vA) and len(vB):
                    vA = float(vA.values[0]) if np.isfinite(vA.values[0]) else np.nan
                    vB = float(vB.values[0]) if np.isfinite(vB.values[0]) else np.nan
                    if np.isfinite(vA) and np.isfinite(vB):
                        diff = vB - vA
                        varp = (diff / (vA + EPS)) * 100.0
                        cc1, cc2, cc3 = st.columns(3)
                        cc1.metric("Valor A", f"{vA:.4f}", date_a)
                        cc2.metric("Valor B", f"{vB:.4f}", date_b)
                        cc3.metric("Δ / %", f"{diff:.4f} / {varp:.2f}%")

            st.markdown("### 📆 Curva sazonal (média por mês no ponto)")
            df_ts["Mês"] = df_ts["Data"].dt.month
            clim = df_ts.groupby("Mês")["Valor"].mean(numeric_only=True).reset_index()
            fig_clim = px.line(
                clim, x="Mês", y="Valor", markers=True,
                title=f"Climatologia mensal no ponto — {var_label}",
                labels={"Valor": var_label}
            )
            fig_clim.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(fig_clim, use_container_width=True)

            with st.expander("Tabela (série no ponto)"):
                st.dataframe(df_ts, use_container_width=True)
    else:
        st.info("Clique em um ponto no mapa para extrair a série temporal e a curva sazonal.")

    st.markdown("---")

    # =================================================================
    # NDVI/NDWI no final (diagnóstico)
    # =================================================================
    st.markdown("### 🧪 Diagnóstico (NDVI e NDWI) — data selecionada (downsample)")
    with st.expander("Ver NDVI e NDWI (mapas)"):
        cA, cB = st.columns(2)
        with cA:
            ndvi_u8, ndvi_min, ndvi_max = normalize_to_uint8(ndvi_ds)
            st.caption(f"NDVI (downsample) • escala [{ndvi_min:.3f}, {ndvi_max:.3f}]")
            st.image(colormap_rgba(ndvi_u8, "viridis"), use_column_width=True)
        with cB:
            ndwi_u8, ndwi_min, ndwi_max = normalize_to_uint8(ndwi_ds)
            st.caption(f"NDWI (downsample) • escala [{ndwi_min:.3f}, {ndwi_max:.3f}]")
            st.image(colormap_rgba(ndwi_u8, "cividis"), use_column_width=True)















