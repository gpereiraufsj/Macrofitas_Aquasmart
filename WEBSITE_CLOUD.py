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
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# =====================================================================
# BOOTSTRAP-LIKE THEME (Streamlit)
# =====================================================================
BOOTSTRAP_CSS = """
<style>
/* --- container --- */
.bs-container { max-width: 1200px; margin: 0 auto; padding: 0 12px; }

/* --- navbar --- */
.bs-navbar {
  display:flex; align-items:center; justify-content:space-between;
  padding: 10px 14px; margin: 10px 0 16px 0;
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 10px;
  background: rgba(255,255,255,.65);
  backdrop-filter: blur(6px);
}
.bs-navbar .brand { font-weight: 700; font-size: 16px; }
.bs-navbar .tag { font-size: 12px; padding: 4px 8px; border-radius: 999px; background: rgba(0,0,0,.06); }

/* --- grid --- */
.bs-row { display:flex; gap: 12px; flex-wrap: wrap; }
.bs-col { flex: 1 1 0; min-width: 260px; }

/* --- cards --- */
.bs-card {
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 12px;
  padding: 14px;
  background: rgba(255,255,255,.75);
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}
.bs-card h3, .bs-card h4 { margin: 0 0 8px 0; }
.bs-muted { color: rgba(0,0,0,.6); font-size: 12px; }

/* --- metric chips --- */
.bs-metric {
  border: 1px solid rgba(0,0,0,.08);
  border-radius: 10px;
  padding: 10px 12px;
  background: rgba(0,0,0,.03);
}
.bs-metric .k { font-size: 12px; color: rgba(0,0,0,.6); }
.bs-metric .v { font-size: 20px; font-weight: 700; line-height: 1.1; margin-top: 3px; }

/* --- tighten Streamlit spacing a bit --- */
.block-container { padding-top: 0.8rem; padding-bottom: 1.2rem; }
</style>
"""

def bs_container_start():
    st.markdown('<div class="bs-container">', unsafe_allow_html=True)

def bs_container_end():
    st.markdown('</div>', unsafe_allow_html=True)

def bs_navbar(brand: str, tag: str):
    st.markdown(
        f"""
        <div class="bs-navbar">
            <div class="brand">{brand}</div>
            <div class="tag">{tag}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def bs_card_start(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="bs-card">
            <h3>{title}</h3>
            {'<div class="bs-muted">'+subtitle+'</div>' if subtitle else ''}
        """,
        unsafe_allow_html=True
    )

def bs_card_end():
    st.markdown("</div>", unsafe_allow_html=True)

def bs_metric_row(metrics: list[tuple[str, str]]):
    cols_html = ""
    for k, v in metrics:
        cols_html += f"""
        <div class="bs-col">
            <div class="bs-metric">
                <div class="k">{k}</div>
                <div class="v">{v}</div>
            </div>
        </div>
        """
    st.markdown(f'<div class="bs-row">{cols_html}</div>', unsafe_allow_html=True)

# =====================================================================
# CONFIGURAÇÃO INICIAL
# =====================================================================
EPS = 1e-6
base_path = pathlib.Path(__file__).parent
classif_folder = base_path
output_vis_folder = base_path

csv_path = "area_macrofitas.csv"
logo_path = "https://raw.githubusercontent.com/gpereiraufsj/Macrofitas_Aquasmart/main/Logo.png"

st.set_page_config(layout="wide", page_title="AQUASMART • Dashboard Científico")
st.markdown(BOOTSTRAP_CSS, unsafe_allow_html=True)

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
# FUNÇÕES AUXILIARES
# =====================================================================
def list_water_files(folder: pathlib.Path):
    return sorted([p for p in folder.glob("DATA_*.tif")])

def parse_date_from_filename(p: pathlib.Path) -> str:
    return p.stem.replace("DATA_", "")

def bounds_3857_to_4326(bounds):
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
    lon1, lat1 = transformer.transform(left, bottom)
    lon2, lat2 = transformer.transform(right, top)
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

def compute_ndvi(R, NIR):
    return (NIR - R) / (NIR + R + EPS)

def compute_ndwi(G, NIR):
    return (G - NIR) / (G + NIR + EPS)

def normalize_to_uint8_diag(a, vmin=None, vmax=None):
    a = a.astype("float32")
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

def make_colorbar_image(vmin: float, vmax: float, cmap_name: str, label: str = "") -> Image.Image:
    fig, ax = plt.subplots(figsize=(5.2, 0.7))
    fig.subplots_adjust(bottom=0.35, left=0.08, right=0.98, top=0.95)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name)),
        cax=ax, orientation="horizontal"
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
# HEADER (Bootstrap-like)
# =====================================================================
bs_container_start()
bs_navbar("AQUASMART • Dashboard Científico", "Macrófitas (ha) • Qualidade da Água (exemplos)")
bs_container_end()

# =====================================================================
# PÁGINA 1 — MACRÓFITAS (seu código, só “embrulhado” em cards)
# =====================================================================
if pagina == "🌿 Macrófitas":
    bs_container_start()

    bs_card_start("🌿 Monitoramento de Macrófitas", "Série temporal + comparação anual + mapa classificado")
    st.caption("Dados: area_macrofitas.csv • classificação: classificado_macrofitas_*.tif")
    bs_card_end()

    df_area = pd.read_csv(csv_path)
    df_area["Data"] = pd.to_datetime(df_area["Data"])
    df_area["Area_ha"] = df_area["Area_m2"] / 10_000
    if "Area_smooth" in df_area.columns:
        df_area["Area_smooth_ha"] = df_area["Area_smooth"] / 10_000

    min_date, max_date = df_area["Data"].min(), df_area["Data"].max()
    start_date, end_date = st.sidebar.date_input(
        "📆 Intervalo de datas:", [min_date, max_date], min_value=min_date, max_value=max_date
    )

    # Mensal
    bs_card_start("📆 Análise Mensal de Área Média", "Média por mês (ha)")
    df_area["Mês"] = df_area["Data"].dt.month
    mensal = df_area.groupby("Mês").mean(numeric_only=True).reset_index()
    fig_mensal = px.bar(
        mensal, x="Mês", y=mensal["Area_m2"] / 10_000,
        labels={"y": "Área Média (ha)"},
        title="Área Média de Macrófitas por Mês", text_auto=".2s"
    )
    st.plotly_chart(fig_mensal, use_container_width=True)
    bs_card_end()

    filtradas = df_area[
        (df_area["Data"] >= pd.to_datetime(start_date)) &
        (df_area["Data"] <= pd.to_datetime(end_date))
    ]

    total_ha = filtradas["Area_ha"].sum()
    max_ha = filtradas["Area_ha"].max()
    data_max = filtradas.loc[filtradas["Area_ha"].idxmax(), "Data"].strftime("%Y-%m-%d")
    mean_ha = filtradas.groupby(filtradas["Data"].dt.year)["Area_ha"].mean()

    bs_card_start("📌 Estatísticas do Período Selecionado", "")
    bs_metric_row([
        ("🌱 Área Total", f"{total_ha:,.2f} ha"),
        ("📈 Máxima", f"{max_ha:,.2f} ha ({data_max})"),
        ("📊 Média Anual", f"{mean_ha.mean():,.2f} ha"),
    ])
    bs_card_end()

    bs_card_start("📈 Evolução da Área de Macrófitas", "Linha temporal (ha)")
    fig_area = px.line(
        filtradas,
        x="Data",
        y=["Area_ha", "Area_smooth_ha"] if "Area_smooth_ha" in filtradas.columns else ["Area_ha"],
        markers=True,
        labels={"value": "Área (ha)", "variable": "Tipo"},
        title="Evolução da Área de Macrófitas (ha)"
    )
    st.plotly_chart(fig_area, use_container_width=True)
    bs_card_end()

    classif_files = sorted([f for f in os.listdir(classif_folder) if f.startswith("classificado_macrofitas") and f.endswith(".tif")])
    dates = [f.replace("classificado_macrofitas_", "").replace(".tif", "") for f in classif_files]
    selected_date = st.selectbox("📅 Selecione a data da imagem:", dates)
    file_selected = os.path.join(classif_folder, f"classificado_macrofitas_{selected_date}.tif")

    # Mapa + série do ponto
    bs_card_start("🗺️ Mapa Classificado + Série no ponto", "Clique no mapa para extrair presença (0/1)")
    col_mapa, col_grafico = st.columns([1.6, 1.0])
    with col_mapa:
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
        click_data = st_folium(m, width=900, height=520)

    with col_grafico:
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
    bs_card_end()

    # Fig estática
    bs_card_start("📷 Visualização: RGB | NDVI | Classificação", "")
    fig_path = os.path.join(output_vis_folder, f"fig_macrofitas_{selected_date}.png")
    if os.path.exists(fig_path):
        st.image(Image.open(fig_path), use_column_width=True)
    else:
        st.warning(f"Imagem não encontrada: {fig_path}")
    bs_card_end()

    # Comparação anual
    bs_card_start("📅 Comparação entre Anos", "")
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
    bs_card_end()

    st.caption("Versão científica interativa • Desenvolvido com 💚 para o Projeto AQUASMART")
    bs_container_end()

# =====================================================================
# PÁGINA 2 — QUALIDADE DA ÁGUA (seu código já “pronto”, embrulhado em cards)
# =====================================================================
else:
    bs_container_start()

    bs_card_start("💧 Qualidade da Água", "DATA_*.tif (EPSG:3857) • NDVI≤0.5 mantém água sem macrófitas • zeros ocultos • NDWI/NDVI diagnóstico")
    st.caption("Equações/proxies são exemplos; reescala robusta para faixas fixas e mapa com contraste (stretch p2–p98 + gamma).")
    bs_card_end()

    NDVI_MACROFITAS_THR = 0.50

    VAR_SPECS = {
        "chlor_a":     {"label": "Clorofila-a", "unit": "µg/L", "vmin": 15.0, "vmax": 140.0},
        "turbidity":   {"label": "Turbidez",    "unit": "NTU",  "vmin": 2.5,  "vmax": 20.0},
        "phycocyanin": {"label": "Fitocianina", "unit": "µg/L", "vmin": 2.5,  "vmax": 22.0},
        "secchi":      {"label": "Secchi",      "unit": "cm",   "vmin": 20.0, "vmax": 100.0},
    }

    def compute_water_variable_scaled(B, G, R, NIR, var_key: str):
        spec = VAR_SPECS[var_key]
        vmin, vmax = float(spec["vmin"]), float(spec["vmax"])

        def _robust_scale_to_range(proxy, vmin_out, vmax_out):
            p2 = np.nanpercentile(proxy, 2)
            p98 = np.nanpercentile(proxy, 98)
            if not np.isfinite(p2) or not np.isfinite(p98) or p98 <= p2:
                return np.full_like(proxy, np.nan, dtype="float32")
            proxy01 = (proxy - p2) / (p98 - p2 + EPS)
            proxy01 = np.clip(proxy01, 0, 1)
            return (vmin_out + proxy01 * (vmax_out - vmin_out)).astype("float32")

        if var_key == "chlor_a":
            proxy = (NIR / (R + EPS))
            out = _robust_scale_to_range(proxy, vmin, vmax)

        elif var_key == "phycocyanin":
            proxy = (R / (G + EPS))
            out = _robust_scale_to_range(proxy, vmin, vmax)

        elif var_key == "turbidity":
            proxy = R / (B + G + EPS)
            out = _robust_scale_to_range(proxy, vmin, vmax)

        elif var_key == "secchi":
            proxy_turb = R / (B + G + EPS)
            turb_nt = _robust_scale_to_range(proxy_turb, 2.5, 20.0)
            sec01 = (turb_nt - 2.5) / (20.0 - 2.5 + EPS)
            out = 100.0 - np.clip(sec01, 0, 1) * (100.0 - 20.0)
            out = np.clip(out, vmin, vmax).astype("float32")

        else:
            raise ValueError("Variável desconhecida.")

        return out

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

            zero_mask = (B == 0) & (G == 0) & (R == 0) & (NIR == 0)
            valid_mask = np.isfinite(ndvi) & (ndvi <= NDVI_MACROFITAS_THR) & (~zero_mask)

            var_scaled = compute_water_variable_scaled(B, G, R, NIR, var_key)
            var_filt = np.where(valid_mask, var_scaled, np.nan)

            meta = {
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "folium_bounds": bounds_3857_to_4326(src.bounds),
            }
            return var_filt, ndvi, ndwi, meta

    water_files = list_water_files(base_path)
    if len(water_files) == 0:
        st.warning("Nenhum arquivo encontrado com padrão DATA_*.tif na raiz do repositório.")
        st.stop()

    water_dates = [parse_date_from_filename(p) for p in water_files]

    var_map = {
        "Clorofila-a": "chlor_a",
        "Fitocianina": "phycocyanin",
        "Turbidez": "turbidity",
        "Secchi": "secchi",
    }

    bs_card_start("⚙️ Controles", "")
    c1, c2, c3, c4, c5 = st.columns([1.4, 1.5, 1.0, 1.3, 1.2])
    with c1:
        var_label = st.selectbox("Variável:", list(var_map.keys()), index=0)
    with c2:
        selected_date = st.selectbox("Data (imagem):", water_dates, index=len(water_dates) - 1)
    with c3:
        cmap_name = st.selectbox("Colormap:", ["viridis", "cividis", "plasma", "inferno", "magma"], index=0)
    with c4:
        use_mean_map = st.checkbox("Mapa médio (média de todas as datas)", value=False)
    with c5:
        show_point_clim = st.checkbox("Climatologia mensal do ponto", value=True)

    gamma = st.slider("Contraste do mapa (gamma)", 0.40, 2.00, 0.85, 0.05)
    use_internal_stretch = st.checkbox("Aumentar contraste (p2–p98 dentro da escala fixa)", value=True)
    bs_card_end()

    var_key = var_map[var_label]
    spec = VAR_SPECS[var_key]
    vmin_fixed, vmax_fixed = float(spec["vmin"]), float(spec["vmax"])
    unit = spec["unit"]
    label_unit = f"{spec['label']} ({unit})"

    tif_path = base_path / f"DATA_{selected_date}.tif"
    var_A, ndvi_A, ndwi_A, meta_A = compute_filtered_var_and_indices(tif_path, var_key)

    @st.cache_data(show_spinner=True)
    def compute_mean_raster(_var_key: str, vmin_fixed: float, vmax_fixed: float):
        sum_arr = None
        cnt_arr = None
        meta_ref = None

        for p in water_files:
            var_f, _, _, meta = compute_filtered_var_and_indices(p, _var_key)
            var_f = np.where(
                np.isfinite(var_f) & (var_f >= vmin_fixed) & (var_f <= vmax_fixed),
                var_f,
                np.nan
            )
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

    stats = {
        "n_pixels": int(vals.size),
        "média": float(np.nanmean(vals)),
        "mediana": float(np.nanmedian(vals)),
        "p10": float(np.nanpercentile(vals, 10)),
        "p90": float(np.nanpercentile(vals, 90)),
    }

    bs_card_start("📊 Estatística espacial", "após filtro NDVI + zeros")
    bs_metric_row([
        ("Pixels válidos", f"{stats['n_pixels']:,}"),
        ("Média", f"{stats['média']:.2f} {unit}"),
        ("Mediana", f"{stats['mediana']:.2f} {unit}"),
        ("p10–p90", f"{stats['p10']:.2f} – {stats['p90']:.2f} {unit}"),
    ])
    bs_card_end()

    if use_internal_stretch:
        p2 = float(np.nanpercentile(vals, 2))
        p98 = float(np.nanpercentile(vals, 98))
        vmin_vis = max(vmin_fixed, p2)
        vmax_vis = min(vmax_fixed, p98)
        if vmax_vis <= vmin_vis:
            vmin_vis, vmax_vis = vmin_fixed, vmax_fixed
    else:
        vmin_vis, vmax_vis = vmin_fixed, vmax_fixed

    inrange_fixed = np.isfinite(map_arr) & (map_arr >= vmin_fixed) & (map_arr <= vmax_fixed)

    norm_arr = np.zeros_like(map_arr, dtype=np.float32)
    norm_arr[inrange_fixed] = (map_arr[inrange_fixed] - vmin_vis) / (vmax_vis - vmin_vis + EPS)
    norm_arr[inrange_fixed] = np.clip(norm_arr[inrange_fixed], 0, 1)
    norm_arr[inrange_fixed] = norm_arr[inrange_fixed] ** float(gamma)

    img_u8 = np.zeros_like(map_arr, dtype=np.uint8)
    img_u8[inrange_fixed] = (norm_arr[inrange_fixed] * 255.0).astype(np.uint8)

    rgba = colormap_rgba(img_u8, cmap_name=cmap_name)
    rgba[..., 3] = 0
    rgba[inrange_fixed, 3] = 255

    folium_bounds = meta_use["folium_bounds"]
    center_lat = (folium_bounds[0][0] + folium_bounds[1][0]) / 2
    center_lon = (folium_bounds[0][1] + folium_bounds[1][1]) / 2

    bs_card_start("🗺️ Mapa interativo", "escala fixa + contraste (stretch interno opcional)")
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
        contraste (visual): [{vmin_vis:.2f}, {vmax_vis:.2f}] {unit} {'(p2–p98 dentro da escala fixa)' if use_internal_stretch else '(igual à escala fixa)'}<br/>
        filtro: NDVI ≤ {NDVI_MACROFITAS_THR:.2f} • pixels zerados: ocultos<br/>
        colormap: {cmap_name} • gamma: {gamma:.2f}
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    click = st_folium(m, width=1100, height=650)
    cb_img = make_colorbar_image(vmin=vmin_fixed, vmax=vmax_fixed, cmap_name=cmap_name, label=label_unit)
    st.image(cb_img, use_column_width=False)
    bs_card_end()

    bs_card_start("📈 Série temporal no ponto", "clique no mapa para extrair valores")
    if click and click.get("last_clicked"):
        lon = click["last_clicked"]["lng"]
        lat = click["last_clicked"]["lat"]
        st.success(f"Coordenada (EPSG:4326): ({lat:.5f}, {lon:.5f})")

        series = []
        for p in water_files:
            dt_str = parse_date_from_filename(p)
            try:
                var_f, _, _, _ = compute_filtered_var_and_indices(p, var_key)
                with rasterio.open(p) as src:
                    val = sample_from_array(src, var_f, lon, lat)
                series.append({"Data": dt_str, "Valor": val})
            except:
                series.append({"Data": dt_str, "Valor": np.nan})

        df_ts = pd.DataFrame(series)
        df_ts["Data"] = pd.to_datetime(df_ts["Data"])
        df_ts = df_ts.sort_values("Data")

        fig_ts = px.line(
            df_ts, x="Data", y="Valor", markers=True,
            title=f"Série temporal — {label_unit}",
            labels={"Valor": label_unit}
        )
        fig_ts.update_yaxes(range=[vmin_fixed, vmax_fixed])
        st.plotly_chart(fig_ts, use_container_width=True)

        if show_point_clim:
            st.markdown("#### 📆 Climatologia mensal do ponto (média por mês)")
            df_ts["Mes"] = df_ts["Data"].dt.month
            clim_pt = df_ts.groupby("Mes")["Valor"].mean().reset_index()

            fig_clim = px.line(
                clim_pt, x="Mes", y="Valor", markers=True,
                title=f"Climatologia mensal no ponto — {label_unit}",
                labels={"Valor": label_unit, "Mes": "Mês"}
            )
            fig_clim.update_layout(xaxis=dict(dtick=1))
            fig_clim.update_yaxes(range=[vmin_fixed, vmax_fixed])
            st.plotly_chart(fig_clim, use_container_width=True)

        with st.expander("Tabela (série no ponto)"):
            st.dataframe(df_ts, use_container_width=True)
    else:
        st.info("Clique em um ponto no mapa para extrair a série temporal e a climatologia mensal do ponto.")
    bs_card_end()

    bs_card_start("🧪 Diagnóstico (NDVI e NDWI)", "data selecionada (autoescala robusta)")
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
    bs_card_end()

    st.caption("Qualidade da Água • filtro: NDVI ≤ 0.5 (remove macrófitas). Pixels zerados ocultos. NDWI apenas diagnóstico.")
    bs_container_end()
