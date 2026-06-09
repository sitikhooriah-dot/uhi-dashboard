"""
Urban Heat Island (UHI) Prediction Dashboard
Cekungan Bandung — Gabungan Perkotaan & Non Perkotaan
PL6265 Urban Analytics
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UHI Dashboard — Cekungan Bandung",
    page_icon="🌡️",
    layout="wide",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #d62728;
        text-align: center; padding: 0.5rem 0;
    }
    .sub-header {
        font-size: 1rem; color: #555; text-align: center; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa; border-left: 4px solid #d62728;
        padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 0.5rem;
    }
    .tipologi-urban   { color: #d62728; font-weight: 700; }
    .tipologi-rural   { color: #2ca02c; font-weight: 700; }
    .info-box {
        background: #eaf4fb; border: 1px solid #aed6f1;
        padding: 0.8rem 1rem; border-radius: 6px; font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🌡️ Urban Heat Island Prediction Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cekungan Bandung • Analisis Gabungan Perkotaan & Non Perkotaan • 2014–2030</div>', unsafe_allow_html=True)
st.divider()

# ─── HELPER: LOAD MODEL ───────────────────────────────────────────────────────
@st.cache_resource
def load_model(path: str):
    """Load .pkl model; return None if not found."""
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ─── HELPER: PREDICT LST ──────────────────────────────────────────────────────
def predict_lst(ndvi, ndbi, dem, dist_road, model=None):
    """
    Predict LST (°C) from input features.
    If a real model is not available, use a physics-informed approximation
    so the dashboard still works for demo / deployment testing.
    """
    X = np.array([[ndvi, ndbi, dem, dist_road]])
    if model is not None:
        return float(model.predict(X)[0])
    # Fallback: simplified linear approximation derived from notebook patterns
    lst = (
        28.5
        - 8.0  * ndvi      # vegetation cools
        + 6.5  * ndbi      # built-up heats
        - 0.006 * dem      # elevation cools
        - 0.002 * dist_road
    )
    return round(float(lst), 2)

def predict_lst_2030(ndvi_2024, ndbi_2024, dem, dist_road,
                     tipologi, model=None):
    """Predict 2030 LST using assumed land-use change scenario."""
    if tipologi == "Perkotaan (Kota Bandung & Cimahi)":
        ndvi_2030 = max(ndvi_2024 - 0.04, 0.0)   # urban vegetation loss
        ndbi_2030 = min(ndbi_2024 + 0.05, 1.0)   # more built-up
    else:
        ndvi_2030 = max(ndvi_2024 - 0.08, 0.0)   # faster conversion at peri-urban
        ndbi_2030 = min(ndbi_2024 + 0.10, 1.0)
    lst_2024 = predict_lst(ndvi_2024, ndbi_2024, dem, dist_road, model)
    lst_2030 = predict_lst(ndvi_2030, ndbi_2030, dem, dist_road, model)
    return lst_2024, lst_2030, ndvi_2030, ndbi_2030

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Konfigurasi")

    tipologi = st.selectbox(
        "Pilih Tipologi Wilayah",
        ["Perkotaan (Kota Bandung & Cimahi)",
         "Non Perkotaan (Kab. Bandung & Kab. Bandung Barat)"],
    )

    st.subheader("📥 Model (.pkl)")
    if tipologi.startswith("Perkotaan"):
        model_label = "RF_model_urban.pkl"
        default_path = "results_urban/RF_model.pkl"
    else:
        model_label = "RF_model_rural.pkl"
        default_path = "results_rural/RF_model.pkl"

    model_path = st.text_input("Path model (opsional)", value=default_path,
                               help="Jika kosong / tidak ditemukan, mode demo aktif")
    model = load_model(model_path)
    if model:
        st.success(f"✅ Model dimuat: {model_label}")
    else:
        st.info("ℹ️ Mode demo: model aproksimasi aktif")

    st.divider()
    st.caption("PL6265 – Urban Analytics | Cekungan Bandung")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Prediksi LST", "📊 Analisis Tren", "ℹ️ Tentang"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDIKSI LST
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    is_urban = tipologi.startswith("Perkotaan")
    tag_color = "tipologi-urban" if is_urban else "tipologi-rural"
    wilayah_label = "Kota Bandung & Cimahi" if is_urban else "Kab. Bandung & Kab. Bandung Barat"

    st.markdown(
        f'Wilayah: <span class="{tag_color}">{wilayah_label}</span>',
        unsafe_allow_html=True,
    )

    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.subheader("Masukkan Nilai Fitur")

        ndvi = st.slider(
            "NDVI (Indeks Vegetasi)",
            min_value=-0.2, max_value=1.0, value=0.3, step=0.01,
            help="Nilai tinggi = vegetasi lebat → LST lebih rendah",
        )
        ndbi = st.slider(
            "NDBI (Indeks Lahan Terbangun)",
            min_value=-1.0, max_value=1.0, value=0.1, step=0.01,
            help="Nilai tinggi = bangunan padat → LST lebih tinggi",
        )
        dem = st.number_input(
            "DEM – Elevasi (m dpl)",
            min_value=600, max_value=2400,
            value=750 if is_urban else 900,
            step=10,
        )
        dist_road = st.number_input(
            "Jarak ke Jalan Terdekat (m)",
            min_value=0, max_value=5000,
            value=200, step=50,
        )

        scenario = st.radio(
            "Skenario Prediksi",
            ["Kondisi 2024 (input saat ini)", "Proyeksi 2030 (skenario BAU)"],
        )

        predict_btn = st.button("🔍 Jalankan Prediksi", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Hasil Prediksi")

        if predict_btn:
            with st.spinner("Menjalankan model..."):
                if scenario.startswith("Kondisi"):
                    lst_pred = predict_lst(ndvi, ndbi, dem, dist_road, model)

                    # UHI classification
                    if lst_pred >= 35:
                        uhi_class, uhi_color = "🔴 Hotspot UHI Tinggi", "#d62728"
                    elif lst_pred >= 30:
                        uhi_class, uhi_color = "🟠 Hangat – Potensi UHI", "#ff7f0e"
                    elif lst_pred >= 25:
                        uhi_class, uhi_color = "🟡 Moderat", "#bcbd22"
                    else:
                        uhi_class, uhi_color = "🟢 Sejuk", "#2ca02c"

                    st.metric("LST Prediksi (°C)", f"{lst_pred:.2f} °C")
                    st.markdown(
                        f'<div class="metric-card"><b>Klasifikasi UHI:</b> '
                        f'<span style="color:{uhi_color}">{uhi_class}</span></div>',
                        unsafe_allow_html=True,
                    )

                    # Feature contribution bar chart
                    fig, ax = plt.subplots(figsize=(5, 2.5))
                    features_vals = {
                        "NDVI": abs(ndvi * -8.0),
                        "NDBI": abs(ndbi * 6.5),
                        "DEM": abs(dem * 0.006),
                        "Jarak Jalan": abs(dist_road * 0.002),
                    }
                    colors_bar = ["#2ca02c", "#d62728", "#1f77b4", "#9467bd"]
                    bars = ax.barh(list(features_vals.keys()),
                                   list(features_vals.values()),
                                   color=colors_bar, edgecolor="white")
                    ax.set_title("Kontribusi Relatif Fitur", fontsize=10)
                    ax.set_xlabel("Pengaruh terhadap LST")
                    ax.invert_yaxis()
                    plt.tight_layout()
                    st.pyplot(fig)

                else:  # 2030 scenario
                    lst_24, lst_30, ndvi_30, ndbi_30 = predict_lst_2030(
                        ndvi, ndbi, dem, dist_road, tipologi, model)
                    delta = lst_30 - lst_24

                    c1, c2 = st.columns(2)
                    c1.metric("LST 2024 (°C)", f"{lst_24:.2f}")
                    c2.metric("LST 2030 (°C)", f"{lst_30:.2f}",
                              delta=f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}",
                              delta_color="inverse")

                    st.markdown(
                        f'<div class="info-box">'
                        f'<b>Perubahan Asumsi BAU:</b><br>'
                        f'• NDVI: {ndvi:.3f} → {ndvi_30:.3f}<br>'
                        f'• NDBI: {ndbi:.3f} → {ndbi_30:.3f}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # 2024 vs 2030 comparison
                    fig2, ax2 = plt.subplots(figsize=(5, 3))
                    ax2.bar(["2024", "2030"], [lst_24, lst_30],
                            color=["#1f77b4", "#d62728"], edgecolor="white", width=0.5)
                    ax2.set_ylabel("LST (°C)")
                    ax2.set_title("Perbandingan LST 2024 vs 2030")
                    for i, v in enumerate([lst_24, lst_30]):
                        ax2.text(i, v + 0.2, f"{v:.2f}°C", ha="center", fontsize=11,
                                 fontweight="bold")
                    plt.tight_layout()
                    st.pyplot(fig2)
        else:
            st.markdown(
                '<div class="info-box">⬅️ Atur nilai fitur di panel kiri, '
                'lalu klik <b>Jalankan Prediksi</b>.</div>',
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALISIS TREN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Tren LST & Hotspot UHI 2014–2024")

    # Simulated trend data (replace with actual results from notebook)
    years = [2014, 2016, 2018, 2020, 2022, 2024]

    urban_lst   = [27.8, 28.3, 28.9, 29.1, 29.6, 30.2]
    rural_lst   = [24.1, 24.5, 25.0, 25.3, 25.8, 26.4]
    urban_hotspot_pct = [8.2, 9.5, 11.0, 11.8, 13.2, 14.9]
    rural_hotspot_pct = [3.1, 3.5, 4.2, 4.8, 5.9, 7.1]

    show_both = st.toggle("Tampilkan kedua tipologi", value=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # — Panel 1: Mean LST
    axes[0].plot(years, urban_lst, "o-", color="#d62728", lw=2.5,
                 label="Perkotaan (Bandung & Cimahi)")
    if show_both:
        axes[0].plot(years, rural_lst, "s--", color="#2ca02c", lw=2,
                     label="Non Perkotaan (Kab. Bandung & KBB)")
    axes[0].fill_between(years, [v - 1 for v in urban_lst],
                         [v + 1 for v in urban_lst],
                         color="#d62728", alpha=0.1)
    axes[0].set_title("Tren Mean LST (°C)", fontweight="bold")
    axes[0].set_xlabel("Tahun"); axes[0].set_ylabel("LST (°C)")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    # — Panel 2: Hotspot pct
    axes[1].bar([y - 0.3 for y in years], urban_hotspot_pct,
                width=0.5, color="#d62728", alpha=0.8, label="Perkotaan")
    if show_both:
        axes[1].bar([y + 0.3 for y in years], rural_hotspot_pct,
                    width=0.5, color="#2ca02c", alpha=0.8, label="Non Perkotaan")
    axes[1].set_title("% Area Hotspot UHI per Tahun", fontweight="bold")
    axes[1].set_xlabel("Tahun"); axes[1].set_ylabel("% Area")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    st.pyplot(fig)

    # Summary table
    st.subheader("Tabel Ringkasan")
    df_summary = pd.DataFrame({
        "Tahun": years,
        "LST Mean Perkotaan (°C)": urban_lst,
        "LST Mean Non Perkotaan (°C)": rural_lst,
        "Hotspot Perkotaan (%)": urban_hotspot_pct,
        "Hotspot Non Perkotaan (%)": rural_hotspot_pct,
    })
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    st.caption("*Data ilustratif. Ganti dengan hasil aktual dari notebook (yearly_stats).")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TENTANG
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("ℹ️ Tentang Aplikasi")
    st.markdown("""
    **Urban Heat Island Prediction Dashboard** ini dikembangkan sebagai
    deployment tugas akhir mata kuliah **PL6265 – Urban Analytics**.

    ### Cakupan Analisis
    | Tipologi | Wilayah | Sumber Model |
    |---|---|---|
    | Perkotaan | Kota Bandung & Kota Cimahi | `RF_model_urban.pkl` |
    | Non Perkotaan | Kab. Bandung & Kab. Bandung Barat | `RF_model_rural.pkl` |

    ### Fitur Model (Random Forest)
    - **NDVI** – Normalized Difference Vegetation Index (pendingin alami)
    - **NDBI** – Normalized Difference Built-up Index (pemanas lahan terbangun)
    - **DEM** – Elevasi digital (suhu menurun seiring ketinggian)
    - **Distance to Road** – Jarak ke jalan terdekat (proxy aktivitas manusia)

    ### Pipeline
    ```
    Input Fitur → Random Forest Regressor → Prediksi LST (°C) → Klasifikasi UHI
    ```

    ### Cara Deploy ke Streamlit Community Cloud
    1. Simpan `app.py` dan `requirements.txt` ke GitHub
    2. Letakkan model `.pkl` di folder `results_urban/` & `results_rural/`
    3. Kunjungi [share.streamlit.io](https://share.streamlit.io) dan deploy

    ### Cara Deploy ke HuggingFace Spaces
    1. Buat Space baru → pilih **Streamlit**
    2. Upload `app.py`, `requirements.txt`, dan model `.pkl`
    3. Space akan build otomatis
    """)

    st.info("Jika model `.pkl` tidak tersedia, aplikasi berjalan dalam **mode demo** "
            "menggunakan aproksimasi linear berbasis pola dari notebook.")
