# Photovoltaic Parameters Calculator
# Version 5.0 - Upload first, then taskbar workflow

# Flow:
# 1. User uploads or pastes EQE data first.
# 2. After EQE is loaded, the app shows taskbar tabs:
#    - Jsc from EQE
#    - Voltage loss
# 3. Eg and Voc are asked only inside the Voltage-loss tab.
# 4. AM15G.csv and PhiiBB.csv are loaded automatically if present beside webapp.py.

# Expected files beside this script:
#     webapp.py
#     AM15G.csv
#     PhiiBB.csv
#     AM0.csv optional
# """

from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


APP_VERSION = "5.0"
APP_DIR = Path(__file__).resolve().parent

Q = 1.602176634e-19
K_B_OVER_Q = 8.617333262145e-5
HC_EV_NM = 1239.841984


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="PV Parameters Calculator",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    .hero {
        padding: 1.35rem 1.55rem;
        border-radius: 1.35rem;
        background: linear-gradient(135deg, rgba(245,158,11,.22), rgba(34,197,94,.08), rgba(14,165,233,.14));
        border: 1px solid rgba(148,163,184,.25);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.25rem;
        line-height: 1.1;
    }
    .hero p {
        margin: .4rem 0 0 0;
        color: #64748b;
        font-size: 1rem;
    }
    .step-card {
        padding: 1rem 1.1rem;
        border-radius: 1rem;
        border: 1px solid rgba(148,163,184,.25);
        background: rgba(148,163,184,.06);
        margin-bottom: 1rem;
    }
    .step-title {
        font-size: 1.08rem;
        font-weight: 800;
        margin-bottom: .35rem;
    }
    .quiet {
        color: #64748b;
        font-size: .92rem;
    }
    div[data-testid="stMetric"] {
        padding: .75rem .85rem;
        border-radius: .9rem;
        background: rgba(148,163,184,.08);
        border: 1px solid rgba(148,163,184,.20);
    }
    .status-ok {
        color: #16a34a;
        font-weight: 700;
    }
    .status-miss {
        color: #ea580c;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# File utilities
# -----------------------------
def candidate_paths(filename: str) -> List[Path]:
    return [
        APP_DIR / filename,
        APP_DIR / "data" / filename,
        APP_DIR / "assets" / filename,
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
        Path.cwd() / "assets" / filename,
    ]


def find_file(filename: str) -> Optional[Path]:
    for path in candidate_paths(filename):
        if path.exists():
            return path
    return None


def is_float_like(value) -> bool:
    try:
        float(str(value).strip())
        return True
    except Exception:
        return False


def read_text_table(text: str) -> pd.DataFrame:
    text = text.strip()
    if not text:
        raise ValueError("No data was pasted.")

    sep = r"[\t,; ]+"

    df_header = pd.read_csv(StringIO(text), sep=sep, engine="python", comment="#", header=0)
    df_noheader = pd.read_csv(StringIO(text), sep=sep, engine="python", comment="#", header=None)

    if df_header.shape[1] < 2 and df_noheader.shape[1] >= 2:
        return df_noheader

    first_names = list(df_header.columns[:2])
    if len(first_names) >= 2 and all(is_float_like(c) for c in first_names):
        return df_noheader

    return df_header


def read_bytes_table(data: bytes, filename: str = "") -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()

    if suffix in [".xls", ".xlsx"]:
        try:
            return pd.read_excel(BytesIO(data))
        except Exception:
            return pd.read_excel(BytesIO(data), header=None)

    text = data.decode("utf-8", errors="ignore")
    return read_text_table(text)


def read_any(source) -> pd.DataFrame:
    if isinstance(source, Path):
        return read_bytes_table(source.read_bytes(), source.name)

    if isinstance(source, str):
        possible_path = Path(source)
        if "\n" not in source and possible_path.exists():
            return read_any(possible_path)
        return read_text_table(source)

    if hasattr(source, "getvalue"):
        return read_bytes_table(source.getvalue(), getattr(source, "name", ""))

    raise ValueError("Unsupported data source.")


@st.cache_data(show_spinner=False)
def read_path_cached(path_string: str) -> pd.DataFrame:
    return read_any(Path(path_string))


def norm_col(col) -> str:
    return str(col).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def pick_numeric_columns(
    df: pd.DataFrame,
    y_words: Tuple[str, ...],
) -> Tuple[pd.Series, pd.Series, str, str]:
    if df.shape[1] < 2:
        raise ValueError("Data must contain at least two columns.")

    converted = df.copy()
    for col in converted.columns:
        converted[col] = pd.to_numeric(converted[col], errors="coerce")

    numeric_cols = [c for c in converted.columns if converted[c].notna().sum() >= 2]
    if len(numeric_cols) < 2:
        raise ValueError("Could not find two numeric columns.")

    x_col = None
    for c in numeric_cols:
        cname = norm_col(c)
        if any(w in cname for w in ("wavelength", "lambda", "wvlgth", "nm")):
            x_col = c
            break
    if x_col is None:
        x_col = numeric_cols[0]

    y_col = None
    for c in numeric_cols:
        if c == x_col:
            continue
        cname = norm_col(c)
        if any(w in cname for w in y_words):
            y_col = c
            break
    if y_col is None:
        y_col = [c for c in numeric_cols if c != x_col][0]

    return converted[x_col], converted[y_col], str(x_col), str(y_col)


def clean_xy(
    df: pd.DataFrame,
    y_words: Tuple[str, ...],
    y_name: str,
    wavelength_unit: str = "nm",
    y_clip: Optional[Tuple[float, float]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []

    x, y, x_col, y_col = pick_numeric_columns(df, y_words)

    out = pd.DataFrame(
        {
            "Wavelength_nm": x,
            y_name: y,
        }
    ).dropna()

    if out.empty:
        raise ValueError("No valid numeric rows found.")

    if wavelength_unit == "µm":
        out["Wavelength_nm"] = out["Wavelength_nm"] * 1000
        notes.append("Converted wavelength from µm to nm.")

    before = len(out)
    out = out[out["Wavelength_nm"] > 0]
    if len(out) < before:
        notes.append(f"Removed {before - len(out)} non-positive wavelength rows.")

    if not out["Wavelength_nm"].is_monotonic_increasing:
        notes.append("Sorted wavelength values internally.")

    duplicates = int(out["Wavelength_nm"].duplicated().sum())
    if duplicates:
        notes.append(f"Averaged {duplicates} duplicate wavelength rows.")

    out = out.groupby("Wavelength_nm", as_index=False)[y_name].mean().sort_values("Wavelength_nm")

    if y_clip is not None:
        low, high = y_clip
        bad = int(((out[y_name] < low) | (out[y_name] > high)).sum())
        if bad:
            notes.append(f"Clipped {bad} values to {low}–{high}.")
            out[y_name] = out[y_name].clip(low, high)

    notes.append(f"Detected columns: wavelength = '{x_col}', value = '{y_col}'.")
    return out.reset_index(drop=True), notes


def clean_eqe(raw: pd.DataFrame, eqe_mode: str, wavelength_unit: str) -> Tuple[pd.DataFrame, List[str], str]:
    df, notes = clean_xy(
        raw,
        y_words=("eqe", "ipce", "qe", "externalquantumefficiency"),
        y_name="EQE_input",
        wavelength_unit=wavelength_unit,
    )

    vals = df["EQE_input"].astype(float)

    if eqe_mode == "Auto":
        if vals.quantile(0.95) <= 1.2 and vals.max() <= 1.5:
            detected = "Fraction 0–1"
        else:
            detected = "Percent 0–100"
    else:
        detected = eqe_mode

    if detected == "Fraction 0–1":
        eqe_fraction = vals
    else:
        eqe_fraction = vals / 100.0

    bad = int(((eqe_fraction < 0) | (eqe_fraction > 1)).sum())
    if bad:
        notes.append(f"{bad} EQE values were outside 0–1 after conversion and were clipped.")

    df["EQE_fraction"] = eqe_fraction.clip(0, 1)
    df["EQE_percent"] = 100 * df["EQE_fraction"]

    return df[["Wavelength_nm", "EQE_fraction", "EQE_percent"]], notes, detected


def clean_spectrum(raw: pd.DataFrame, kind: str) -> Tuple[pd.DataFrame, List[str]]:
    if kind == "AM0":
        y_words = ("etr", "extraterrestrial", "am0", "intensity", "irradiance", "flux")
    else:
        y_words = ("globaltilt", "global", "am15", "am1.5", "intensity", "irradiance", "direct")

    return clean_xy(
        raw,
        y_words=y_words,
        y_name="Irradiance_W_m2_nm",
        wavelength_unit="nm",
        y_clip=(0, np.inf),
    )


def clean_phibb(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    return clean_xy(
        raw,
        y_words=("phibb", "phi", "bb", "blackbody", "intensity", "flux"),
        y_name="PhiBB",
        wavelength_unit="nm",
        y_clip=(0, np.inf),
    )


def load_spectrum(filename: str, kind: str, uploaded=None) -> Tuple[Optional[pd.DataFrame], List[str]]:
    if uploaded is not None:
        raw = read_any(uploaded)
        spectrum, notes = clean_spectrum(raw, kind)
        notes.insert(0, f"Loaded {kind} from uploaded file.")
        return spectrum, notes

    path = find_file(filename)
    if path is None:
        return None, [f"{kind} file not found: {filename}."]

    raw = read_path_cached(str(path))
    spectrum, notes = clean_spectrum(raw, kind)
    notes.insert(0, f"Loaded {kind} from {path.name}.")
    return spectrum, notes


def load_phibb(uploaded=None) -> Tuple[Optional[pd.DataFrame], List[str]]:
    if uploaded is not None:
        raw = read_any(uploaded)
        phibb, notes = clean_phibb(raw)
        notes.insert(0, "Loaded PhiBB from uploaded file.")
        return phibb, notes

    path = find_file("PhiiBB.csv")
    if path is None:
        return None, ["PhiiBB.csv file not found."]

    raw = read_path_cached(str(path))
    phibb, notes = clean_phibb(raw)
    notes.insert(0, f"Loaded PhiBB from {path.name}.")
    return phibb, notes


# -----------------------------
# Calculations
# -----------------------------
def energy_ev(wavelength_nm: np.ndarray) -> np.ndarray:
    return HC_EV_NM / wavelength_nm


def cumtrapz_np(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.zeros_like(x)
    increments = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    return np.concatenate([[0.0], np.cumsum(increments)])


def trapz_np(y: np.ndarray, x: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def eqe_on_grid(eqe: pd.DataFrame, wl: np.ndarray) -> np.ndarray:
    return np.interp(
        wl,
        eqe["Wavelength_nm"].to_numpy(float),
        eqe["EQE_fraction"].to_numpy(float),
        left=0.0,
        right=0.0,
    )


def calculate_jsc(spectrum: pd.DataFrame, eqe: pd.DataFrame) -> Dict:
    spectrum_wl = spectrum["Wavelength_nm"].to_numpy(float)
    spectrum_irr = spectrum["Irradiance_W_m2_nm"].to_numpy(float)

    eqe_wl = eqe["Wavelength_nm"].to_numpy(float)
    eqe_fraction = eqe["EQE_fraction"].to_numpy(float)

    # Use only the wavelength range where measured EQE
    # and the illumination spectrum actually overlap.
    wl_min = max(eqe_wl.min(), spectrum_wl.min())
    wl_max = min(eqe_wl.max(), spectrum_wl.max())

    if wl_max <= wl_min:
        raise ValueError(
            "The EQE wavelength range does not overlap the illumination spectrum."
        )

    # Build one common wavelength grid from both datasets.
    # This preserves the user's measured endpoints and avoids extrapolation.
    spectrum_points = spectrum_wl[
        (spectrum_wl >= wl_min) & (spectrum_wl <= wl_max)
    ]

    eqe_points = eqe_wl[
        (eqe_wl >= wl_min) & (eqe_wl <= wl_max)
    ]

    wl = np.unique(
        np.concatenate(
            [
                [wl_min],
                spectrum_points,
                eqe_points,
                [wl_max],
            ]
        )
    )

    # Interpolate only inside the validated overlap range.
    irr = np.interp(wl, spectrum_wl, spectrum_irr)
    eqe_grid = np.interp(wl, eqe_wl, eqe_fraction)

    e = energy_ev(wl)

    # AM1.5G irradiance is W m^-2 nm^-1.
    # Dividing by photon energy in eV and applying the 0.1 conversion
    # gives integrated Jsc in mA/cm^2.
    integrand = eqe_grid * irr / e

    cumulative = 0.1 * cumtrapz_np(integrand, wl)
    jsc = float(cumulative[-1]) if len(cumulative) else 0.0

    processed = pd.DataFrame(
        {
            "Wavelength_nm": wl,
            "Photon_energy_eV": e,
            "Irradiance_W_m2_nm": irr,
            "EQE_fraction": eqe_grid,
            "EQE_percent": 100 * eqe_grid,
            "Jsc_integrand": integrand,
            "Cumulative_Jsc_mA_cm2": cumulative,
        }
    )

    return {
        "jsc": jsc,
        "processed": processed,
        "integration_min_nm": float(wl_min),
        "integration_max_nm": float(wl_max),
    }



def calculate_sq_jsc(spectrum: pd.DataFrame, bandgap_ev: float, ideal_eqe_fraction: float = 1.0) -> float:
    wl = spectrum["Wavelength_nm"].to_numpy(float)
    irr = spectrum["Irradiance_W_m2_nm"].to_numpy(float)
    e = energy_ev(wl)
    ideal_eqe = np.where(e >= bandgap_ev, ideal_eqe_fraction, 0.0)
    integrand = ideal_eqe * irr / e
    cumulative = 0.1 * cumtrapz_np(integrand, wl)
    return float(cumulative[-1]) if len(cumulative) else 0.0


def calculate_voltage_loss(
    am15: pd.DataFrame,
    eqe: pd.DataFrame,
    phibb: pd.DataFrame,
    bandgap_ev: float,
    voc: float,
    temperature_k: float,
    use_eqe_weighted_j0: bool,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    jsc_result = calculate_jsc(am15, eqe)
    jsc = jsc_result["jsc"]
    jsc_sq = calculate_sq_jsc(am15, bandgap_ev, ideal_eqe_fraction=1.0)

    wl_bb = phibb["Wavelength_nm"].to_numpy(float)
    phi = phibb["PhiBB"].to_numpy(float)
    e_bb = energy_ev(wl_bb)
    above_bg = e_bb >= bandgap_ev

    if not np.any(above_bg):
        raise ValueError("No PhiBB points above selected bandgap. Check Eg or PhiBB file.")

    # PhiBB is expressed per eV, so integrate over photon energy, not wavelength.
    e_sq = e_bb[above_bg]
    phi_sq = phi[above_bg]

    order = np.argsort(e_sq)
    j0_sq = 1000 * Q * trapz_np(phi_sq[order], e_sq[order])



    if use_eqe_weighted_j0:
        # Voc_rad uses the measured EQE over its measured wavelength range.
        # Do not extrapolate EQE and do not impose the Eg cutoff.
        eqe_min = eqe["Wavelength_nm"].min()
        eqe_max = eqe["Wavelength_nm"].max()

        overlap = (wl_bb >= eqe_min) & (wl_bb <= eqe_max)
        if not np.any(overlap):
            raise ValueError("EQE and PhiBB wavelength ranges do not overlap.")

        wl_rad = wl_bb[overlap]
        e_rad = e_bb[overlap]
        phi_rad = phi[overlap]

        eqe_rad = np.interp(
            wl_rad,
            eqe["Wavelength_nm"].to_numpy(float),
            eqe["EQE_fraction"].to_numpy(float),
        )

        # PhiBB is per eV, therefore integrate over photon energy.
        order = np.argsort(e_rad)
        j0_rad = 1000 * Q * trapz_np(
            (eqe_rad * phi_rad)[order],
            e_rad[order],
        )
    else:
        j0_rad = j0_sq

    j0_sq = max(j0_sq, 1e-300)
    j0_rad = max(j0_rad, 1e-300)

    vt = K_B_OVER_Q * temperature_k
    voc_sq = vt * np.log((jsc_sq / j0_sq) + 1)
    voc_rad = vt * np.log((jsc / j0_rad) + 1)

    values = {
        "Eg": float(bandgap_ev),
        "Voc": float(voc),
        "Jsc": float(jsc),
        "Jsc_SQ": float(jsc_sq),
        "J0_SQ": float(j0_sq),
        "J0_rad": float(j0_rad),
        "Voc_SQ": float(voc_sq),
        "Voc_rad": float(voc_rad),
        "dV1": float(bandgap_ev - voc_sq),
        "dV2": float(voc_sq - voc_rad),
        "dV3": float(voc_rad - voc),
        "Total_loss": float(bandgap_ev - voc),
    }

    table = pd.DataFrame(
        {
            "Parameter": [
                "Eg",
                "Measured Voc",
                "Jsc",
                "Jsc_SQ",
                "J0_SQ",
                "J0_rad",
                "Voc_SQ",
                "Voc_rad",
                "ΔV1 = Eg/q - Voc_SQ",
                "ΔV2 = Voc_SQ - Voc_rad",
                "ΔV3 = Voc_rad - Voc",
                "Total loss = Eg/q - Voc",
            ],
            "Value": [
                values["Eg"],
                values["Voc"],
                values["Jsc"],
                values["Jsc_SQ"],
                values["J0_SQ"],
                values["J0_rad"],
                values["Voc_SQ"],
                values["Voc_rad"],
                values["dV1"],
                values["dV2"],
                values["dV3"],
                values["Total_loss"],
            ],
            "Units": [
                "eV",
                "V",
                "mA/cm²",
                "mA/cm²",
                "mA/cm²",
                "mA/cm²",
                "V",
                "V",
                "V",
                "V",
                "V",
                "V",
            ],
        }
    )

    return values, table, jsc_result["processed"]


# -----------------------------
# Display helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def parse_float(text: str, name: str) -> float:
    text = str(text).strip()
    if not text:
        raise ValueError(f"Please enter {name}.")
    try:
        value = float(text)
    except Exception:
        raise ValueError(f"{name} must be a number.")
    return value


def show_notes(notes: List[str]):
    notes = [n for n in notes if n]
    if notes:
        with st.expander("Details and file loading notes"):
            for note in notes:
                st.write(f"• {note}")


def jsc_plot(results: Dict[str, Dict]) -> go.Figure:
    fig = go.Figure()

    for name, result in results.items():
        df = result["processed"]
        fig.add_trace(
            go.Scatter(
                x=df["Wavelength_nm"],
                y=df["EQE_percent"],
                mode="lines",
                name=f"{name} EQE",
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["Wavelength_nm"],
                y=df["Cumulative_Jsc_mA_cm2"],
                mode="lines",
                name=f"{name} cumulative Jsc",
                yaxis="y2",
            )
        )

    fig.update_layout(
        title="EQE and cumulative Jsc",
        xaxis_title="Wavelength (nm)",
        yaxis=dict(title="EQE (%)"),
        yaxis2=dict(title="Cumulative Jsc (mA/cm²)", overlaying="y", side="right"),
        hovermode="x unified",
        height=520,
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=50, r=60, t=80, b=45),
    )

    return fig


def voltage_loss_plot(values: Dict) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=["Eg/q", "Voc_SQ", "Voc_rad", "Voc"],
            y=[values["Eg"], values["Voc_SQ"], values["Voc_rad"], values["Voc"]],
            name="Voltage level",
            text=[
                f"{values['Eg']:.3f}",
                f"{values['Voc_SQ']:.3f}",
                f"{values['Voc_rad']:.3f}",
                f"{values['Voc']:.3f}",
            ],
            textposition="inside",
        )
    )

    fig.add_trace(
        go.Bar(
            x=["Voc_SQ", "Voc_rad", "Voc"],
            y=[values["dV1"], values["dV2"], values["dV3"]],
            name="Loss component",
            text=[
                f"ΔV1 {values['dV1']:.3f}",
                f"ΔV2 {values['dV2']:.3f}",
                f"ΔV3 {values['dV3']:.3f}",
            ],
            textposition="inside",
        )
    )

    fig.update_layout(
        title="Voltage-loss analysis",
        yaxis_title="Voltage (V)",
        barmode="stack",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", y=1.08),
        margin=dict(l=50, r=40, t=80, b=45),
    )

    return fig


def file_status_label(filename: str) -> str:
    return "found" if find_file(filename) else "missing"


# -----------------------------
# Sidebar taskbar/support panel
# -----------------------------
with st.sidebar:
    st.markdown("## ☀️ PV Calculator")
    st.caption(f"Version {APP_VERSION}")

    st.markdown("### Default files")
    for filename in ["AM15G.csv", "PhiiBB.csv", "AM0.csv"]:
        status = file_status_label(filename)
        if status == "found":
            st.markdown(f"<span class='status-ok'>✓</span> {filename}", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='status-miss'>○</span> {filename}", unsafe_allow_html=True)

    with st.expander("Where files are searched"):
        st.write(f"App folder: `{APP_DIR}`")
        st.write(f"Run folder: `{Path.cwd()}`")


# -----------------------------
# Header
# -----------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>☀️ Photovoltaic Parameters Calculator</h1>
        <p>Upload EQE once, then choose Jsc or voltage-loss analysis from the taskbar.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Step 1: EQE upload first
# -----------------------------
st.markdown('<div class="step-card">', unsafe_allow_html=True)
st.markdown('<div class="step-title">Step 1 · Upload or paste EQE data</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="quiet">Use two columns: wavelength and EQE. EQE can be percent, e.g. 90, or fraction, e.g. 0.90.</div>',
    unsafe_allow_html=True,
)

input_method = st.radio(
    "EQE input method",
    ["Upload file", "Paste data"],
    horizontal=True,
    label_visibility="collapsed",
)

with st.expander("EQE settings", expanded=False):
    eqe_mode = st.selectbox("EQE format", ["Auto", "Percent 0–100", "Fraction 0–1"])
    wavelength_unit = st.selectbox("Wavelength unit", ["nm", "µm"])

if input_method == "Upload file":
    uploaded_eqe = st.file_uploader(
        "Upload EQE file",
        type=["csv", "txt", "xls", "xlsx"],
        help="CSV/TXT/XLSX with wavelength and EQE columns.",
    )
    pasted_eqe = ""
else:
    uploaded_eqe = None
    pasted_eqe = st.text_area(
        "Paste EQE data",
        height=170,
        placeholder="Wavelength,EQE\n300,0.02\n305,0.04\n310,0.08",
    )

st.markdown('</div>', unsafe_allow_html=True)

try:
    if uploaded_eqe is not None:
        raw_eqe = read_any(uploaded_eqe)
        eqe_df, eqe_notes, eqe_scale = clean_eqe(raw_eqe, eqe_mode, wavelength_unit)
    elif pasted_eqe.strip():
        raw_eqe = read_any(pasted_eqe)
        eqe_df, eqe_notes, eqe_scale = clean_eqe(raw_eqe, eqe_mode, wavelength_unit)
    else:
        eqe_df, eqe_notes, eqe_scale = None, [], ""
except Exception as exc:
    st.error(f"Could not read EQE data: {exc}")
    st.stop()

if eqe_df is None:
    st.info("Add EQE data first. After that, the analysis taskbar will appear.")
    st.stop()

# EQE loaded summary
c1, c2, c3 = st.columns(3)
c1.metric("EQE points", f"{len(eqe_df):,}")
c2.metric("EQE scale", eqe_scale)
c3.metric(
    "Wavelength range",
    f"{eqe_df['Wavelength_nm'].min():.0f}–{eqe_df['Wavelength_nm'].max():.0f} nm",
)

st.success("EQE loaded. Choose analysis below.")


# -----------------------------
# Step 2: taskbar after data upload
# -----------------------------
st.markdown('<div class="step-title">Step 2 · Choose analysis</div>', unsafe_allow_html=True)
jsc_tab, loss_tab = st.tabs(["📈 Jsc from EQE", "⚡ Voltage-loss analysis"])


# -----------------------------
# Jsc task
# -----------------------------
with jsc_tab:
    st.subheader("Jsc from EQE")

    left, right = st.columns([1, 1])
    with left:
        st.write("AM1.5G Jsc is calculated using `AM15G.csv` automatically.")
    with right:
        with st.expander("Optional AM0 and spectrum upload"):
            uploaded_am15 = st.file_uploader(
                "Upload AM1.5G spectrum instead of AM15G.csv",
                type=["csv", "txt", "xls", "xlsx"],
                key="jsc_am15_upload",
            )
            include_am0 = st.checkbox("Also calculate AM0 Jsc", value=False, key="include_am0")
            uploaded_am0 = None
            if include_am0:
                uploaded_am0 = st.file_uploader(
                    "Upload AM0 spectrum instead of AM0.csv",
                    type=["csv", "txt", "xls", "xlsx"],
                    key="jsc_am0_upload",
                )

    if st.button("Calculate Jsc", type="primary", use_container_width=True):
        all_notes = list(eqe_notes)
        results = {}
        rows = []

        am15, notes = load_spectrum("AM15G.csv", "AM1.5G", uploaded_am15)
        all_notes.extend(notes)

        if am15 is not None:
            result = calculate_jsc(am15, eqe_df)
            results["AM1.5G"] = result
            rows.append({"Spectrum": "AM1.5G", "Jsc (mA/cm²)": result["jsc"]})

        if include_am0:
            am0, notes = load_spectrum("AM0.csv", "AM0", uploaded_am0)
            all_notes.extend(notes)
            if am0 is not None:
                result = calculate_jsc(am0, eqe_df)
                results["AM0"] = result
                rows.append({"Spectrum": "AM0", "Jsc (mA/cm²)": result["jsc"]})

        if not results:
            st.error("AM1.5G spectrum was not found. Put AM15G.csv beside webapp.py or upload it.")
            show_notes(all_notes)
            st.stop()

        st.markdown("### Result")
        metric_cols = st.columns(len(rows))
        for col, row in zip(metric_cols, rows):
            col.metric(row["Spectrum"], f"{row['Jsc (mA/cm²)']:.3f} mA/cm²")

        st.plotly_chart(jsc_plot(results), use_container_width=True)

        with st.expander("Table and downloads"):
            summary = pd.DataFrame(rows)
            display = summary.copy()
            display["Jsc (mA/cm²)"] = display["Jsc (mA/cm²)"].map(lambda x: f"{x:.4f}")
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.download_button("Download Jsc summary", csv_bytes(summary), "jsc_summary.csv", "text/csv")
            st.download_button("Download cleaned EQE", csv_bytes(eqe_df), "cleaned_eqe.csv", "text/csv")

            for name, result in results.items():
                safe = name.replace(".", "").replace(" ", "_")
                st.download_button(
                    f"Download processed {name} data",
                    csv_bytes(result["processed"]),
                    f"processed_{safe}.csv",
                    "text/csv",
                )

        show_notes(all_notes)


# -----------------------------
# Voltage-loss task
# -----------------------------
with loss_tab:
    st.subheader("Voltage-loss analysis")
    st.caption("Enter measured device values. The app does not assume Eg or Voc.")

    col1, col2, col3 = st.columns(3)
    with col1:
        eg_text = st.text_input("Bandgap Eg (eV)", placeholder="Example: 1.50", key="eg_text")
    with col2:
        voc_text = st.text_input("Measured Voc (V)", placeholder="Example: 0.800", key="voc_text")
    with col3:
        temp_text = st.text_input("Temperature (K)", value="300", key="temp_text")

    with st.expander("Optional files and settings"):
        uploaded_am15_voc = st.file_uploader(
            "Upload AM1.5G spectrum instead of AM15G.csv",
            type=["csv", "txt", "xls", "xlsx"],
            key="voc_am15_upload",
        )
        uploaded_phibb = st.file_uploader(
            "Upload PhiBB file instead of PhiiBB.csv",
            type=["csv", "txt", "xls", "xlsx"],
            key="phibb_upload",
        )
        use_eqe_weighted_j0 = st.checkbox("Use EQE-weighted J0 for Voc_rad", value=True)

    if st.button("Calculate voltage loss", type="primary", use_container_width=True):
        try:
            eg = parse_float(eg_text, "bandgap Eg")
            voc = parse_float(voc_text, "measured Voc")
            temp = parse_float(temp_text, "temperature")

            if eg <= 0:
                raise ValueError("Bandgap Eg must be greater than 0.")
            if voc < 0:
                raise ValueError("Measured Voc cannot be negative.")
            if temp <= 0:
                raise ValueError("Temperature must be greater than 0 K.")
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        all_notes = list(eqe_notes)

        am15, notes = load_spectrum("AM15G.csv", "AM1.5G", uploaded_am15_voc)
        all_notes.extend(notes)

        phibb, notes = load_phibb(uploaded_phibb)
        all_notes.extend(notes)

        if am15 is None:
            st.error("AM1.5G spectrum is required. Put AM15G.csv beside webapp.py or upload it.")
            show_notes(all_notes)
            st.stop()

        if phibb is None:
            st.error("PhiBB data is required. Put PhiiBB.csv beside webapp.py or upload it.")
            show_notes(all_notes)
            st.stop()

        try:
            values, table, processed = calculate_voltage_loss(
                am15=am15,
                eqe=eqe_df,
                phibb=phibb,
                bandgap_ev=eg,
                voc=voc,
                temperature_k=temp,
                use_eqe_weighted_j0=use_eqe_weighted_j0,
            )
        except Exception as exc:
            st.error(f"Could not calculate voltage loss: {exc}")
            st.stop()

        st.markdown("### Result")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jsc", f"{values['Jsc']:.3f} mA/cm²")
        c2.metric("Voc_SQ", f"{values['Voc_SQ']:.3f} V")
        c3.metric("Voc_rad", f"{values['Voc_rad']:.3f} V")
        c4.metric("Total loss", f"{values['Total_loss']:.3f} V")

        c5, c6, c7 = st.columns(3)
        c5.metric("ΔV1", f"{values['dV1']:.3f} V")
        c6.metric("ΔV2", f"{values['dV2']:.3f} V")
        c7.metric("ΔV3", f"{values['dV3']:.3f} V")

        if values["dV3"] < 0:
            st.warning("ΔV3 is negative. Please check Eg, measured Voc, EQE, and PhiBB assumptions.")

        st.plotly_chart(voltage_loss_plot(values), use_container_width=True)

        with st.expander("Table and downloads"):
            display = table.copy()
            display["Value"] = display["Value"].map(lambda x: f"{x:.6g}")
            st.dataframe(display, use_container_width=True, hide_index=True)
            st.download_button("Download voltage-loss results", csv_bytes(table), "voltage_loss_results.csv", "text/csv")
            st.download_button("Download cleaned EQE", csv_bytes(eqe_df), "cleaned_eqe.csv", "text/csv")
            st.download_button("Download processed AM1.5G data", csv_bytes(processed), "processed_voltage_loss_data.csv", "text/csv")

        show_notes(all_notes)

st.caption("Validate results with known examples before publishing a public version.")