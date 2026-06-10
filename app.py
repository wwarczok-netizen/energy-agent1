import io
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(page_title="OZE PV Agent", page_icon="☀️", layout="wide")

# ============================================================
# OZE PV Agent — Streamlit app.py
# Na bazie notebooka Colab multi-sheet:
# - XLS/XLSX/CSV
# - wiele zakładek
# - próba wykrycia wielu PPE w jednym CSV / arkuszu
# - max PV dla SC >= zadany próg, domyślnie 80%
# - orientacyjny BESS
# - oszczędność, CAPEX, ROI
# - eksport do XLSX
# ============================================================


def normalize_col_name(x):
    return str(x).strip().lower().replace("\n", " ").replace("\r", " ")


def safe_sheet_name(name: str) -> str:
    name = re.sub(r"[\\/*?:\[\]]", "_", str(name))
    return name[:31] if name else "Arkusz"


@st.cache_data(show_spinner=False)
def read_workbook_or_csv(filename: str, content: bytes):
    lower = filename.lower()
    if lower.endswith(".csv"):
        # Najpierw próbujemy z automatycznym separatorem i nagłówkiem.
        try:
            df_header = pd.read_csv(io.BytesIO(content), sep=None, engine="python")
            df_raw = pd.read_csv(io.BytesIO(content), sep=None, engine="python", header=None)
        except UnicodeDecodeError:
            df_header = pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding="cp1250")
            df_raw = pd.read_csv(io.BytesIO(content), sep=None, engine="python", header=None, encoding="cp1250")
        return {"CSV": {"header": df_header, "raw": df_raw}}

    if lower.endswith(".xls"):
        engine = "xlrd"
    else:
        engine = "openpyxl"

    xl = pd.ExcelFile(io.BytesIO(content), engine=engine)
    out = {}
    for sheet in xl.sheet_names:
        raw = pd.read_excel(io.BytesIO(content), sheet_name=sheet, header=None, engine=engine)
        try:
            header = pd.read_excel(io.BytesIO(content), sheet_name=sheet, engine=engine)
        except Exception:
            header = raw.copy()
        out[sheet] = {"header": header, "raw": raw}
    return out


def find_ppe_column(df: pd.DataFrame):
    candidates = []
    for col in df.columns:
        name = normalize_col_name(col)
        if "ppe" in name or "punkt" in name or "kod" in name or "meter" in name or "licznik" in name:
            candidates.append(col)
    # Dla Enea/Tauron często PPE ma 18 cyfr i może być w nienazwanej kolumnie.
    for col in df.columns:
        vals = df[col].dropna().astype(str).head(2000)
        hits = vals.str.contains(r"\b\d{18}\b", regex=True).sum()
        if hits >= 5:
            candidates.append(col)
    return candidates[0] if candidates else None


def split_by_ppe_if_possible(sheet_name: str, data_obj: dict):
    """Zwraca listę (nazwa, raw_df). Jeżeli jest kolumna PPE, dzieli na PPE."""
    header = data_obj["header"]
    raw = data_obj["raw"]

    # Dzielimy tylko wtedy, gdy dataframe z nagłówkiem sensownie zawiera PPE.
    try:
        ppe_col = find_ppe_column(header)
        if ppe_col is not None:
            unique_vals = header[ppe_col].dropna().astype(str).unique()
            ppes = [v for v in unique_vals if re.search(r"\d{8,}", v)]
            if 1 < len(ppes) <= 50:
                parts = []
                for ppe in ppes:
                    sub = header[header[ppe_col].astype(str) == str(ppe)].copy()
                    # Do dalszej ekstrakcji łatwiej użyć dataframe z nagłówkiem jako raw-like.
                    sub_raw = pd.DataFrame([sub.columns.tolist()] + sub.astype(object).values.tolist())
                    parts.append((f"{sheet_name} | PPE {ppe}", sub_raw))
                return parts
    except Exception:
        pass

    return [(sheet_name, raw)]


def extract_profile_from_sheet(raw: pd.DataFrame):
    """
    Szuka pary kolumn: data/czas + wartość liczbowa.
    Działa także, gdy dane zaczynają się po wierszach opisowych.
    Zwraca profil z kolumnami: timestamp, load_kwh.
    """
    best = None
    rows, cols = raw.shape

    for c_date in range(cols):
        parsed_dates = pd.to_datetime(raw.iloc[:, c_date], errors="coerce", dayfirst=True)
        date_count = parsed_dates.notna().sum()
        if date_count < 10:
            continue

        for c_val in range(cols):
            if c_val == c_date:
                continue
            nums = pd.to_numeric(raw.iloc[:, c_val], errors="coerce")
            overlap_mask = parsed_dates.notna() & nums.notna()
            overlap = overlap_mask.sum()
            if overlap < 10:
                continue

            # Preferujemy kolumny z dodatnim wolumenem i dużą liczbą obserwacji.
            positive_sum = nums[overlap_mask].clip(lower=0).sum()
            score = overlap + min(float(positive_sum), 1_000_000) / 1_000_000
            if best is None or score > best[0]:
                best = (score, c_date, c_val, parsed_dates, nums)

    if best is None:
        return None, {"error": "Nie znaleziono pary: data/czas + wartość liczbowa."}

    _, c_date, c_val, dates, nums = best
    prof = pd.DataFrame({"timestamp": dates, "load_kwh": nums})
    prof = prof.dropna().sort_values("timestamp")
    prof = prof[prof["load_kwh"] >= 0]
    prof = prof.drop_duplicates(subset=["timestamp"], keep="last")

    if len(prof) < 24:
        return None, {"error": "Za mało danych po oczyszczeniu."}

    deltas = prof["timestamp"].diff().dropna().dt.total_seconds() / 3600
    interval_h = float(deltas.median()) if len(deltas) else 1.0
    if interval_h <= 0 or np.isnan(interval_h):
        interval_h = 1.0

    prof["hour_float"] = prof["timestamp"].dt.hour + prof["timestamp"].dt.minute / 60
    prof["dayofyear"] = prof["timestamp"].dt.dayofyear
    prof["date"] = prof["timestamp"].dt.date
    prof["weekday"] = prof["timestamp"].dt.dayofweek

    days_covered = max((prof["timestamp"].max() - prof["timestamp"].min()).days + 1, 1)
    expected_intervals = int(round(days_covered * 24 / interval_h)) if interval_h > 0 else len(prof)
    completeness = len(prof) / expected_intervals if expected_intervals else 1

    warnings_list = []
    if completeness < 0.95:
        warnings_list.append(f"Niepełny profil: kompletność ok. {completeness:.1%}.")
    if prof["load_kwh"].sum() < 10_000:
        warnings_list.append("Niskie zużycie roczne — sprawdź czy arkusz/PPE jest kompletny.")
    if prof["load_kwh"].max() == 0:
        warnings_list.append("Same wartości zerowe.")

    meta = {
        "date_col": c_date,
        "value_col": c_val,
        "interval_h": interval_h,
        "days_covered": days_covered,
        "completeness": completeness,
        "warnings": warnings_list,
        "start": prof["timestamp"].min(),
        "end": prof["timestamp"].max(),
    }
    return prof, meta


def synthetic_pv_1kwp(profile: pd.DataFrame, annual_yield: float, orientation: str = "Południe"):
    """
    Uproszczony profil PV 1 kWp. To screening handlowy, nie PVsyst.
    annual_yield: kWh/kWp/rok.
    """
    doy = profile["dayofyear"].to_numpy()
    hour = profile["hour_float"].to_numpy()

    seasonal = 0.55 + 0.45 * np.sin(2 * np.pi * (doy - 80) / 365)
    seasonal = np.clip(seasonal, 0.12, None)

    if orientation == "Wschód-Zachód":
        morning = np.sin(np.pi * (hour - 4.5) / 10)
        afternoon = np.sin(np.pi * (hour - 9.5) / 10)
        sun = np.clip(morning, 0, None) ** 2.0 + np.clip(afternoon, 0, None) ** 2.0
        sun = sun / max(np.max(sun), 1e-9)
    else:
        sun = np.sin(np.pi * (hour - 5) / 15)
        sun = np.clip(sun, 0, None) ** 1.7

    raw = seasonal * sun
    if raw.sum() <= 0:
        raw = np.ones(len(profile))

    days_covered = max((profile["timestamp"].max() - profile["timestamp"].min()).days + 1, 1)
    expected_yield_for_period = annual_yield * days_covered / 365
    return raw / raw.sum() * expected_yield_for_period


def evaluate(profile: pd.DataFrame, pv_1kwp: np.ndarray, kwp: float):
    load = profile["load_kwh"].to_numpy()
    pv = pv_1kwp * kwp
    self_used = np.minimum(load, pv)
    export = np.maximum(pv - load, 0)
    pv_prod = pv.sum()
    load_sum = load.sum()
    return {
        "PV_kWp": kwp,
        "PV_prod_kWh": pv_prod,
        "Self_kWh": self_used.sum(),
        "Export_kWh": export.sum(),
        "SC": self_used.sum() / pv_prod if pv_prod > 0 else 0,
        "Coverage": self_used.sum() / load_sum if load_sum > 0 else 0,
    }


def find_max_pv(profile: pd.DataFrame, pv_1kwp: np.ndarray, target_sc: float, step_kwp: float, max_kwp_user: float | None):
    annual_load = profile["load_kwh"].sum()
    rough_kwp = annual_load / max(pv_1kwp.sum(), 1)
    max_scan = max_kwp_user if max_kwp_user and max_kwp_user > 0 else max(10, rough_kwp * 3)
    grid = np.arange(step_kwp, max_scan + step_kwp, step_kwp)

    best = None
    rows = []
    for kwp in grid:
        r = evaluate(profile, pv_1kwp, float(kwp))
        rows.append(r)
        if r["SC"] >= target_sc:
            best = r
    return best, pd.DataFrame(rows)


def bess_heuristic(profile: pd.DataFrame, pv_1kwp: np.ndarray, pv_kwp: float):
    load = profile["load_kwh"].to_numpy()
    pv = pv_1kwp * pv_kwp
    export = np.maximum(pv - load, 0)
    days = max((profile["timestamp"].max() - profile["timestamp"].min()).days + 1, 1)
    avg_daily_export = export.sum() / days
    p95_export_kw = np.percentile(export / max(float(profile["timestamp"].diff().dropna().dt.total_seconds().median() / 3600), 0.25), 95)

    suggested_kwh = max(0, round(avg_daily_export * 0.35))
    suggested_kw = round(min(p95_export_kw, suggested_kwh / 2)) if suggested_kwh > 0 else 0
    return suggested_kwh, suggested_kw


def make_excel(summary: pd.DataFrame, curves: dict, profiles: dict):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Wyniki")
        for name, curve in curves.items():
            curve_out = curve.copy()
            for col in ["SC", "Coverage"]:
                if col in curve_out:
                    curve_out[col] = curve_out[col] * 100
            curve_out.to_excel(writer, index=False, sheet_name=safe_sheet_name(f"Krzywa_{name}"))
        for name, profile in profiles.items():
            monthly = profile.copy()
            monthly["Miesiąc"] = monthly["timestamp"].dt.to_period("M").astype(str)
            monthly = monthly.groupby("Miesiąc", as_index=False)["load_kwh"].sum()
            monthly["Zużycie_MWh"] = monthly["load_kwh"] / 1000
            monthly.drop(columns=["load_kwh"], inplace=True)
            monthly.to_excel(writer, index=False, sheet_name=safe_sheet_name(f"Mies_{name}"))
    return output.getvalue()


st.title("☀️ OZE PV Agent — profile zużycia, SC i dobór PV")
st.caption("Wersja Streamlit na bazie notebooka Colab multi-sheet. Screening handlowy — nie zastępuje PVsyst/projektu technicznego.")

with st.sidebar:
    st.header("Założenia")
    target_sc = st.slider("Minimalna autokonsumpcja PV / s.c.", 50, 99, 80, 1) / 100
    annual_yield = st.number_input("Uzysk PV [kWh/kWp/rok]", min_value=600, max_value=1400, value=1050, step=25)
    orientation = st.selectbox("Profil produkcji PV", ["Południe", "Wschód-Zachód"], index=0)
    energy_price = st.number_input("Wartość energii + dystrybucji [PLN/MWh]", min_value=0, value=750, step=25)
    pv_capex = st.number_input("CAPEX PV [PLN/kWp]", min_value=0, value=2800, step=50)
    bess_capex = st.number_input("CAPEX BESS [PLN/kWh]", min_value=0, value=1600, step=50)
    om_percent = st.number_input("O&M PV [% CAPEX/rok]", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    step_kwp = st.number_input("Krok obliczeń PV [kWp]", min_value=1, max_value=100, value=5, step=1)
    max_kwp_user = st.number_input("Opcjonalny limit skanowania [kWp], 0 = auto", min_value=0, value=0, step=100)

uploaded_file = st.file_uploader("Wgraj profil zużycia: CSV, XLS albo XLSX", type=["csv", "xls", "xlsx"])

if uploaded_file is None:
    st.info("Wgraj plik z profilem. Aplikacja spróbuje sama wykryć kolumnę daty/czasu i kolumnę zużycia energii.")
    st.stop()

content = uploaded_file.getvalue()

try:
    sheets = read_workbook_or_csv(uploaded_file.name, content)
except ImportError as e:
    st.error(f"Brakuje biblioteki do odczytu pliku: {e}. Dla .xls zainstaluj: pip install xlrd")
    st.stop()
except Exception as e:
    st.error(f"Nie udało się odczytać pliku: {e}")
    st.stop()

st.success(f"Wczytano plik: {uploaded_file.name}")

all_parts = []
for sheet_name, data_obj in sheets.items():
    all_parts.extend(split_by_ppe_if_possible(sheet_name, data_obj))

st.write(f"Znalezione arkusze / profile: **{len(all_parts)}**")

results = []
curves = {}
profiles = {}

progress = st.progress(0)
for idx, (name, raw) in enumerate(all_parts, start=1):
    profile, meta = extract_profile_from_sheet(raw)
    progress.progress(idx / max(len(all_parts), 1))

    if profile is None:
        results.append({"Profil": name, "Status": meta.get("error", "Błąd odczytu")})
        continue

    pv_1kwp = synthetic_pv_1kwp(profile, annual_yield, orientation)
    best, curve = find_max_pv(profile, pv_1kwp, target_sc, step_kwp, max_kwp_user if max_kwp_user > 0 else None)
    curves[name] = curve
    profiles[name] = profile

    annual_load = profile["load_kwh"].sum()
    if best is None:
        results.append({
            "Profil": name,
            "Status": f"Brak PV spełniającej SC >= {target_sc:.0%}",
            "Zużycie_MWh": annual_load / 1000,
            "Ostrzeżenia": " | ".join(meta["warnings"]),
        })
        continue

    pv_kwp = best["PV_kWp"]
    bess_kwh, bess_kw = bess_heuristic(profile, pv_1kwp, pv_kwp)
    annual_saving = best["Self_kWh"] / 1000 * energy_price
    pv_capex_total = pv_kwp * pv_capex
    bess_capex_total = bess_kwh * bess_capex
    om = pv_capex_total * om_percent / 100
    net_saving = annual_saving - om
    roi = pv_capex_total / net_saving if net_saving > 0 else np.nan

    results.append({
        "Profil": name,
        "Status": "OK",
        "Od": meta["start"],
        "Do": meta["end"],
        "Interwał_h": meta["interval_h"],
        "Dni_danych": meta["days_covered"],
        "Kompletność_%": meta["completeness"] * 100,
        "Zużycie_MWh": annual_load / 1000,
        f"Max_PV_kWp_SC_{int(target_sc*100)}": pv_kwp,
        "Produkcja_PV_MWh": best["PV_prod_kWh"] / 1000,
        "Autokonsumpcja_%": best["SC"] * 100,
        "Pokrycie_zużycia_%": best["Coverage"] * 100,
        "Eksport_MWh": best["Export_kWh"] / 1000,
        "Sugerowany_BESS_kWh": bess_kwh,
        "Sugerowany_BESS_kW": bess_kw,
        "CAPEX_BESS_PLN": bess_capex_total,
        "Oszczędność_PLN_rok": annual_saving,
        "CAPEX_PV_PLN": pv_capex_total,
        "O&M_PLN_rok": om,
        "ROI_lata": roi,
        "Ostrzeżenia": " | ".join(meta["warnings"]),
    })

summary = pd.DataFrame(results)

st.subheader("Wyniki")
st.dataframe(summary, use_container_width=True)

ok = summary[summary["Status"] == "OK"].copy() if "Status" in summary else pd.DataFrame()
if not ok.empty:
    pv_cols = [c for c in ok.columns if c.startswith("Max_PV_kWp")]
    pv_col = pv_cols[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Łączne zużycie", f"{ok['Zużycie_MWh'].sum():,.0f} MWh".replace(",", " "))
    c2.metric("Suma max PV", f"{ok[pv_col].sum():,.0f} kWp".replace(",", " "))
    c3.metric("Produkcja PV", f"{ok['Produkcja_PV_MWh'].sum():,.0f} MWh".replace(",", " "))
    c4.metric("Autokonsumpcja średnia", f"{ok['Autokonsumpcja_%'].mean():.1f}%")

    st.subheader("Wykres — max PV według profilu")
    fig = px.bar(ok, x="Profil", y=pv_col, hover_data=["Zużycie_MWh", "Autokonsumpcja_%", "Pokrycie_zużycia_%"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Krzywe SC")
    chosen = st.selectbox("Wybierz profil do podglądu krzywej", list(curves.keys()))
    curve = curves[chosen].copy()
    curve["Autokonsumpcja_%"] = curve["SC"] * 100
    fig2 = px.line(curve, x="PV_kWp", y="Autokonsumpcja_%")
    fig2.add_hline(y=target_sc * 100, line_dash="dash")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Profil miesięczny zużycia")
    prof = profiles[chosen].copy()
    prof["Miesiąc"] = prof["timestamp"].dt.to_period("M").astype(str)
    monthly = prof.groupby("Miesiąc", as_index=False)["load_kwh"].sum()
    monthly["Zużycie_MWh"] = monthly["load_kwh"] / 1000
    fig3 = px.bar(monthly, x="Miesiąc", y="Zużycie_MWh")
    st.plotly_chart(fig3, use_container_width=True)

excel_bytes = make_excel(summary, curves, profiles)
st.download_button(
    label="⬇️ Pobierz wyniki Excel",
    data=excel_bytes,
    file_name="wyniki_oze_pv_agent_streamlit.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.caption("Uwaga: profil PV jest syntetyczny i służy do szybkiego screeningu. Do oferty końcowej użyj PVsyst / projektu technicznego.")
