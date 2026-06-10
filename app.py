
import io
import math
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

warnings.filterwarnings("ignore")

st.set_page_config(page_title="OZE PV Agent Multi-sheet", page_icon="☀️", layout="wide")
st.title("☀️ OZE PV Agent — multi-sheet")
st.caption("Czyta XLS/XLSX/CSV, wykrywa zakładki i liczy osobno profile: PV dla SC ≥ 80%, BESS, ROI, oszczędność.")

with st.sidebar:
    st.header("Założenia")
    TARGET_SC = st.slider("Minimalna autokonsumpcja PV", 0.50, 0.98, 0.80, 0.01)
    ANNUAL_YIELD = st.number_input("Uzysk PV [kWh/kWp/rok]", 700, 1300, 1000, 10)
    ENERGY_PRICE_PLN_MWH = st.number_input("Wartość oszczędności [PLN/MWh]", 100, 3000, 750, 25)
    PV_CAPEX_PLN_KWP = st.number_input("CAPEX PV [PLN/kWp]", 1000, 6000, 2800, 50)
    BESS_CAPEX_PLN_KWH = st.number_input("CAPEX BESS [PLN/kWh]", 500, 5000, 1600, 50)
    OM_PV_PERCENT = st.number_input("O&M PV [% CAPEX rocznie]", 0.0, 5.0, 1.0, 0.1)

def read_workbook_or_csv(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data), sep=None, engine="python", header=None)
        return {"CSV": df}
    return {s: pd.read_excel(io.BytesIO(data), sheet_name=s, header=None)
            for s in pd.ExcelFile(io.BytesIO(data)).sheet_names}

def extract_profile_from_sheet(raw):
    best = None
    rows, cols = raw.shape
    for c_date in range(cols):
        dates = pd.to_datetime(raw.iloc[:, c_date], errors="coerce", dayfirst=True)
        if dates.notna().sum() < 10:
            continue
        for c_val in range(cols):
            if c_val == c_date:
                continue
            nums = pd.to_numeric(raw.iloc[:, c_val], errors="coerce")
            overlap = (dates.notna() & nums.notna()).sum()
            if overlap >= 10 and (best is None or overlap > best[0]):
                best = (overlap, c_date, c_val, dates, nums)
    if best is None:
        return None, "Nie znaleziono pary: data/czas + wartość liczbowa."

    _, c_date, c_val, dates, nums = best
    prof = pd.DataFrame({"timestamp": dates, "load_kwh": nums}).dropna().sort_values("timestamp")
    prof = prof[prof["load_kwh"] >= 0]
    if len(prof) < 24:
        return None, "Za mało danych po oczyszczeniu."

    deltas = prof["timestamp"].diff().dropna().dt.total_seconds() / 3600
    interval_h = float(deltas.median()) if len(deltas) else 1.0
    prof["hour_float"] = prof["timestamp"].dt.hour + prof["timestamp"].dt.minute / 60
    prof["dayofyear"] = prof["timestamp"].dt.dayofyear

    days = max((prof["timestamp"].max() - prof["timestamp"].min()).days + 1, 1)
    expected_intervals = int(round(days * 24 / interval_h)) if interval_h > 0 else len(prof)
    completeness = len(prof) / expected_intervals if expected_intervals else 1

    warns = []
    if completeness < 0.95:
        warns.append(f"Niepełny profil: kompletność ok. {completeness:.1%}.")
    if prof["load_kwh"].sum() < 10000:
        warns.append("Bardzo niskie zużycie roczne — możliwy pusty/niepełny profil lub błędny arkusz.")
    if prof["load_kwh"].quantile(0.99) == 0 and prof["load_kwh"].max() > 0:
        warns.append("Większość wartości to zera, ale są pojedyncze skoki — sprawdź jakość danych.")

    return prof, {"interval_h": interval_h, "days": days, "completeness": completeness, "warnings": warns}

def synthetic_pv_1kwp(profile, annual_yield):
    doy = profile["dayofyear"].to_numpy()
    hour = profile["hour_float"].to_numpy()
    seasonal = 0.55 + 0.45 * np.sin(2 * np.pi * (doy - 80) / 365)
    seasonal = np.clip(seasonal, 0.12, None)
    sun = np.sin(np.pi * (hour - 5) / 15)
    sun = np.clip(sun, 0, None) ** 1.7
    raw = seasonal * sun
    if raw.sum() <= 0:
        raw = np.ones(len(profile))
    days = max((profile["timestamp"].max() - profile["timestamp"].min()).days + 1, 1)
    expected_yield = annual_yield * days / 365
    return raw / raw.sum() * expected_yield

def evaluate(profile, pv_1kwp, kwp):
    load = profile["load_kwh"].to_numpy()
    pv = pv_1kwp * kwp
    self_used = np.minimum(load, pv)
    export = np.maximum(pv - load, 0)
    pv_prod = pv.sum()
    return {
        "PV_kWp": kwp,
        "PV_prod_kWh": pv_prod,
        "Self_kWh": self_used.sum(),
        "Export_kWh": export.sum(),
        "SC": self_used.sum() / pv_prod if pv_prod > 0 else 0,
        "Coverage": self_used.sum() / load.sum() if load.sum() > 0 else 0
    }

def find_max_pv(profile, pv_1kwp, target_sc):
    annual_load = profile["load_kwh"].sum()
    rough_kwp = annual_load / max(pv_1kwp.sum(), 1)
    max_scan = max(10, rough_kwp * 3)
    rows, best = [], None
    for kwp in np.linspace(1, max_scan, 700):
        r = evaluate(profile, pv_1kwp, kwp)
        rows.append(r)
        if r["SC"] >= target_sc:
            best = r
    return best, pd.DataFrame(rows)

def bess_heuristic(profile, pv_1kwp, pv_kwp):
    load = profile["load_kwh"].to_numpy()
    pv = pv_1kwp * pv_kwp
    export = np.maximum(pv - load, 0)
    days = max((profile["timestamp"].max() - profile["timestamp"].min()).days + 1, 1)
    avg_daily_export = export.sum() / days
    suggested_kwh = max(0, round(avg_daily_export * 0.35))
    suggested_kw = round(suggested_kwh / 2) if suggested_kwh > 0 else 0
    return suggested_kwh, suggested_kw

uploaded = st.file_uploader("Wgraj plik XLS/XLSX/CSV", type=["xls", "xlsx", "csv"])
if not uploaded:
    st.info("Wgraj plik z profilem/profilami.")
    st.stop()

try:
    sheets = read_workbook_or_csv(uploaded)
except Exception as e:
    st.error(f"Nie udało się odczytać pliku: {e}")
    st.stop()

results, curves = [], {}
for sheet_name, raw in sheets.items():
    profile, meta = extract_profile_from_sheet(raw)
    if profile is None:
        results.append({"Arkusz": sheet_name, "Status": meta})
        continue

    pv_1kwp = synthetic_pv_1kwp(profile, ANNUAL_YIELD)
    best, curve = find_max_pv(profile, pv_1kwp, TARGET_SC)
    curves[sheet_name] = curve

    annual_load = profile["load_kwh"].sum()
    if best is None:
        results.append({"Arkusz": sheet_name, "Status": "Brak PV spełniającej próg SC", "Zużycie_MWh": annual_load/1000})
        continue

    bess_kwh, bess_kw = bess_heuristic(profile, pv_1kwp, best["PV_kWp"])
    annual_saving = best["Self_kWh"] / 1000 * ENERGY_PRICE_PLN_MWH
    pv_capex = best["PV_kWp"] * PV_CAPEX_PLN_KWP
    om = pv_capex * OM_PV_PERCENT / 100
    roi = pv_capex / (annual_saving - om) if annual_saving > om else np.nan

    results.append({
        "Arkusz": sheet_name,
        "Status": "OK",
        "Interwał_h": meta["interval_h"],
        "Dni_danych": meta["days"],
        "Kompletność": meta["completeness"],
        "Zużycie_MWh": annual_load / 1000,
        "Max_PV_kWp_SC": best["PV_kWp"],
        "Produkcja_PV_MWh": best["PV_prod_kWh"] / 1000,
        "Autokonsumpcja_%": best["SC"] * 100,
        "Pokrycie_zużycia_%": best["Coverage"] * 100,
        "Eksport_MWh": best["Export_kWh"] / 1000,
        "Sugerowany_BESS_kWh": bess_kwh,
        "Sugerowany_BESS_kW": bess_kw,
        "Oszczędność_PLN_rok": annual_saving,
        "CAPEX_PV_PLN": pv_capex,
        "ROI_lata": roi,
        "Ostrzeżenia": " | ".join(meta["warnings"])
    })

summary = pd.DataFrame(results)
st.subheader("Wyniki")
st.dataframe(summary, use_container_width=True)

ok = summary[summary.get("Status", "") == "OK"] if "Status" in summary else pd.DataFrame()
if len(ok):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Suma zużycia", f"{ok['Zużycie_MWh'].sum():,.1f} MWh".replace(",", " "))
    c2.metric("Suma PV", f"{ok['Max_PV_kWp_SC'].sum():,.0f} kWp".replace(",", " "))
    c3.metric("Oszczędność roczna", f"{ok['Oszczędność_PLN_rok'].sum():,.0f} PLN".replace(",", " "))
    c4.metric("Średni ROI", f"{ok['ROI_lata'].replace([np.inf], np.nan).mean():.1f} lat")

for sheet_name, curve in curves.items():
    st.subheader(f"Krzywa autokonsumpcji — {sheet_name}")
    plot = curve.copy()
    plot["PV_kWp"] = plot["PV_kWp"].round(0)
    plot["Autokonsumpcja_%"] = plot["SC"] * 100
    fig = px.line(plot, x="PV_kWp", y="Autokonsumpcja_%")
    fig.add_hline(y=TARGET_SC*100, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

csv = summary.to_csv(index=False).encode("utf-8-sig")
st.download_button("Pobierz wynik CSV", csv, "wyniki_oze_pv_agent.csv", "text/csv")

st.warning("To jest screening handlowy. Moduł produkcji PV jest syntetyczny; do oferty wiążącej podmień na PVGIS/PVsyst.")
