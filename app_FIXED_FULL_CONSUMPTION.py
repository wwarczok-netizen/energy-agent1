# Energy Agent v16+ — PV / SC / BESS / multi-PPE
# Wersja naprawia import XLS/XLSX typu GPZ Buk/Opalenica: czyta WSZYSTKIE arkusze i wszystkie rekordy 15-min.

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Energy Agent — PV / SC / BESS", layout="wide")

# ---------------------------
# Helpers
# ---------------------------

def fmt_mwh(x: float) -> str:
    return f"{x:,.1f} MWh".replace(",", " ")

def fmt_kwh(x: float) -> str:
    return f"{x:,.0f} kWh".replace(",", " ")

def fmt_pln(x: float) -> str:
    return f"{x:,.0f} zł".replace(",", " ")

def find_ppe(text: str, fallback: str) -> str:
    m = re.search(r"\[(\d{3,})\]", str(text))
    if m:
        return m.group(1)
    m = re.search(r"id\s*(\d{3,})", str(fallback), re.I)
    if m:
        return m.group(1)
    return str(fallback)


def detect_interval_hours(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return 1.0
    diffs = df["datetime"].sort_values().diff().dropna().dt.total_seconds() / 3600
    diffs = diffs[(diffs > 0) & (diffs < 48)]
    if diffs.empty:
        return 1.0
    return float(diffs.mode().iloc[0])


def read_excel_all_sheets(uploaded_file) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Czyta wszystkie arkusze. Obsługuje format raportu GPZ: metadane w wierszach 0-8, dane od wiersza 9.
    Zwraca: profile long [datetime, ppe, sheet, consumption_kwh, status] oraz summary per PPE.
    """
    xls = pd.ExcelFile(uploaded_file)
    frames = []
    meta_rows = []

    for sheet in xls.sheet_names:
        raw = pd.read_excel(uploaded_file, sheet_name=sheet, header=None)
        if raw.empty:
            continue

        # szukamy wiersza z nagłówkiem "Data"
        data_row_idx = None
        for i in range(min(30, len(raw))):
            vals = [str(v).strip().lower() for v in raw.iloc[i].tolist()]
            if any(v == "data" for v in vals):
                data_row_idx = i
                break

        # format GPZ: data w kolumnie 0, wartość w kolumnie 1, status w kolumnie 2, dane od wiersza data_row_idx+2
        if data_row_idx is not None and raw.shape[1] >= 2:
            ppe = find_ppe(" ".join(raw.iloc[: min(10, len(raw)), 0].dropna().astype(str).tolist()), sheet)
            start_idx = data_row_idx + 2
            tmp = raw.iloc[start_idx:, :3].copy()
            tmp.columns = ["datetime", "consumption_kwh", "status"]
            tmp["datetime"] = pd.to_datetime(tmp["datetime"], errors="coerce")
            tmp["consumption_kwh"] = pd.to_numeric(tmp["consumption_kwh"], errors="coerce")
            tmp = tmp.dropna(subset=["datetime", "consumption_kwh"])
            tmp = tmp[tmp["consumption_kwh"] >= 0]
            tmp["ppe"] = str(ppe)
            tmp["sheet"] = sheet
            frames.append(tmp[["datetime", "ppe", "sheet", "consumption_kwh", "status"]])
            meta_rows.append({"sheet": sheet, "ppe": str(ppe), "rows_read": len(tmp), "sum_kwh": tmp["consumption_kwh"].sum()})
            continue

        # format tabelaryczny klasyczny: próbujemy odczytać z nagłówkami
        tab = pd.read_excel(uploaded_file, sheet_name=sheet)
        cols = {str(c).lower(): c for c in tab.columns}
        date_col = next((c for k, c in cols.items() if "data" in k or "date" in k or "czas" in k or "time" in k), None)
        val_col = next((c for k, c in cols.items() if "kwh" in k or "energia" in k or "zuży" in k or "warto" in k or "value" in k), None)
        ppe_col = next((c for k, c in cols.items() if "ppe" in k or "punkt" in k or "id" == k.strip()), None)
        if date_col is not None and val_col is not None:
            tmp = tab[[date_col, val_col] + ([ppe_col] if ppe_col else [])].copy()
            tmp.columns = ["datetime", "consumption_kwh"] + (["ppe"] if ppe_col else [])
            tmp["datetime"] = pd.to_datetime(tmp["datetime"], errors="coerce")
            tmp["consumption_kwh"] = pd.to_numeric(tmp["consumption_kwh"], errors="coerce")
            tmp = tmp.dropna(subset=["datetime", "consumption_kwh"])
            if "ppe" not in tmp.columns:
                tmp["ppe"] = find_ppe(sheet, sheet)
            tmp["ppe"] = tmp["ppe"].astype(str)
            tmp["sheet"] = sheet
            tmp["status"] = ""
            frames.append(tmp[["datetime", "ppe", "sheet", "consumption_kwh", "status"]])
            meta_rows.append({"sheet": sheet, "ppe": str(tmp["ppe"].iloc[0]), "rows_read": len(tmp), "sum_kwh": tmp["consumption_kwh"].sum()})

    if not frames:
        raise ValueError("Nie udało się wykryć danych: oczekuję arkuszy z kolumnami Data/Wartość albo raportu GPZ.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["ppe", "datetime"]).reset_index(drop=True)
    summary = pd.DataFrame(meta_rows)
    return df, summary


def aggregate_profile(df: pd.DataFrame, selected_ppes: List[str]) -> pd.DataFrame:
    d = df[df["ppe"].astype(str).isin([str(x) for x in selected_ppes])].copy()
    agg = d.groupby("datetime", as_index=False)["consumption_kwh"].sum()
    agg = agg.sort_values("datetime")
    return agg


def make_pv_profile(index: pd.Series, kwp: float, specific_yield: float = 1075.0) -> pd.Series:
    """Prosty syntetyczny profil PV. Roczna produkcja = kwp * specific_yield.
    Rozkład miesięczny + dzienny sinus. Działa dla danych 15-min/h.
    """
    dt = pd.to_datetime(index)
    interval_h = float(pd.Series(dt).sort_values().diff().dropna().dt.total_seconds().div(3600).mode().iloc[0]) if len(dt) > 2 else 1.0
    month_weights = np.array([0.025, 0.045, 0.080, 0.110, 0.130, 0.135, 0.135, 0.120, 0.090, 0.060, 0.040, 0.030])
    month_weights = month_weights / month_weights.sum()
    raw = np.zeros(len(dt), dtype=float)
    for m in range(1, 13):
        mask = dt.dt.month.values == m
        if not mask.any():
            continue
        # długość dnia roboczo od 8h zimą do 16h latem
        day_len = 12 + 4 * np.sin((m - 3) / 12 * 2 * np.pi)
        sunrise = 12 - day_len / 2
        sunset = 12 + day_len / 2
        hour = dt.dt.hour.values + dt.dt.minute.values / 60
        x = (hour - sunrise) / max(day_len, 1)
        shape = np.sin(np.pi * np.clip(x, 0, 1))
        shape[(hour < sunrise) | (hour > sunset)] = 0
        raw[mask] = shape[mask]
        s = raw[mask].sum()
        if s > 0:
            raw[mask] *= (kwp * specific_yield * month_weights[m-1]) / s
    return pd.Series(raw, index=index)


def calc_sc(load_kwh: pd.Series, pv_kwh: pd.Series) -> Dict[str, float]:
    auto = np.minimum(load_kwh.values, pv_kwh.values).sum()
    pv = pv_kwh.sum()
    load = load_kwh.sum()
    export = max(pv - auto, 0)
    return {
        "load_kwh": float(load),
        "pv_kwh": float(pv),
        "auto_kwh": float(auto),
        "export_kwh": float(export),
        "sc_pct": float(auto / pv * 100) if pv > 0 else 0,
        "coverage_pct": float(auto / load * 100) if load > 0 else 0,
    }


def find_pv_for_sc(profile: pd.DataFrame, target_sc: float, specific_yield: float) -> Tuple[float, Dict[str, float]]:
    load = profile.set_index("datetime")["consumption_kwh"]
    lo, hi = 0.0, max(10.0, load.sum() / specific_yield * 2.5)
    best_kwp = 0.0
    best_res = {}
    for _ in range(35):
        mid = (lo + hi) / 2
        pv = make_pv_profile(load.index.to_series(), mid, specific_yield)
        res = calc_sc(load, pv)
        if res["sc_pct"] >= target_sc:
            best_kwp, best_res = mid, res
            lo = mid
        else:
            hi = mid
    return best_kwp, best_res


def simulate_bess(load_kwh: pd.Series, pv_kwh: pd.Series, cap_kwh: float, power_kw: float, interval_h: float, rte: float = 0.90) -> Dict[str, float]:
    soc = 0.0
    auto_direct = 0.0
    charged = 0.0
    discharged = 0.0
    export = 0.0
    grid = 0.0
    max_step = power_kw * interval_h
    eta_c = np.sqrt(rte)
    eta_d = np.sqrt(rte)
    for l, p in zip(load_kwh.values, pv_kwh.values):
        direct = min(l, p)
        auto_direct += direct
        surplus = max(p - l, 0)
        deficit = max(l - p, 0)
        ch = min(surplus, max_step, max((cap_kwh - soc) / eta_c, 0))
        soc += ch * eta_c
        charged += ch
        dis = min(deficit / eta_d, max_step, soc)
        soc -= dis
        delivered = dis * eta_d
        discharged += delivered
        grid += max(deficit - delivered, 0)
        export += max(surplus - ch, 0)
    return {
        "auto_direct_kwh": float(auto_direct),
        "bess_charged_kwh": float(charged),
        "bess_discharged_kwh": float(discharged),
        "export_kwh": float(export),
        "grid_kwh": float(grid),
        "auto_total_kwh": float(auto_direct + discharged),
        "sc_with_bess_pct": float((auto_direct + discharged) / pv_kwh.sum() * 100) if pv_kwh.sum() > 0 else 0,
        "coverage_with_bess_pct": float((auto_direct + discharged) / load_kwh.sum() * 100) if load_kwh.sum() > 0 else 0,
    }

# ---------------------------
# UI
# ---------------------------

st.title("Energy Agent — PV / SC / BESS / multi-PPE")
st.caption("Importer v16+ FIX: zużycie roczne liczone z WSZYSTKICH interwałów 15-min, bez filtra godzin pracy / bez SC / bez niedziel.")

uploaded = st.file_uploader("Wgraj profil zużycia XLS/XLSX/CSV", type=["xls", "xlsx", "csv"])

with st.sidebar:
    st.header("Założenia")
    specific_yield = st.number_input("Produkcja PV [kWh/kWp/rok]", min_value=700.0, max_value=1400.0, value=1075.0, step=25.0)
    target_sc = st.slider("Docelowa autokonsumpcja SC [%]", 50, 100, 80, 1)
    price_energy = st.number_input("Wartość energii zastąpionej [zł/MWh]", min_value=0.0, value=750.0, step=10.0)
    capex_pv_kwp = st.number_input("CAPEX PV [zł/kWp]", min_value=0.0, value=2300.0, step=50.0)
    capex_bess_kwh = st.number_input("CAPEX BESS [zł/kWh]", min_value=0.0, value=1300.0, step=50.0)
    bess_power_kw = st.number_input("BESS moc [kW]", min_value=0.0, value=100.0, step=10.0)
    bess_cap_kwh = st.number_input("BESS pojemność [kWh]", min_value=0.0, value=200.0, step=10.0)

if uploaded is None:
    st.info("Wgraj plik z profilami PPE. Dla GPZ Buk/Opalenica aplikacja powinna pokazać ok. 20 554 MWh łącznie, a nie 8 103 MWh.")
    st.stop()

try:
    if uploaded.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded)
        st.error("CSV: w tej wersji użyj XLS/XLSX albo dopasuj kolumny w kodzie. Import GPZ XLS/XLSX jest gotowy.")
        st.stop()
    df, sheet_summary = read_excel_all_sheets(uploaded)
except Exception as e:
    st.exception(e)
    st.stop()

ppe_summary = df.groupby("ppe", as_index=False).agg(
    records=("consumption_kwh", "size"),
    consumption_kwh=("consumption_kwh", "sum"),
    first_date=("datetime", "min"),
    last_date=("datetime", "max"),
)
ppe_summary["consumption_mwh"] = ppe_summary["consumption_kwh"] / 1000
ppe_summary["consumption_gwh"] = ppe_summary["consumption_kwh"] / 1_000_000

st.subheader("1) Kontrola importu PPE")
cols = st.columns(4)
cols[0].metric("Liczba PPE", ppe_summary["ppe"].nunique())
cols[1].metric("Rekordy łącznie", f"{len(df):,}".replace(",", " "))
cols[2].metric("Zużycie łączne BRUTTO", fmt_mwh(df["consumption_kwh"].sum()/1000))
cols[3].metric("Zakres", f"{df['datetime'].min().date()} → {df['datetime'].max().date()}")

st.dataframe(ppe_summary[["ppe", "records", "consumption_mwh", "consumption_gwh", "first_date", "last_date"]], use_container_width=True)

all_ppes = ppe_summary["ppe"].astype(str).tolist()
selected_ppes = st.multiselect("Wybierz PPE do analizy", all_ppes, default=all_ppes)
if not selected_ppes:
    st.warning("Wybierz minimum jedno PPE.")
    st.stop()

profile = aggregate_profile(df, selected_ppes)
interval_h = detect_interval_hours(profile.rename(columns={"consumption_kwh":"consumption_kwh"}))
load = profile.set_index("datetime")["consumption_kwh"]

# Diagnostyka: wynik ok. 8,1 GWh zwykle oznacza filtr tylko dni robocze/godziny dzienne, a nie całe zużycie roczne.
_dbg = profile.copy()
_dbg["weekday"] = _dbg["datetime"].dt.weekday
_dbg["hour"] = _dbg["datetime"].dt.hour + _dbg["datetime"].dt.minute/60
work_6_18_kwh = _dbg.loc[(_dbg["weekday"] < 5) & (_dbg["hour"] >= 6) & (_dbg["hour"] < 18), "consumption_kwh"].sum()
work_7_18_kwh = _dbg.loc[(_dbg["weekday"] < 5) & (_dbg["hour"] >= 7) & (_dbg["hour"] < 18), "consumption_kwh"].sum()

st.subheader("2) Profil zużycia wybranych PPE")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Zużycie roczne BRUTTO", fmt_mwh(load.sum()/1000))
c2.metric("Średnia moc", f"{load.sum() / max(len(load)*interval_h,1):,.0f} kW".replace(",", " "))
c3.metric("Maks. energia interwału", fmt_kwh(load.max()))
c4.metric("Interwał", f"{interval_h*60:.0f} min")

with st.expander("Diagnostyka różnicy 8 103 MWh"):
    st.write("Jeżeli wcześniej widziałeś ok. 8 103 MWh, to aplikacja najpewniej liczyła tylko wycinek profilu, np. dni robocze i godziny pracy. Poniżej kontrola:")
    st.metric("Całość profilu — pełne zużycie", fmt_mwh(load.sum()/1000))
    st.metric("Tylko dni robocze 6:00–18:00", fmt_mwh(work_6_18_kwh/1000))
    st.metric("Tylko dni robocze 7:00–18:00", fmt_mwh(work_7_18_kwh/1000))
    st.write("Do dalszych obliczeń PV/SC/BESS używany jest pełny profil wybranych PPE, chyba że sam odfiltrujesz PPE w selektorze.")

monthly = profile.copy()
monthly["month"] = monthly["datetime"].dt.to_period("M").astype(str)
monthly = monthly.groupby("month", as_index=False)["consumption_kwh"].sum()
monthly["consumption_mwh"] = monthly["consumption_kwh"] / 1000
fig = px.bar(monthly, x="month", y="consumption_mwh", title="Zużycie miesięczne [MWh]")
st.plotly_chart(fig, use_container_width=True)

st.subheader("3) Dobór PV pod SC")
kwp, res = find_pv_for_sc(profile, float(target_sc), float(specific_yield))
load_series = profile.set_index("datetime")["consumption_kwh"]
pv_series = make_pv_profile(load_series.index.to_series(), kwp, specific_yield)
base = calc_sc(load_series, pv_series)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("PV dla SC", f"{kwp:,.0f} kWp".replace(",", " "))
c2.metric("Produkcja PV", fmt_mwh(base["pv_kwh"]/1000))
c3.metric("Autokonsumpcja", f"{base['sc_pct']:.1f}%")
c4.metric("Pokrycie zużycia", f"{base['coverage_pct']:.1f}%")
c5.metric("Eksport/nadwyżka", fmt_mwh(base["export_kwh"]/1000))

st.subheader("4) BESS")
bess = simulate_bess(load_series, pv_series, bess_cap_kwh, bess_power_kw, interval_h)
capex_pv = kwp * capex_pv_kwp
capex_bess = bess_cap_kwh * capex_bess_kwh
annual_savings_base = base["auto_kwh"] / 1000 * price_energy
annual_savings_bess = bess["auto_total_kwh"] / 1000 * price_energy

c1, c2, c3, c4 = st.columns(4)
c1.metric("SC z BESS", f"{bess['sc_with_bess_pct']:.1f}%")
c2.metric("Pokrycie z BESS", f"{bess['coverage_with_bess_pct']:.1f}%")
c3.metric("Energia z BESS", fmt_mwh(bess["bess_discharged_kwh"]/1000))
c4.metric("Eksport po BESS", fmt_mwh(bess["export_kwh"]/1000))

st.subheader("5) Ekonomia uproszczona")
e1, e2, e3, e4 = st.columns(4)
e1.metric("CAPEX PV", fmt_pln(capex_pv))
e2.metric("CAPEX PV+BESS", fmt_pln(capex_pv + capex_bess))
e3.metric("Oszczędność PV", fmt_pln(annual_savings_base) + "/rok")
e4.metric("Oszczędność PV+BESS", fmt_pln(annual_savings_bess) + "/rok")

payback_pv = capex_pv / annual_savings_base if annual_savings_base > 0 else np.nan
payback_bess = (capex_pv + capex_bess) / annual_savings_bess if annual_savings_bess > 0 else np.nan
st.write(f"**Prosty payback PV:** {payback_pv:.1f} lat | **Prosty payback PV+BESS:** {payback_bess:.1f} lat")

# wykres przykładowego tygodnia
st.subheader("6) Podgląd profilu — pierwszy tydzień")
plot_df = pd.DataFrame({"load_kwh": load_series, "pv_kwh": pv_series})
plot_df = plot_df.iloc[: int(min(len(plot_df), round(7*24/interval_h)))]
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=plot_df.index, y=plot_df["load_kwh"], name="Zużycie kWh/interwał"))
fig2.add_trace(go.Scatter(x=plot_df.index, y=plot_df["pv_kwh"], name="PV kWh/interwał"))
fig2.update_layout(title="Zużycie vs PV — przykładowy tydzień", xaxis_title="Czas", yaxis_title="kWh/interwał")
st.plotly_chart(fig2, use_container_width=True)

with st.expander("Diagnostyka importu arkuszy"):
    st.dataframe(sheet_summary, use_container_width=True)
    st.write("Suma kontrolna XLS/XLSX:", fmt_mwh(df["consumption_kwh"].sum()/1000))
