import io
import re
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Energy Agent MVP | PV + BESS", layout="wide")


def parse_number_pl(x):
    if pd.isna(x):
        return 0.0
    s = str(x).strip().replace(" ", "")
    if s in {"---", "-", "", "nan", "None"}:
        return 0.0
    return float(s.replace(",", "."))


def load_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    last_error = None
    for enc in ["utf-8-sig", "cp1250", "latin1"]:
        for sep in ["\t", ";", ","]:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc)
                if df.shape[1] >= 2:
                    return normalize_profile(df)
            except Exception as e:
                last_error = e
    raise ValueError(f"Nie udało się odczytać CSV: {last_error}")


def normalize_profile(df: pd.DataFrame) -> pd.DataFrame:
    # Column detection
    date_col = df.columns[0]
    load_col = None
    for c in df.columns:
        cl = str(c).lower()
        if "pobranej" in cl or "zuży" in cl or "pobor" in cl or "load" in cl:
            load_col = c
            break
    if load_col is None:
        load_col = df.columns[1]

    out = pd.DataFrame()
    # Handle Polish DSO daylight-saving suffixes such as 02:59A / 02:59B
    dt_text = df[date_col].astype(str).str.replace(r"([0-9]{2}:[0-9]{2})[A-Z]$", r"\1", regex=True)
    out["timestamp"] = pd.to_datetime(dt_text, errors="coerce")
    out["load_kwh"] = df[load_col].map(parse_number_pl)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    out["date"] = out["timestamp"].dt.date
    out["hour"] = out["timestamp"].dt.hour
    out["month"] = out["timestamp"].dt.month
    out["weekday"] = out["timestamp"].dt.dayofweek
    out["is_sunday"] = out["weekday"].eq(6)
    return out


def synthetic_pv_profile(df: pd.DataFrame, pv_kwp: float, annual_yield_kwh_kwp: float) -> np.ndarray:
    ts = df["timestamp"]
    doy = ts.dt.dayofyear.to_numpy()
    hour = ts.dt.hour.to_numpy() + 0.5

    # Seasonal factor: peak in late June, lower in winter
    seasonal = 0.18 + 0.82 * np.maximum(0, np.sin(np.pi * (doy - 20) / 365)) ** 1.35

    # Approximate daylight length and sunrise/sunset by season
    day_len = 8.0 + 8.0 * np.maximum(0, np.sin(np.pi * (doy - 80) / 365))
    sunrise = 12 - day_len / 2
    sunset = 12 + day_len / 2
    daylight_pos = (hour - sunrise) / np.maximum(day_len, 1)
    diurnal = np.where((hour >= sunrise) & (hour <= sunset), np.sin(np.pi * daylight_pos) ** 1.7, 0)

    raw = seasonal * diurnal
    annual_target = pv_kwp * annual_yield_kwh_kwp
    scale = annual_target / raw.sum() if raw.sum() > 0 else 0
    return raw * scale


@dataclass
class BessConfig:
    capacity_kwh: float
    power_kw: float
    roundtrip_eff: float
    initial_soc_pct: float = 10.0
    min_soc_pct: float = 5.0


def simulate_pv_bess(
    df: pd.DataFrame,
    pv_kwp: float,
    annual_yield: float,
    bess: BessConfig,
    peak_target_kw: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    sim = df.copy()
    load = sim["load_kwh"].to_numpy(dtype=float)  # hourly data, kWh ~= avg kW
    pv = synthetic_pv_profile(sim, pv_kwp, annual_yield)
    dt_h = 1.0
    eta = np.sqrt(bess.roundtrip_eff)

    direct = np.minimum(load, pv)
    pv_surplus = np.maximum(pv - load, 0)
    residual_load = np.maximum(load - pv, 0)

    soc = bess.capacity_kwh * bess.initial_soc_pct / 100 if bess.capacity_kwh > 0 else 0
    min_soc = bess.capacity_kwh * bess.min_soc_pct / 100 if bess.capacity_kwh > 0 else 0
    max_soc = bess.capacity_kwh

    charge = np.zeros(len(sim))
    discharge_self = np.zeros(len(sim))
    discharge_peak = np.zeros(len(sim))
    export_after_bess = np.zeros(len(sim))
    soc_series = np.zeros(len(sim))
    grid_after = np.zeros(len(sim))

    for i in range(len(sim)):
        # 1) charge BESS only from PV surplus, no arbitrage
        if bess.capacity_kwh > 0 and bess.power_kw > 0 and pv_surplus[i] > 0:
            max_charge_from_power = bess.power_kw * dt_h
            max_charge_from_space = max(0, (max_soc - soc) / eta)
            ch = min(pv_surplus[i], max_charge_from_power, max_charge_from_space)
            charge[i] = ch
            soc += ch * eta

        # 2) discharge for self-consumption against residual load
        grid = residual_load[i]
        if bess.capacity_kwh > 0 and bess.power_kw > 0 and grid > 0:
            available = max(0, (soc - min_soc) * eta)
            power_left = max(0, bess.power_kw * dt_h)
            dis = min(grid, available, power_left)
            discharge_self[i] = dis
            soc -= dis / eta
            grid -= dis

        # 3) additional peak shaving to target, if possible
        if bess.capacity_kwh > 0 and bess.power_kw > 0 and grid > peak_target_kw:
            available = max(0, (soc - min_soc) * eta)
            used_power = discharge_self[i]
            power_left = max(0, bess.power_kw * dt_h - used_power)
            peak_need = grid - peak_target_kw
            dis = min(peak_need, available, power_left)
            discharge_peak[i] = dis
            soc -= dis / eta
            grid -= dis

        export_after_bess[i] = max(0, pv_surplus[i] - charge[i])
        soc_series[i] = soc
        grid_after[i] = grid

    sim["pv_kwh"] = pv
    sim["direct_self_kwh"] = direct
    sim["pv_surplus_kwh"] = pv_surplus
    sim["bess_charge_kwh"] = charge
    sim["bess_discharge_self_kwh"] = discharge_self
    sim["bess_discharge_peak_kwh"] = discharge_peak
    sim["export_potential_kwh"] = pv_surplus
    sim["export_after_bess_kwh"] = export_after_bess
    sim["grid_after_kwh"] = grid_after
    sim["soc_kwh"] = soc_series

    pv_total = pv.sum()
    used_pv = direct.sum() + charge.sum()
    autokonsumpcja = used_pv / pv_total if pv_total else 0
    export_potential = pv_surplus.sum()
    export_after = export_after_bess.sum()
    peak_before = load.max()
    peak_after = grid_after.max()
    peak_reduction_kw = max(0, peak_before - peak_after)

    metrics = {
        "load_mwh": load.sum() / 1000,
        "pv_mwh": pv_total / 1000,
        "autokonsumpcja_pct": autokonsumpcja * 100,
        "export_potential_mwh": export_potential / 1000,
        "export_after_bess_mwh": export_after / 1000,
        "bess_charge_mwh": charge.sum() / 1000,
        "bess_discharge_mwh": (discharge_self.sum() + discharge_peak.sum()) / 1000,
        "peak_before_kw": peak_before,
        "peak_after_kw": peak_after,
        "peak_reduction_kw": peak_reduction_kw,
        "grid_after_mwh": grid_after.sum() / 1000,
    }
    return sim, metrics


def financials(metrics: Dict[str, float], pv_kwp: float, bess: BessConfig, params: Dict[str, float]) -> Dict[str, float]:
    energy_value = params["energy_price_pln_mwh"] + params["distribution_price_pln_mwh"]
    avoided_grid_mwh = metrics["pv_mwh"] - metrics["export_after_bess_mwh"]
    annual_energy_saving = avoided_grid_mwh * energy_value
    annual_capacity_saving = metrics["peak_reduction_kw"] * params["capacity_fee_pln_kw_month"] * 12
    annual_benefit = annual_energy_saving + annual_capacity_saving
    capex = pv_kwp * params["pv_capex_pln_kwp"] + bess.capacity_kwh * params["bess_capex_pln_kwh"]
    opex = capex * params["opex_pct"] / 100
    net_cash = annual_benefit - opex
    roi = net_cash / capex if capex else 0
    payback = capex / net_cash if net_cash > 0 else np.nan

    years = int(params["analysis_years"])
    debt_share = params["debt_share_pct"] / 100
    debt = capex * debt_share
    equity = capex - debt
    debt_service = debt / params["debt_years"] if params["debt_years"] > 0 else 0
    dscr = net_cash / debt_service if debt_service > 0 else np.nan

    cashflows = [-equity] + [net_cash - debt_service for _ in range(years)]
    try:
        irr = npf_irr(cashflows)
    except Exception:
        irr = np.nan

    saas_monthly_fee = params["saas_fee_pln_month"]
    saas_annual_net = annual_benefit - saas_monthly_fee * 12

    return {
        "annual_energy_saving_pln": annual_energy_saving,
        "annual_capacity_saving_pln": annual_capacity_saving,
        "annual_benefit_pln": annual_benefit,
        "capex_pln": capex,
        "annual_opex_pln": opex,
        "net_cash_pln": net_cash,
        "roi_pct": roi * 100,
        "payback_years": payback,
        "irr_pct": irr * 100 if not np.isnan(irr) else np.nan,
        "dscr": dscr,
        "saas_annual_net_pln": saas_annual_net,
    }


def npf_irr(values, guess=0.1):
    # Small dependency-free IRR approximation by binary search
    def npv(rate):
        return sum(v / ((1 + rate) ** i) for i, v in enumerate(values))
    low, high = -0.95, 2.0
    for _ in range(100):
        mid = (low + high) / 2
        if npv(mid) > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def optimize_variants(df, annual_yield, params, min_sc=80.0):
    rows = []
    pv_sizes = np.arange(params["pv_min_kwp"], params["pv_max_kwp"] + 1, params["pv_step_kwp"])
    bess_step = max(50, int(params.get("opt_bess_step_kwh", 250)))
    bess_max = max(0, int(params.get("opt_bess_max_kwh", 2000)))
    bess_sizes = list(range(0, bess_max + 1, bess_step))
    peak_target = params["peak_target_kw"]
    for pv in pv_sizes:
        for cap in bess_sizes:
            power = min(cap, params["bess_power_kw"]) if cap > 0 else 0
            bess = BessConfig(capacity_kwh=cap, power_kw=power, roundtrip_eff=params["bess_eff_pct"] / 100)
            _, m = simulate_pv_bess(df, pv, annual_yield, bess, peak_target)
            f = financials(m, pv, bess, params)
            rows.append({"PV kWp": pv, "BESS kWh": cap, "BESS kW": power, **m, **f})
    res = pd.DataFrame(rows)
    res["spełnia 80% SC"] = res["autokonsumpcja_pct"] >= min_sc
    return res.sort_values(["spełnia 80% SC", "net_cash_pln"], ascending=[False, False])


st.title("Energy Agent MVP — dobór PV + BESS")
st.caption("MVP: autokonsumpcja, eksport jako potencjał BESS, peak shaving, ROI, IRR, DSCR, SaaS/CAPEX, opłata mocowa. Bez arbitrażu cenowego.")

with st.sidebar:
    st.header("Dane wejściowe")
    uploaded = st.file_uploader("Wgraj CSV z profilem dobowo-godzinowym", type=["csv", "txt"])
    st.subheader("Założenia techniczne")
    pv_kwp = st.number_input("Moc PV [kWp]", 10, 10000, 1200, 10)
    annual_yield = st.number_input("Uzysk PV [kWh/kWp/rok]", 700, 1300, 1050, 10)
    bess_capacity = st.number_input("Pojemność BESS [kWh]", 0, 10000, 500, 50)
    bess_power = st.number_input("Moc BESS [kW]", 0, 10000, 500, 50)
    bess_eff = st.slider("Sprawność round-trip BESS [%]", 70, 98, 90)
    peak_target = st.number_input("Cel peak shaving — moc po redukcji [kW]", 0, 10000, 1000, 10)

    st.subheader("Optimizer — automatyczny dobór")
    optimizer_goal = st.selectbox(
        "Cel optymalizacji",
        [
            "Najwyższy cash flow netto",
            "Najwyższy ROI",
            "Najniższy eksport po BESS",
            "Największy peak shaving",
        ],
    )
    opt_pv_min = st.number_input("Optimizer: PV od [kWp]", 10, 10000, 500, 50)
    opt_pv_max = st.number_input("Optimizer: PV do [kWp]", 10, 10000, 2000, 50)
    opt_pv_step = st.number_input("Optimizer: krok PV [kWp]", 10, 1000, 100, 10)
    opt_bess_max = st.number_input("Optimizer: BESS do [kWh]", 0, 20000, 2000, 250)
    opt_bess_step = st.number_input("Optimizer: krok BESS [kWh]", 50, 5000, 250, 50)

    st.subheader("Założenia finansowe")
    energy_price = st.number_input("Energia czynna [PLN/MWh]", 0, 3000, 500, 10)
    distribution_price = st.number_input("Dystrybucja zmienna [PLN/MWh]", 0, 2000, 250, 10)
    capacity_fee = st.number_input("Korzyść z redukcji mocy / opłata mocowa [PLN/kW/mc]", 0.0, 500.0, 25.0, 1.0)
    pv_capex = st.number_input("CAPEX PV [PLN/kWp]", 500, 10000, 2800, 50)
    bess_capex = st.number_input("CAPEX BESS [PLN/kWh]", 500, 8000, 1600, 50)
    opex_pct = st.number_input("OPEX roczny [% CAPEX]", 0.0, 10.0, 1.5, 0.1)
    saas_fee = st.number_input("SaaS / abonament [PLN/mc]", 0, 1000000, 0, 1000)
    debt_share = st.slider("Udział długu [%]", 0, 100, 70)
    debt_years = st.number_input("Okres spłaty długu [lata]", 1, 30, 10, 1)
    analysis_years = st.number_input("Okres analizy IRR [lata]", 1, 30, 15, 1)

if not uploaded:
    st.info("Wgraj CSV, żeby uruchomić analizę.")
    st.stop()

try:
    df = load_csv(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

params = {
    "energy_price_pln_mwh": energy_price,
    "distribution_price_pln_mwh": distribution_price,
    "capacity_fee_pln_kw_month": capacity_fee,
    "pv_capex_pln_kwp": pv_capex,
    "bess_capex_pln_kwh": bess_capex,
    "opex_pct": opex_pct,
    "saas_fee_pln_month": saas_fee,
    "debt_share_pct": debt_share,
    "debt_years": debt_years,
    "analysis_years": analysis_years,
    "pv_min_kwp": int(opt_pv_min),
    "pv_max_kwp": int(max(opt_pv_min, opt_pv_max)),
    "pv_step_kwp": int(opt_pv_step),
    "opt_bess_max_kwh": int(opt_bess_max),
    "opt_bess_step_kwh": int(opt_bess_step),
    "bess_power_kw": bess_power,
    "bess_eff_pct": bess_eff,
    "peak_target_kw": peak_target,
}

bess = BessConfig(bess_capacity, bess_power, bess_eff / 100)
sim, metrics = simulate_pv_bess(df, pv_kwp, annual_yield, bess, peak_target)
fin = financials(metrics, pv_kwp, bess, params)

kpi = {**metrics, **fin}

st.subheader("Podsumowanie wariantu")
cols = st.columns(6)
cols[0].metric("Zużycie", f"{kpi['load_mwh']:,.0f} MWh".replace(",", " "))
cols[1].metric("Produkcja PV", f"{kpi['pv_mwh']:,.0f} MWh".replace(",", " "))
cols[2].metric("Autokonsumpcja", f"{kpi['autokonsumpcja_pct']:.1f}%", "cel min. 80%")
cols[3].metric("Eksport/potencjał BESS", f"{kpi['export_potential_mwh']:,.0f} MWh".replace(",", " "))
cols[4].metric("Redukcja peak", f"{kpi['peak_reduction_kw']:.0f} kW")
cols[5].metric("Korzyść roczna", f"{kpi['annual_benefit_pln']:,.0f} PLN".replace(",", " "))

cols2 = st.columns(5)
cols2[0].metric("CAPEX", f"{kpi['capex_pln']:,.0f} PLN".replace(",", " "))
cols2[1].metric("ROI", f"{kpi['roi_pct']:.1f}%")
cols2[2].metric("IRR", "—" if pd.isna(kpi['irr_pct']) else f"{kpi['irr_pct']:.1f}%")
cols2[3].metric("DSCR", "—" if pd.isna(kpi['dscr']) else f"{kpi['dscr']:.2f}")
cols2[4].metric("SaaS net/rok", f"{kpi['saas_annual_net_pln']:,.0f} PLN".replace(",", " "))

if kpi["autokonsumpcja_pct"] < 80:
    st.warning("Ten wariant nie spełnia celu minimalnej autokonsumpcji 80%. Zmniejsz PV albo zwiększ BESS.")
else:
    st.success("Ten wariant spełnia cel minimalnej autokonsumpcji 80%.")

st.subheader("Wykresy")

# 1) Miesięczne porównanie zużycia, PV i eksportu
monthly = sim.groupby("month", as_index=False).agg(
    load_mwh=("load_kwh", lambda x: x.sum()/1000),
    pv_mwh=("pv_kwh", lambda x: x.sum()/1000),
    export_potential_mwh=("export_potential_kwh", lambda x: x.sum()/1000),
    export_after_bess_mwh=("export_after_bess_kwh", lambda x: x.sum()/1000),
)
fig = go.Figure()
fig.add_bar(x=monthly["month"], y=monthly["load_mwh"], name="Zużycie")
fig.add_bar(x=monthly["month"], y=monthly["pv_mwh"], name="PV")
fig.add_bar(x=monthly["month"], y=monthly["export_potential_mwh"], name="Eksport potencjalny")
fig.update_layout(barmode="group", xaxis_title="Miesiąc", yaxis_title="MWh")
st.plotly_chart(fig, use_container_width=True)

# 2) Heatmapa godzinowa po dniach roku — najlepsza do rozmowy z klientem o profilu pracy
st.subheader("Heatmapa godzinowa zużycia — dzień roku × godzina")
sim["day_of_year"] = sim["timestamp"].dt.dayofyear
heat_year = sim.pivot_table(
    index="hour",
    columns="day_of_year",
    values="load_kwh",
    aggfunc="mean"
)
fig_heat_year = px.imshow(
    heat_year,
    aspect="auto",
    labels=dict(x="Dzień roku", y="Godzina", color="kWh/h"),
    title="Profil zużycia energii w układzie godzinowym"
)
fig_heat_year.update_yaxes(autorange="reversed")
st.plotly_chart(fig_heat_year, use_container_width=True)

# 3) Średni profil dobowy — osobno dni robocze i niedziele
st.subheader("Średni profil dobowy")
sim["typ_dnia"] = np.where(sim["is_sunday"], "Niedziela", "Pon.–sob.")
daily_profile = sim.groupby(["hour", "typ_dnia"], as_index=False).agg(
    load_kwh=("load_kwh", "mean"),
    pv_kwh=("pv_kwh", "mean"),
    grid_after_kwh=("grid_after_kwh", "mean"),
)
fig_daily = go.Figure()
for day_type in daily_profile["typ_dnia"].unique():
    part = daily_profile[daily_profile["typ_dnia"] == day_type]
    fig_daily.add_scatter(x=part["hour"], y=part["load_kwh"], mode="lines", name=f"Zużycie — {day_type}")
fig_daily.add_scatter(
    x=daily_profile.groupby("hour", as_index=False)["pv_kwh"].mean()["hour"],
    y=daily_profile.groupby("hour", as_index=False)["pv_kwh"].mean()["pv_kwh"],
    mode="lines",
    name="PV — średnio"
)
fig_daily.update_layout(xaxis_title="Godzina", yaxis_title="kWh/h")
st.plotly_chart(fig_daily, use_container_width=True)

# 4) Peak shaving — przed i po BESS
st.subheader("Peak shaving — przed i po PV+BESS")
peak_df = pd.DataFrame({
    "Stan": ["Przed PV+BESS", "Po PV+BESS"],
    "Moc [kW]": [kpi["peak_before_kw"], kpi["peak_after_kw"]],
})
fig_peak = px.bar(
    peak_df,
    x="Stan",
    y="Moc [kW]",
    text="Moc [kW]",
    title=f"Redukcja peaku: {kpi['peak_reduction_kw']:.0f} kW"
)
fig_peak.update_traces(texttemplate="%{text:.0f} kW", textposition="outside")
fig_peak.update_layout(yaxis_title="kW")
st.plotly_chart(fig_peak, use_container_width=True)

# 5) Próbka godzinowa — pierwsze 14 dni z baterią i poborem po optymalizacji
st.subheader("Przebieg godzinowy — próbka pierwszych 14 dni")
sample = sim.iloc[: min(24*14, len(sim))]
fig2 = go.Figure()
fig2.add_scatter(x=sample["timestamp"], y=sample["load_kwh"], name="Zużycie kWh")
fig2.add_scatter(x=sample["timestamp"], y=sample["pv_kwh"], name="PV kWh")
fig2.add_scatter(x=sample["timestamp"], y=sample["grid_after_kwh"], name="Pobór z sieci po PV+BESS")
fig2.add_scatter(x=sample["timestamp"], y=sample["soc_kwh"], name="SoC BESS kWh", yaxis="y2")
fig2.update_layout(
    xaxis_title="Czas",
    yaxis_title="kWh / h",
    yaxis2=dict(title="SoC BESS [kWh]", overlaying="y", side="right", showgrid=False),
)
st.plotly_chart(fig2, use_container_width=True)

# 6) Heatmapa typowego tygodnia — szybki obraz rytmu pracy zakładu
st.subheader("Heatmapa typowego tygodnia — średnie zużycie")
weekday_labels = {0: "Pon", 1: "Wt", 2: "Śr", 3: "Czw", 4: "Pt", 5: "Sob", 6: "Niedz"}
heat = sim.pivot_table(index="hour", columns="weekday", values="load_kwh", aggfunc="mean")
heat = heat.rename(columns=weekday_labels)
fig3 = px.imshow(heat, labels=dict(x="Dzień tygodnia", y="Godzina", color="Śr. kWh"), aspect="auto")
fig3.update_yaxes(autorange="reversed")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Automatyczny dobór PV + BESS")
st.caption("Optimizer testuje siatkę wariantów i wybiera konfigurację spełniającą cel min. 80% autokonsumpcji. Bez arbitrażu — BESS ładuje się z nadwyżki PV i pracuje pod self-consumption oraz peak shaving.")

goal_to_col = {
    "Najwyższy cash flow netto": ("net_cash_pln", False),
    "Najwyższy ROI": ("roi_pct", False),
    "Najniższy eksport po BESS": ("export_after_bess_mwh", True),
    "Największy peak shaving": ("peak_reduction_kw", False),
}

if st.button("Policz rekomendację i ranking wariantów"):
    with st.spinner("Liczenie wariantów PV+BESS..."):
        ranking = optimize_variants(df, annual_yield, params, min_sc=80.0)

    eligible = ranking[ranking["spełnia 80% SC"]].copy()
    if eligible.empty:
        st.error("Żaden wariant w zadanym zakresie nie spełnia celu 80% autokonsumpcji. Zwiększ zakres BESS albo zmniejsz zakres PV.")
        eligible = ranking.copy()

    sort_col, ascending = goal_to_col[optimizer_goal]
    recommended = eligible.sort_values(sort_col, ascending=ascending).iloc[0]

    st.success(f"Rekomendacja wg celu: {optimizer_goal}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("PV", f"{recommended['PV kWp']:.0f} kWp")
    c2.metric("BESS", f"{recommended['BESS kWh']:.0f} kWh / {recommended['BESS kW']:.0f} kW")
    c3.metric("Autokonsumpcja", f"{recommended['autokonsumpcja_pct']:.1f}%")
    c4.metric("Eksport po BESS", f"{recommended['export_after_bess_mwh']:.0f} MWh")
    c5.metric("Peak shaving", f"{recommended['peak_reduction_kw']:.0f} kW")
    c6.metric("ROI", f"{recommended['roi_pct']:.1f}%")

    show_cols = ["PV kWp", "BESS kWh", "BESS kW", "autokonsumpcja_pct", "export_potential_mwh", "export_after_bess_mwh", "peak_reduction_kw", "annual_benefit_pln", "net_cash_pln", "capex_pln", "roi_pct", "irr_pct", "dscr", "spełnia 80% SC"]

    top = eligible.sort_values(sort_col, ascending=ascending).head(20)
    fig_opt = px.scatter(
        top,
        x="PV kWp",
        y="BESS kWh",
        size="annual_benefit_pln",
        color="autokonsumpcja_pct",
        hover_data=["roi_pct", "irr_pct", "dscr", "peak_reduction_kw", "export_after_bess_mwh"],
        title="TOP warianty — wielkość punktu = korzyść roczna, kolor = autokonsumpcja",
    )
    st.plotly_chart(fig_opt, use_container_width=True)

    st.dataframe(top[show_cols], use_container_width=True)
    st.download_button("Pobierz pełny ranking CSV", ranking.to_csv(index=False).encode("utf-8-sig"), "ranking_pv_bess.csv", "text/csv")

st.subheader("Dane godzinowe po symulacji")
st.dataframe(sim.head(200), use_container_width=True)
st.download_button("Pobierz pełną symulację CSV", sim.to_csv(index=False).encode("utf-8-sig"), "symulacja_pv_bess.csv", "text/csv")
