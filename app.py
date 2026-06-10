
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="OZE PV Agent MVP",
    page_icon="☀️",
    layout="wide"
)

st.title("☀️ OZE PV Agent MVP")
st.caption("Profil zużycia → max PV dla autokonsumpcji ≥ 80% → potencjał BESS → oszczędność i ROI")

# -----------------------------
# Helpers
# -----------------------------

def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        raw = uploaded_file.read()
        # try common encodings and separators
        for enc in ["utf-8-sig", "utf-8", "cp1250", "latin1"]:
            try:
                text = raw.decode(enc)
                break
            except Exception:
                continue
        else:
            text = raw.decode("utf-8", errors="ignore")

        for sep in [";", ",", "\t"]:
            try:
                df = pd.read_csv(io.StringIO(text), sep=sep)
                if df.shape[1] >= 2:
                    return df
            except Exception:
                pass
        return pd.read_csv(io.StringIO(text))
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Obsługiwane pliki: CSV, XLSX, XLS")


def normalize_profile(df, time_col, value_col, unit):
    out = df[[time_col, value_col]].copy()
    out.columns = ["timestamp", "value"]
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", dayfirst=True)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["timestamp", "value"]).sort_values("timestamp")

    if len(out) < 2:
        raise ValueError("Za mało poprawnych danych po imporcie.")

    # Estimate interval in hours from timestamps
    deltas = out["timestamp"].diff().dropna().dt.total_seconds() / 3600
    interval_h = float(deltas.median())
    if interval_h <= 0 or interval_h > 24:
        interval_h = 1.0

    if unit == "kW":
        out["load_kwh"] = out["value"] * interval_h
    else:
        out["load_kwh"] = out["value"]

    out["hour"] = out["timestamp"].dt.hour
    out["month"] = out["timestamp"].dt.month
    out["dayofyear"] = out["timestamp"].dt.dayofyear
    return out[["timestamp", "load_kwh", "hour", "month", "dayofyear"]], interval_h


def synthetic_pv_profile(profile, annual_yield_kwh_per_kwp=1000, interval_h=1.0):
    """
    Creates normalized PV generation profile for 1 kWp.
    It is a pragmatic sales-screening approximation, not PVsyst.
    """
    df = profile.copy()
    doy = df["dayofyear"].to_numpy()
    hour = df["hour"].to_numpy()

    # daylight length approximation for Poland-ish latitude
    seasonal = 0.55 + 0.45 * np.sin(2 * np.pi * (doy - 80) / 365)
    seasonal = np.clip(seasonal, 0.12, None)

    # daily bell curve: production roughly 5-20 in summer, 8-16 in winter
    sun = np.sin(np.pi * (hour - 5) / 15)
    sun = np.clip(sun, 0, None) ** 1.7

    raw = seasonal * sun
    if raw.sum() == 0:
        raw = np.ones(len(df))

    # Scale to annual yield. If file covers less/more than full year, scale within available profile period.
    days_covered = max((df["timestamp"].max() - df["timestamp"].min()).days + 1, 1)
    expected_yield_for_period = annual_yield_kwh_per_kwp * days_covered / 365.0
    pv_1kwp = raw / raw.sum() * expected_yield_for_period
    return pv_1kwp


def evaluate_pv(profile, pv_1kwp, pv_kwp):
    pv = pv_1kwp * pv_kwp
    load = profile["load_kwh"].to_numpy()
    self_consumed = np.minimum(load, pv)
    export = np.maximum(pv - load, 0)
    import_energy = np.maximum(load - pv, 0)

    pv_prod = pv.sum()
    sc_ratio = self_consumed.sum() / pv_prod if pv_prod > 0 else 0
    load_coverage = self_consumed.sum() / load.sum() if load.sum() > 0 else 0

    return {
        "pv_kwp": pv_kwp,
        "pv_prod_kwh": pv_prod,
        "self_consumed_kwh": self_consumed.sum(),
        "export_kwh": export.sum(),
        "import_kwh": import_energy.sum(),
        "sc_ratio": sc_ratio,
        "load_coverage": load_coverage,
    }


def find_max_pv(profile, pv_1kwp, target_sc):
    annual_load = profile["load_kwh"].sum()
    estimated_yield = pv_1kwp.sum()
    if estimated_yield <= 0:
        return 0, pd.DataFrame()

    # broad search range based on load/yield
    rough_kwp = annual_load / estimated_yield
    max_kwp = max(50, rough_kwp * 3)
    candidates = np.unique(np.concatenate([
        np.linspace(1, 100, 100),
        np.linspace(100, max_kwp, 450)
    ]))

    rows = []
    feasible = []
    for kwp in candidates:
        res = evaluate_pv(profile, pv_1kwp, kwp)
        rows.append(res)
        if res["sc_ratio"] >= target_sc:
            feasible.append(res)

    curve = pd.DataFrame(rows)
    if not feasible:
        return 0, curve
    best = max(feasible, key=lambda x: x["pv_kwp"])
    return best["pv_kwp"], curve


def simulate_bess(profile, pv_1kwp, pv_kwp, bess_kwh, bess_power_kw=None, interval_h=1.0, rte=0.90):
    load = profile["load_kwh"].to_numpy()
    pv = pv_1kwp * pv_kwp
    if bess_power_kw is None:
        bess_power_kw = max(bess_kwh / 2, 1)  # 2h battery by default

    max_charge_per_step = bess_power_kw * interval_h
    max_discharge_per_step = bess_power_kw * interval_h

    soc = 0.0
    self_consumed = 0.0
    direct_self = 0.0
    batt_discharge_to_load = 0.0
    export = 0.0
    grid_import = 0.0

    for l, p in zip(load, pv):
        direct = min(l, p)
        direct_self += direct
        remaining_load = l - direct
        excess_pv = p - direct

        # charge battery from excess PV
        charge_possible = min(excess_pv, max_charge_per_step, bess_kwh - soc)
        soc += charge_possible * math.sqrt(rte)
        excess_after_charge = excess_pv - charge_possible

        # discharge battery to cover remaining load
        discharge_possible = min(remaining_load, max_discharge_per_step, soc)
        soc -= discharge_possible
        delivered = discharge_possible * math.sqrt(rte)
        batt_discharge_to_load += delivered

        grid_import += max(remaining_load - delivered, 0)
        export += max(excess_after_charge, 0)

    self_consumed = direct_self + batt_discharge_to_load
    pv_prod = pv.sum()
    sc_ratio = self_consumed / pv_prod if pv_prod > 0 else 0
    load_coverage = self_consumed / load.sum() if load.sum() > 0 else 0

    return {
        "bess_kwh": bess_kwh,
        "bess_power_kw": bess_power_kw,
        "pv_kwp": pv_kwp,
        "pv_prod_kwh": pv_prod,
        "self_consumed_kwh": self_consumed,
        "direct_self_kwh": direct_self,
        "bess_to_load_kwh": batt_discharge_to_load,
        "export_kwh": export,
        "import_kwh": grid_import,
        "sc_ratio": sc_ratio,
        "load_coverage": load_coverage
    }


def format_pln(x):
    return f"{x:,.0f} PLN".replace(",", " ")


def format_mwh(x):
    return f"{x/1000:,.1f} MWh".replace(",", " ")


# -----------------------------
# Sidebar assumptions
# -----------------------------
st.sidebar.header("Założenia")

target_sc = st.sidebar.slider("Minimalna autokonsumpcja PV", 0.50, 0.98, 0.80, 0.01)
annual_yield = st.sidebar.number_input("Uzysk PV [kWh/kWp/rok]", min_value=700, max_value=1300, value=1000, step=10)
energy_price = st.sidebar.number_input("Wartość 1 MWh zaoszczędzonej energii [PLN/MWh]", min_value=100, max_value=3000, value=750, step=25)
pv_capex = st.sidebar.number_input("CAPEX PV [PLN/kWp netto]", min_value=1000, max_value=6000, value=2800, step=50)
bess_capex = st.sidebar.number_input("CAPEX BESS [PLN/kWh netto]", min_value=500, max_value=5000, value=1600, step=50)
om_percent = st.sidebar.number_input("O&M rocznie [% CAPEX PV]", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
target_bess_sc = st.sidebar.slider("Docelowa autokonsumpcja po BESS", 0.80, 0.99, 0.90, 0.01)

uploaded = st.file_uploader("Wgraj profil zużycia CSV/XLSX", type=["csv", "xlsx", "xls"])

with st.expander("Format danych"):
    st.write("""
    Minimalnie potrzebne są dwie kolumny:
    - data/czas, np. `timestamp`, `Data`, `Czas`,
    - zużycie energii w `kWh` albo moc średnia w `kW`.

    Dane mogą być godzinowe albo 15-minutowe. Przy kolumnie w kW aplikacja przeliczy energię po interwale.
    """)

if uploaded is None:
    st.info("Wgraj profil zużycia, aby policzyć analizę.")
    st.stop()

try:
    raw_df = read_uploaded_file(uploaded)
except Exception as e:
    st.error(f"Nie udało się odczytać pliku: {e}")
    st.stop()

st.subheader("1) Mapowanie kolumn")
cols = list(raw_df.columns)

c1, c2, c3 = st.columns(3)
with c1:
    time_col = st.selectbox("Kolumna daty/czasu", cols)
with c2:
    value_col = st.selectbox("Kolumna zużycia/mocy", cols, index=min(1, len(cols)-1))
with c3:
    unit = st.radio("Jednostka kolumny", ["kWh", "kW"], horizontal=True)

try:
    profile, interval_h = normalize_profile(raw_df, time_col, value_col, unit)
except Exception as e:
    st.error(f"Problem z normalizacją danych: {e}")
    st.stop()

pv_1kwp = synthetic_pv_profile(profile, annual_yield, interval_h)
max_pv_kwp, curve = find_max_pv(profile, pv_1kwp, target_sc)

if max_pv_kwp <= 0:
    st.warning("Nie znaleziono PV spełniającej zadany próg autokonsumpcji. Sprawdź dane wejściowe lub obniż próg.")
    st.stop()

base = evaluate_pv(profile, pv_1kwp, max_pv_kwp)

# BESS scan
bess_candidates = np.concatenate([[0], np.linspace(50, max(max_pv_kwp * 2, 100), 40)])
bess_rows = [simulate_bess(profile, pv_1kwp, max_pv_kwp, b, interval_h=interval_h) for b in bess_candidates]
bess_df = pd.DataFrame(bess_rows)

# Candidate BESS: first capacity reaching target SC, otherwise best practical
target_rows = bess_df[bess_df["sc_ratio"] >= target_bess_sc]
if len(target_rows):
    bess_best = target_rows.iloc[0].to_dict()
else:
    # point where incremental self-consumption gains flatten
    bess_df["gain_kwh"] = bess_df["self_consumed_kwh"].diff().fillna(0)
    bess_df["gain_per_kwh_bess"] = bess_df["gain_kwh"] / bess_df["bess_kwh"].diff().replace(0, np.nan)
    useful = bess_df[(bess_df["bess_kwh"] > 0) & (bess_df["gain_per_kwh_bess"] > 100)]
    bess_best = (useful.iloc[-1] if len(useful) else bess_df.iloc[min(10, len(bess_df)-1)]).to_dict()

# Economics
annual_self_no_bess = base["self_consumed_kwh"]
annual_saving_no_bess = annual_self_no_bess / 1000 * energy_price
pv_capex_total = max_pv_kwp * pv_capex
om_annual = pv_capex_total * om_percent / 100
net_saving_no_bess = annual_saving_no_bess - om_annual
roi_no_bess = pv_capex_total / net_saving_no_bess if net_saving_no_bess > 0 else np.nan

annual_self_with_bess = bess_best["self_consumed_kwh"]
annual_saving_with_bess = annual_self_with_bess / 1000 * energy_price
bess_capex_total = bess_best["bess_kwh"] * bess_capex
total_capex_with_bess = pv_capex_total + bess_capex_total
net_saving_with_bess = annual_saving_with_bess - om_annual
roi_with_bess = total_capex_with_bess / net_saving_with_bess if net_saving_with_bess > 0 else np.nan

# -----------------------------
# Results
# -----------------------------
st.subheader("2) Wynik główny")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Max PV dla s.c. ≥ progu", f"{max_pv_kwp:,.0f} kWp".replace(",", " "))
m2.metric("Autokonsumpcja PV", f"{base['sc_ratio']*100:.1f}%")
m3.metric("Produkcja PV", format_mwh(base["pv_prod_kwh"]))
m4.metric("Zużycie roczne z profilu", format_mwh(profile["load_kwh"].sum()))

m5, m6, m7, m8 = st.columns(4)
m5.metric("Energia zużyta z PV", format_mwh(base["self_consumed_kwh"]))
m6.metric("Pokrycie zużycia PV", f"{base['load_coverage']*100:.1f}%")
m7.metric("Oszczędność brutto / rok", format_pln(annual_saving_no_bess))
m8.metric("Prosty ROI PV", f"{roi_no_bess:.1f} lat" if not np.isnan(roi_no_bess) else "n/a")

st.subheader("3) Potencjalny BESS")

b1, b2, b3, b4 = st.columns(4)
b1.metric("Sugerowany BESS", f"{bess_best['bess_kwh']:,.0f} kWh".replace(",", " "))
b2.metric("Sugerowana moc PCS", f"{bess_best['bess_power_kw']:,.0f} kW".replace(",", " "))
b3.metric("Autokonsumpcja z BESS", f"{bess_best['sc_ratio']*100:.1f}%")
b4.metric("ROI PV+BESS", f"{roi_with_bess:.1f} lat" if not np.isnan(roi_with_bess) else "n/a")

b5, b6, b7, b8 = st.columns(4)
b5.metric("Dodatkowa energia z BESS", format_mwh(bess_best["bess_to_load_kwh"]))
b6.metric("Oszczędność brutto PV+BESS / rok", format_pln(annual_saving_with_bess))
b7.metric("CAPEX PV", format_pln(pv_capex_total))
b8.metric("CAPEX PV+BESS", format_pln(total_capex_with_bess))

st.subheader("4) Wykresy")

curve_show = curve.copy()
curve_show["Autokonsumpcja [%]"] = curve_show["sc_ratio"] * 100
curve_show["PV [kWp]"] = curve_show["pv_kwp"]

fig1 = px.line(curve_show, x="PV [kWp]", y="Autokonsumpcja [%]", title="Autokonsumpcja względem wielkości PV")
fig1.add_hline(y=target_sc*100, line_dash="dash", annotation_text=f"Próg {target_sc*100:.0f}%")
st.plotly_chart(fig1, use_container_width=True)

bess_show = bess_df.copy()
bess_show["BESS [kWh]"] = bess_show["bess_kwh"]
bess_show["Autokonsumpcja [%]"] = bess_show["sc_ratio"] * 100
fig2 = px.line(bess_show, x="BESS [kWh]", y="Autokonsumpcja [%]", title="Wpływ BESS na autokonsumpcję")
fig2.add_hline(y=target_bess_sc*100, line_dash="dash", annotation_text=f"Cel {target_bess_sc*100:.0f}%")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("5) Tabela wynikowa")

summary = pd.DataFrame([
    ["PV bez BESS", max_pv_kwp, 0, base["sc_ratio"], base["load_coverage"], base["pv_prod_kwh"], base["self_consumed_kwh"], annual_saving_no_bess, pv_capex_total, roi_no_bess],
    ["PV + BESS", max_pv_kwp, bess_best["bess_kwh"], bess_best["sc_ratio"], bess_best["load_coverage"], bess_best["pv_prod_kwh"], bess_best["self_consumed_kwh"], annual_saving_with_bess, total_capex_with_bess, roi_with_bess],
], columns=[
    "Scenariusz", "PV_kWp", "BESS_kWh", "Autokonsumpcja", "Pokrycie_zuzycia",
    "Produkcja_PV_kWh", "Zuzycie_z_PV_kWh", "Oszczednosc_brutto_PLN_rok", "CAPEX_PLN", "ROI_lata"
])

st.dataframe(summary, use_container_width=True)

csv = summary.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Pobierz wyniki CSV",
    data=csv,
    file_name="wyniki_oze_pv_agent.csv",
    mime="text/csv"
)

st.warning("""
To jest MVP do screeningu handlowego. Produkcja PV jest liczona syntetycznie, a nie z PVsyst.
Do oferty wiążącej trzeba podmienić moduł produkcji na dane PVGIS/PVsyst oraz doprecyzować taryfy, opłaty dystrybucyjne, profil 15-min i zasady eksportu.
""")
