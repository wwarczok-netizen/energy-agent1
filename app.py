import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Energy Agent MVP", page_icon="☀️", layout="wide")

st.title("☀️ Energy Agent MVP")
st.caption("MVP do szybkiego screeningu profilu zużycia: PV dla s.c. ≥ 80%, BESS, ROI i oszczędność.")

# -----------------------------
# Parametry użytkownika
# -----------------------------
with st.sidebar:
    st.header("Założenia")

    TARGET_SC = st.slider(
        "Minimalna autokonsumpcja / s.c. [%]",
        min_value=50,
        max_value=95,
        value=80,
        step=1,
    ) / 100

    ANNUAL_YIELD = st.number_input(
        "Uzysk PV [kWh/kWp/rok]",
        min_value=600,
        max_value=1400,
        value=1000,
        step=25,
    )

    ENERGY_PRICE = st.number_input(
        "Cena energii + dystrybucja [PLN/MWh]",
        min_value=0,
        max_value=3000,
        value=750,
        step=25,
    )

    PV_CAPEX = st.number_input(
        "CAPEX PV [PLN/kWp]",
        min_value=1000,
        max_value=10000,
        value=2800,
        step=100,
    )

    BESS_CAPEX = st.number_input(
        "CAPEX BESS [PLN/kWh]",
        min_value=500,
        max_value=5000,
        value=1600,
        step=100,
    )

# -----------------------------
# Funkcje pomocnicze
# -----------------------------
def read_input_file(uploaded_file):
    """Czyta CSV/XLS/XLSX."""
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file, sep=None, engine="python")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=None, engine="python", encoding="cp1250")

    if name.endswith(".xls"):
        return pd.read_excel(uploaded_file, engine="xlrd")

    return pd.read_excel(uploaded_file, engine="openpyxl")


def prepare_profile(df, time_col, value_col, unit):
    """Porządkuje profil zużycia i przelicza go do kWh/interwał."""
    profile = df[[time_col, value_col]].copy()
    profile.columns = ["timestamp", "value"]

    profile["timestamp"] = pd.to_datetime(
        profile["timestamp"],
        errors="coerce",
        dayfirst=True,
    )

    profile["value"] = pd.to_numeric(
        profile["value"],
        errors="coerce",
    )

    profile = profile.dropna().sort_values("timestamp").reset_index(drop=True)

    if profile.empty:
        raise ValueError("Nie udało się odczytać poprawnych danych czasu i zużycia.")

    deltas = profile["timestamp"].diff().dropna().dt.total_seconds() / 3600
    interval_h = float(deltas.median()) if len(deltas) else 1.0

    if unit == "kW":
        profile["load_kwh"] = profile["value"] * interval_h
    else:
        profile["load_kwh"] = profile["value"]

    profile["hour"] = profile["timestamp"].dt.hour
    profile["dayofyear"] = profile["timestamp"].dt.dayofyear

    return profile, interval_h


def build_pv_curve(profile, annual_yield):
    """
    Prosty syntetyczny profil PV 1 kWp.
    To jest screening handlowy, nie zamiennik PVsyst.
    """
    doy = profile["dayofyear"].to_numpy()
    hour = profile["hour"].to_numpy()

    seasonal = 0.55 + 0.45 * np.sin(2 * np.pi * (doy - 80) / 365)
    seasonal = np.clip(seasonal, 0.12, None)

    sun = np.sin(np.pi * (hour - 5) / 15)
    sun = np.clip(sun, 0, None) ** 1.7

    raw = seasonal * sun

    if raw.sum() == 0:
        raise ValueError("Nie udało się wygenerować profilu PV — sprawdź dane czasu.")

    days_covered = max(
        (profile["timestamp"].max() - profile["timestamp"].min()).days + 1,
        1,
    )

    expected_yield = annual_yield * days_covered / 365
    pv_1kwp = raw / raw.sum() * expected_yield

    return pv_1kwp, days_covered


def evaluate_pv(load, pv_1kwp, kwp):
    pv = pv_1kwp * kwp
    self_consumed = np.minimum(load, pv)
    export = np.maximum(pv - load, 0)

    pv_sum = pv.sum()
    load_sum = load.sum()
    self_sum = self_consumed.sum()

    sc = self_sum / pv_sum if pv_sum > 0 else 0
    coverage = self_sum / load_sum if load_sum > 0 else 0

    return {
        "kwp": kwp,
        "pv_kwh": pv_sum,
        "self_kwh": self_sum,
        "export_kwh": export.sum(),
        "sc": sc,
        "coverage": coverage,
    }


def find_max_pv_for_sc(profile, pv_1kwp, target_sc):
    load = profile["load_kwh"].to_numpy()
    annual_load = load.sum()

    rough_kwp = annual_load / max(pv_1kwp.sum(), 1)
    max_scan = max(10, rough_kwp * 3)

    best = None
    curve_rows = []

    for kwp in np.linspace(1, max_scan, 500):
        result = evaluate_pv(load, pv_1kwp, kwp)
        curve_rows.append(result)

        if result["sc"] >= target_sc:
            best = result

    return best, pd.DataFrame(curve_rows)


def calculate_bess_heuristic(profile, pv_1kwp, pv_kwp, days_covered):
    load = profile["load_kwh"].to_numpy()
    pv = pv_1kwp * pv_kwp

    export = np.maximum(pv - load, 0)
    daily_export = export.sum() / max(days_covered, 1)

    # Robocza heurystyka z MVP: część średniego dziennego eksportu jako użyteczna pojemność.
    suggested_bess_kwh = round(daily_export * 0.35)

    return suggested_bess_kwh


def make_excel(summary, curve):
    output = io.BytesIO()

    curve_out = curve.copy()
    curve_out["Autokonsumpcja_%"] = curve_out["sc"] * 100
    curve_out["Pokrycie_zużycia_%"] = curve_out["coverage"] * 100
    curve_out = curve_out.rename(
        columns={
            "kwp": "PV_kWp",
            "pv_kwh": "Produkcja_PV_kWh",
            "self_kwh": "Autokonsumpcja_kWh",
            "export_kwh": "Eksport_kWh",
        }
    )

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Wynik")
        curve_out.to_excel(writer, index=False, sheet_name="Krzywa_SC")

    return output.getvalue()


# -----------------------------
# Interfejs
# -----------------------------
uploaded_file = st.file_uploader(
    "Wgraj CSV, XLS lub XLSX z profilem zużycia",
    type=["csv", "xls", "xlsx"],
)

if uploaded_file is None:
    st.info("Wgraj plik z profilem zużycia. MVP wymaga jednej kolumny czasu i jednej kolumny wartości.")
    st.stop()

try:
    df = read_input_file(uploaded_file)

    st.subheader("Podgląd danych")
    st.dataframe(df.head(20), use_container_width=True)

    st.write("Wykryte kolumny:", list(df.columns))

    col1, col2, col3 = st.columns(3)

    with col1:
        time_col = st.selectbox("Kolumna czasu", df.columns)

    with col2:
        value_col = st.selectbox("Kolumna zużycia / mocy", df.columns)

    with col3:
        unit = st.radio("Jednostka danych", ["kWh", "kW"], horizontal=True)

    if st.button("Policz Energy Agent MVP", type="primary"):
        profile, interval_h = prepare_profile(df, time_col, value_col, unit)
        pv_1kwp, days_covered = build_pv_curve(profile, ANNUAL_YIELD)

        best, curve = find_max_pv_for_sc(profile, pv_1kwp, TARGET_SC)

        if best is None:
            st.error("Nie znaleziono mocy PV spełniającej zadany poziom autokonsumpcji.")
            st.stop()

        max_pv = round(best["kwp"])
        annual_load = profile["load_kwh"].sum()
        annual_self = best["self_kwh"]
        annual_saving = annual_self / 1000 * ENERGY_PRICE
        pv_capex_total = max_pv * PV_CAPEX
        roi = pv_capex_total / annual_saving if annual_saving > 0 else None

        suggested_bess = calculate_bess_heuristic(
            profile,
            pv_1kwp,
            max_pv,
            days_covered,
        )

        bess_capex_total = suggested_bess * BESS_CAPEX

        st.subheader("Wynik")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Max PV", f"{max_pv:,.0f} kWp")
        m2.metric("Autokonsumpcja / s.c.", f"{best['sc'] * 100:.1f}%")
        m3.metric("Produkcja PV", f"{best['pv_kwh'] / 1000:,.1f} MWh")
        m4.metric("ROI", f"{roi:.1f} lat" if roi else "brak")

        m5, m6, m7, m8 = st.columns(4)

        m5.metric("Roczne zużycie", f"{annual_load / 1000:,.1f} MWh")
        m6.metric("PV zużyte na miejscu", f"{annual_self / 1000:,.1f} MWh")
        m7.metric("Oszczędność roczna", f"{annual_saving:,.0f} PLN")
        m8.metric("Sugerowany BESS", f"{suggested_bess:,.0f} kWh")

        st.subheader("Ekonomia")
        econ = pd.DataFrame(
            [
                ["CAPEX PV", pv_capex_total],
                ["CAPEX BESS orientacyjny", bess_capex_total],
                ["Oszczędność roczna", annual_saving],
                ["ROI PV", roi],
            ],
            columns=["Parametr", "Wartość"],
        )
        st.dataframe(econ, use_container_width=True)

        st.subheader("Krzywa autokonsumpcji")
        chart_df = curve.copy()
        chart_df["Autokonsumpcja_%"] = chart_df["sc"] * 100
        chart_df["PV_kWp"] = chart_df["kwp"]

        st.line_chart(
            chart_df.set_index("PV_kWp")["Autokonsumpcja_%"],
            height=320,
        )

        summary = pd.DataFrame(
            [
                {
                    "Plik": uploaded_file.name,
                    "Interwał_h": interval_h,
                    "Dni_danych": days_covered,
                    "Zużycie_MWh": annual_load / 1000,
                    "Max_PV_kWp_SC": max_pv,
                    "SC_%": best["sc"] * 100,
                    "Produkcja_PV_MWh": best["pv_kwh"] / 1000,
                    "Autokonsumpcja_MWh": annual_self / 1000,
                    "Eksport_MWh": best["export_kwh"] / 1000,
                    "Pokrycie_zużycia_%": best["coverage"] * 100,
                    "Oszczędność_PLN_rok": annual_saving,
                    "CAPEX_PV_PLN": pv_capex_total,
                    "ROI_lata": roi,
                    "Sugerowany_BESS_kWh": suggested_bess,
                    "CAPEX_BESS_PLN": bess_capex_total,
                }
            ]
        )

        excel_bytes = make_excel(summary, curve)

        st.download_button(
            "Pobierz wynik XLSX",
            data=excel_bytes,
            file_name="energy_agent_mvp_wynik.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.caption(
            "Uwaga: model PV jest syntetyczny i służy do screeningu handlowego. "
            "Nie zastępuje PVsyst, projektu technicznego ani analizy przyłączeniowej."
        )

except Exception as e:
    st.error(f"Nie udało się przeliczyć pliku: {e}")
