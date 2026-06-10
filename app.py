import io
import math
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Energy Agent — PV/BESS MVP+", page_icon="⚡", layout="wide")

st.title("⚡ Energy Agent — PV/BESS MVP+")
st.caption(
    "Analiza profilu zużycia: dobór PV pod autokonsumpcję, symulacja BESS, ROI, "
    "oszczędności, eksport, pokrycie zużycia i krzywe wariantowe."
)

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Założenia PV")

    target_sc = st.slider("Minimalna autokonsumpcja / s.c. [%]", 50, 99, 80, 1) / 100
    annual_yield = st.number_input("Uzysk PV [kWh/kWp/rok]", 600, 1400, 1050, 25)
    pv_step_kwp = st.number_input("Krok skanowania PV [kWp]", 1, 100, 10, 1)
    max_pv_user = st.number_input("Maksymalna moc PV do skanowania [kWp, 0 = auto]", 0, 20000, 0, 50)

    st.header("Ekonomia")
    energy_price = st.number_input("Cena energii + dystrybucja [PLN/MWh]", 0, 3000, 750, 25)
    export_price = st.number_input("Cena sprzedaży nadwyżek [PLN/MWh]", 0, 1500, 0, 25)
    pv_capex = st.number_input("CAPEX PV [PLN/kWp]", 500, 10000, 2800, 100)
    bess_capex = st.number_input("CAPEX BESS [PLN/kWh]", 500, 5000, 1600, 100)
    om_percent = st.number_input("O&M PV [% CAPEX/rok]", 0.0, 10.0, 1.0, 0.1)
    degradation = st.number_input("Degradacja PV [%/rok]", 0.0, 3.0, 0.5, 0.1)

    st.header("BESS")
    bess_enabled = st.checkbox("Licz warianty BESS", value=True)
    bess_efficiency = st.slider("Sprawność round-trip BESS [%]", 70, 98, 90, 1) / 100
    bess_kw_options_text = st.text_input("Warianty mocy BESS [kW]", "100,250,500,1000")
    bess_kwh_options_text = st.text_input("Warianty pojemności BESS [kWh]", "200,500,1000,2000,4000")

    st.header("Filtry profilu")
    exclude_sundays = st.checkbox("Policz wariant bez niedziel", value=False)
    exclude_weekends = st.checkbox("Policz wariant bez weekendów", value=False)

# ============================================================
# Helpers
# ============================================================
def parse_number_list(text):
    out = []
    for part in str(text).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = float(part)
            if value > 0:
                out.append(value)
        except ValueError:
            pass
    return sorted(set(out))


def read_input_file(uploaded_file):
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


def detect_ppe_column(df):
    for col in df.columns:
        name = str(col).lower()
        if "ppe" in name or "punkt" in name or "licznik" in name or "meter" in name:
            return col

    for col in df.columns:
        vals = df[col].dropna().astype(str).head(5000)
        hits = vals.str.contains(r"\b\d{18}\b", regex=True).sum()
        if hits >= 5:
            return col

    return None


def prepare_profile(df, time_col, value_col, unit, ppe_col=None, ppe_value=None):
    work = df.copy()

    if ppe_col is not None and ppe_value is not None:
        work = work[work[ppe_col].astype(str) == str(ppe_value)].copy()

    profile = work[[time_col, value_col]].copy()
    profile.columns = ["timestamp", "value"]

    profile["timestamp"] = pd.to_datetime(profile["timestamp"], errors="coerce", dayfirst=True)
    profile["value"] = (
        profile["value"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
    )
    profile["value"] = pd.to_numeric(profile["value"], errors="coerce")

    profile = profile.dropna().sort_values("timestamp").reset_index(drop=True)

    if profile.empty:
        raise ValueError("Brak poprawnych danych czasu i zużycia po oczyszczeniu pliku.")

    if exclude_sundays:
        profile = profile[profile["timestamp"].dt.dayofweek != 6].copy()

    if exclude_weekends:
        profile = profile[profile["timestamp"].dt.dayofweek < 5].copy()

    deltas = profile["timestamp"].diff().dropna().dt.total_seconds() / 3600
    interval_h = float(deltas.median()) if len(deltas) else 1.0
    interval_h = max(interval_h, 1 / 60)

    if unit == "kW":
        profile["load_kwh"] = profile["value"] * interval_h
    else:
        profile["load_kwh"] = profile["value"]

    profile = profile[profile["load_kwh"] >= 0].copy()
    profile["hour_float"] = profile["timestamp"].dt.hour + profile["timestamp"].dt.minute / 60
    profile["hour"] = profile["timestamp"].dt.hour
    profile["dayofyear"] = profile["timestamp"].dt.dayofyear
    profile["date"] = profile["timestamp"].dt.date
    profile["month"] = profile["timestamp"].dt.to_period("M").astype(str)
    profile["weekday"] = profile["timestamp"].dt.day_name()

    return profile, interval_h


def build_pv_1kwp(profile, annual_yield):
    doy = profile["dayofyear"].to_numpy()
    hour = profile["hour_float"].to_numpy()

    seasonal = 0.55 + 0.45 * np.sin(2 * np.pi * (doy - 80) / 365)
    seasonal = np.clip(seasonal, 0.08, None)

    sun = np.sin(np.pi * (hour - 5) / 15)
    sun = np.clip(sun, 0, None) ** 1.7

    raw = seasonal * sun

    if raw.sum() <= 0:
        raise ValueError("Nie udało się wygenerować profilu PV.")

    days_covered = max((profile["timestamp"].max() - profile["timestamp"].min()).days + 1, 1)
    expected_yield = annual_yield * days_covered / 365

    return raw / raw.sum() * expected_yield, days_covered


def evaluate(load, pv):
    self_direct = np.minimum(load, pv)
    export = np.maximum(pv - load, 0)
    grid = np.maximum(load - pv, 0)

    pv_sum = pv.sum()
    load_sum = load.sum()
    self_sum = self_direct.sum()

    return {
        "load_kwh": load_sum,
        "pv_kwh": pv_sum,
        "self_kwh": self_sum,
        "export_kwh": export.sum(),
        "grid_kwh": grid.sum(),
        "sc": self_sum / pv_sum if pv_sum > 0 else 0,
        "coverage": self_sum / load_sum if load_sum > 0 else 0,
    }


def scan_pv(profile, pv_1kwp, target_sc, step_kwp, max_pv_user):
    load = profile["load_kwh"].to_numpy()
    annual_load = load.sum()
    rough_kwp = annual_load / max(pv_1kwp.sum(), 1)

    max_scan = max_pv_user if max_pv_user and max_pv_user > 0 else max(10, rough_kwp * 3)
    grid = np.arange(step_kwp, max_scan + step_kwp, step_kwp)

    rows = []
    best = None
    for kwp in grid:
        pv = pv_1kwp * kwp
        r = evaluate(load, pv)
        r["pv_kwp"] = float(kwp)
        rows.append(r)
        if r["sc"] >= target_sc:
            best = r

    return best, pd.DataFrame(rows)


def simulate_bess(profile, pv, bess_kw, bess_kwh, efficiency):
    load = profile["load_kwh"].to_numpy()
    timestamps = profile["timestamp"]
    deltas = timestamps.diff().dropna().dt.total_seconds() / 3600
    interval_h = float(deltas.median()) if len(deltas) else 1.0
    interval_h = max(interval_h, 1 / 60)

    charge_eff = math.sqrt(efficiency)
    discharge_eff = math.sqrt(efficiency)

    soc = 0.0
    bess_charge_from_pv = []
    bess_discharge_to_load = []
    soc_series = []
    curtailed_or_export = []
    grid_after = []

    max_charge_per_step = bess_kw * interval_h
    max_discharge_per_step = bess_kw * interval_h

    for l, p in zip(load, pv):
        surplus = max(p - l, 0)
        deficit = max(l - p, 0)

        charge_input = min(surplus, max_charge_per_step, max((bess_kwh - soc) / charge_eff, 0))
        soc += charge_input * charge_eff

        remaining_surplus = surplus - charge_input

        discharge_from_battery = min(soc, max_discharge_per_step, deficit / discharge_eff if discharge_eff > 0 else 0)
        delivered = discharge_from_battery * discharge_eff
        soc -= discharge_from_battery

        remaining_deficit = deficit - delivered

        bess_charge_from_pv.append(charge_input)
        bess_discharge_to_load.append(delivered)
        soc_series.append(soc)
        curtailed_or_export.append(remaining_surplus)
        grid_after.append(remaining_deficit)

    direct_self = np.minimum(load, pv)
    bess_to_load = np.array(bess_discharge_to_load)
    export_after = np.array(curtailed_or_export)
    grid_after = np.array(grid_after)
    charge_from_pv = np.array(bess_charge_from_pv)

    self_total = direct_self.sum() + bess_to_load.sum()
    pv_sum = pv.sum()
    load_sum = load.sum()

    return {
        "bess_kw": bess_kw,
        "bess_kwh": bess_kwh,
        "direct_self_kwh": direct_self.sum(),
        "bess_charge_kwh": charge_from_pv.sum(),
        "bess_discharge_kwh": bess_to_load.sum(),
        "self_total_kwh": self_total,
        "export_after_kwh": export_after.sum(),
        "grid_after_kwh": grid_after.sum(),
        "sc_with_bess": self_total / pv_sum if pv_sum > 0 else 0,
        "coverage_with_bess": self_total / load_sum if load_sum > 0 else 0,
        "soc_max_kwh": max(soc_series) if soc_series else 0,
        "cycles_equiv": charge_from_pv.sum() / bess_kwh if bess_kwh > 0 else 0,
    }


def run_bess_variants(profile, pv, kw_options, kwh_options, efficiency):
    rows = []
    for kw in kw_options:
        for kwh in kwh_options:
            rows.append(simulate_bess(profile, pv, kw, kwh, efficiency))
    return pd.DataFrame(rows)


def economics(base_eval, pv_kwp, bess_row=None):
    pv_capex_total = pv_kwp * pv_capex
    bess_capex_total = 0 if bess_row is None else bess_row["bess_kwh"] * bess_capex

    used_kwh = base_eval["self_kwh"] if bess_row is None else bess_row["self_total_kwh"]
    export_kwh = base_eval["export_kwh"] if bess_row is None else bess_row["export_after_kwh"]

    savings = used_kwh / 1000 * energy_price
    export_revenue = export_kwh / 1000 * export_price
    om = pv_capex_total * om_percent / 100
    net_savings = savings + export_revenue - om
    total_capex = pv_capex_total + bess_capex_total
    roi = total_capex / net_savings if net_savings > 0 else np.nan

    return {
        "CAPEX_PV_PLN": pv_capex_total,
        "CAPEX_BESS_PLN": bess_capex_total,
        "CAPEX_total_PLN": total_capex,
        "Oszczędność_PLN_rok": savings,
        "Przychód_z_eksportu_PLN_rok": export_revenue,
        "O&M_PLN_rok": om,
        "Korzyść_netto_PLN_rok": net_savings,
        "ROI_lata": roi,
    }


def make_excel(summary, pv_curve, bess_table, monthly):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, index=False, sheet_name="Podsumowanie")
        pv_curve.to_excel(writer, index=False, sheet_name="Krzywa_PV_SC")
        monthly.to_excel(writer, index=False, sheet_name="Miesięcznie")
        if bess_table is not None and not bess_table.empty:
            bess_table.to_excel(writer, index=False, sheet_name="BESS_warianty")
    return output.getvalue()


# ============================================================
# Upload
# ============================================================
uploaded = st.file_uploader("Wgraj CSV / XLS / XLSX z profilem zużycia", type=["csv", "xls", "xlsx"])

if uploaded is None:
    st.info("Wgraj profil zużycia. Aplikacja pozwoli wybrać kolumnę czasu, zużycia oraz opcjonalnie PPE.")
    st.stop()

try:
    df = read_input_file(uploaded)

    st.subheader("1. Dane wejściowe")
    st.dataframe(df.head(30), use_container_width=True)

    ppe_guess = detect_ppe_column(df)
    use_ppe = False
    ppe_col = None
    ppe_value = None

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        time_col = st.selectbox("Kolumna czasu", df.columns)

    with c2:
        value_col = st.selectbox("Kolumna zużycia / mocy", df.columns)

    with c3:
        unit = st.radio("Jednostka", ["kWh", "kW"], horizontal=True)

    with c4:
        if ppe_guess is not None:
            use_ppe = st.checkbox(f"Dziel po PPE ({ppe_guess})", value=True)
            ppe_col = ppe_guess
        else:
            use_ppe = st.checkbox("W pliku jest kolumna PPE", value=False)
            if use_ppe:
                ppe_col = st.selectbox("Kolumna PPE", df.columns)

    if use_ppe and ppe_col is not None:
        ppes = sorted(df[ppe_col].dropna().astype(str).unique())
        ppe_mode = st.radio("Tryb PPE", ["Wszystkie razem", "Wybrane PPE"], horizontal=True)
        if ppe_mode == "Wybrane PPE":
            ppe_value = st.selectbox("Wybierz PPE", ppes)
        else:
            ppe_value = None

    if st.button("Policz pełną analizę", type="primary"):
        profiles = []

        if use_ppe and ppe_col is not None and ppe_value is None:
            for ppe in sorted(df[ppe_col].dropna().astype(str).unique()):
                prof, interval_h = prepare_profile(df, time_col, value_col, unit, ppe_col, ppe)
                profiles.append((str(ppe), prof, interval_h))
        else:
            label = str(ppe_value) if ppe_value is not None else "SUMA"
            prof, interval_h = prepare_profile(df, time_col, value_col, unit, ppe_col if ppe_value else None, ppe_value)
            profiles.append((label, prof, interval_h))

        all_summaries = []
        all_curves = []
        all_bess = []
        all_monthly = []

        for label, profile, interval_h in profiles:
            pv_1kwp, days_covered = build_pv_1kwp(profile, annual_yield)
            best, pv_curve = scan_pv(profile, pv_1kwp, target_sc, pv_step_kwp, max_pv_user)

            if best is None:
                st.warning(f"{label}: nie znaleziono PV spełniającego s.c. >= {target_sc*100:.0f}%.")
                continue

            pv_kwp = round(best["pv_kwp"], 2)
            pv = pv_1kwp * pv_kwp
            load = profile["load_kwh"].to_numpy()
            base = evaluate(load, pv)
            base_econ = economics(base, pv_kwp)

            bess_table = pd.DataFrame()
            best_bess = None

            if bess_enabled:
                kw_options = parse_number_list(bess_kw_options_text)
                kwh_options = parse_number_list(bess_kwh_options_text)
                if kw_options and kwh_options:
                    bess_table = run_bess_variants(profile, pv, kw_options, kwh_options, bess_efficiency)

                    # Korzyść wariantu BESS względem PV only
                    for k, v in base_econ.items():
                        bess_table[f"PV_only_{k}"] = v

                    econ_rows = []
                    for _, row in bess_table.iterrows():
                        econ_rows.append(economics(base, pv_kwp, row))
                    econ_df = pd.DataFrame(econ_rows)
                    bess_table = pd.concat([bess_table.reset_index(drop=True), econ_df.reset_index(drop=True)], axis=1)

                    bess_table["Dodatkowa_autokonsumpcja_MWh"] = (
                        bess_table["self_total_kwh"] - base["self_kwh"]
                    ) / 1000
                    bess_table["Wzrost_SC_pp"] = (bess_table["sc_with_bess"] - base["sc"]) * 100
                    bess_table["Wzrost_pokrycia_pp"] = (bess_table["coverage_with_bess"] - base["coverage"]) * 100

                    # Wybór rekomendacji: najlepsza dodatkowa autokonsumpcja przy ROI dodatnim,
                    # a jak brak sensownego ROI, wariant z największym wzrostem SC.
                    positive = bess_table[bess_table["Korzyść_netto_PLN_rok"] > base_econ["Korzyść_netto_PLN_rok"]].copy()
                    if not positive.empty:
                        best_bess = positive.sort_values(["ROI_lata", "Dodatkowa_autokonsumpcja_MWh"], ascending=[True, False]).iloc[0].to_dict()
                    else:
                        best_bess = bess_table.sort_values("Wzrost_SC_pp", ascending=False).iloc[0].to_dict()

            monthly = profile.copy()
            monthly["PV_kWh"] = pv
            monthly["Autokonsumpcja_direct_kWh"] = np.minimum(monthly["load_kwh"], monthly["PV_kWh"])
            monthly["Eksport_direct_kWh"] = np.maximum(monthly["PV_kWh"] - monthly["load_kwh"], 0)
            monthly["Pobór_z_sieci_direct_kWh"] = np.maximum(monthly["load_kwh"] - monthly["PV_kWh"], 0)
            monthly_out = monthly.groupby("month", as_index=False).agg(
                Zużycie_kWh=("load_kwh", "sum"),
                PV_kWh=("PV_kWh", "sum"),
                Autokonsumpcja_kWh=("Autokonsumpcja_direct_kWh", "sum"),
                Eksport_kWh=("Eksport_direct_kWh", "sum"),
                Pobór_z_sieci_kWh=("Pobór_z_sieci_direct_kWh", "sum"),
            )
            monthly_out.insert(0, "Profil", label)

            summary = {
                "Profil": label,
                "Interwał_h": interval_h,
                "Dni_danych": days_covered,
                "Zużycie_MWh": base["load_kwh"] / 1000,
                "Max_PV_kWp_SC_target": pv_kwp,
                "Produkcja_PV_MWh": base["pv_kwh"] / 1000,
                "Autokonsumpcja_direct_MWh": base["self_kwh"] / 1000,
                "Eksport_direct_MWh": base["export_kwh"] / 1000,
                "Pobór_z_sieci_po_PV_MWh": base["grid_kwh"] / 1000,
                "SC_direct_%": base["sc"] * 100,
                "Pokrycie_zużycia_direct_%": base["coverage"] * 100,
                **base_econ,
                "Rekomendowany_BESS_kW": None,
                "Rekomendowany_BESS_kWh": None,
                "SC_z_BESS_%": None,
                "Pokrycie_z_BESS_%": None,
                "Autokonsumpcja_z_BESS_MWh": None,
                "Eksport_z_BESS_MWh": None,
                "Dodatkowa_autokonsumpcja_BESS_MWh": None,
                "ROI_z_BESS_lata": None,
            }

            if best_bess:
                summary.update({
                    "Rekomendowany_BESS_kW": best_bess["bess_kw"],
                    "Rekomendowany_BESS_kWh": best_bess["bess_kwh"],
                    "SC_z_BESS_%": best_bess["sc_with_bess"] * 100,
                    "Pokrycie_z_BESS_%": best_bess["coverage_with_bess"] * 100,
                    "Autokonsumpcja_z_BESS_MWh": best_bess["self_total_kwh"] / 1000,
                    "Eksport_z_BESS_MWh": best_bess["export_after_kwh"] / 1000,
                    "Dodatkowa_autokonsumpcja_BESS_MWh": best_bess["Dodatkowa_autokonsumpcja_MWh"],
                    "ROI_z_BESS_lata": best_bess["ROI_lata"],
                })

            pv_curve_out = pv_curve.copy()
            pv_curve_out.insert(0, "Profil", label)
            pv_curve_out["SC_%"] = pv_curve_out["sc"] * 100
            pv_curve_out["Pokrycie_zużycia_%"] = pv_curve_out["coverage"] * 100

            if not bess_table.empty:
                bess_table.insert(0, "Profil", label)

            all_summaries.append(summary)
            all_curves.append(pv_curve_out)
            all_monthly.append(monthly_out)
            if not bess_table.empty:
                all_bess.append(bess_table)

        if not all_summaries:
            st.error("Nie udało się policzyć żadnego profilu.")
            st.stop()

        summary_df = pd.DataFrame(all_summaries)
        pv_curve_df = pd.concat(all_curves, ignore_index=True)
        monthly_df = pd.concat(all_monthly, ignore_index=True)
        bess_df = pd.concat(all_bess, ignore_index=True) if all_bess else pd.DataFrame()

        st.subheader("2. Podsumowanie")
        st.dataframe(summary_df, use_container_width=True)

        # Metryki dla pierwszego profilu / sumy
        row = summary_df.iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max PV dla s.c.", f"{row['Max_PV_kWp_SC_target']:,.0f} kWp")
        m2.metric("SC direct", f"{row['SC_direct_%']:.1f}%")
        m3.metric("Produkcja PV", f"{row['Produkcja_PV_MWh']:,.1f} MWh")
        m4.metric("ROI PV", f"{row['ROI_lata']:.1f} lat" if pd.notna(row["ROI_lata"]) else "brak")

        if bess_enabled and not bess_df.empty:
            st.subheader("3. BESS — warianty")
            st.dataframe(bess_df, use_container_width=True)

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Rekomendowany BESS kW", f"{row['Rekomendowany_BESS_kW']:,.0f}" if pd.notna(row["Rekomendowany_BESS_kW"]) else "-")
            b2.metric("Rekomendowany BESS kWh", f"{row['Rekomendowany_BESS_kWh']:,.0f}" if pd.notna(row["Rekomendowany_BESS_kWh"]) else "-")
            b3.metric("SC z BESS", f"{row['SC_z_BESS_%']:.1f}%" if pd.notna(row["SC_z_BESS_%"]) else "-")
            b4.metric("Dodatkowa autokonsumpcja", f"{row['Dodatkowa_autokonsumpcja_BESS_MWh']:,.1f} MWh" if pd.notna(row["Dodatkowa_autokonsumpcja_BESS_MWh"]) else "-")

        st.subheader("4. Krzywa SC od mocy PV")
        chart_profile = st.selectbox("Profil do wykresu", pv_curve_df["Profil"].unique())
        chart = pv_curve_df[pv_curve_df["Profil"] == chart_profile].copy()
        st.line_chart(chart.set_index("pv_kwp")["SC_%"], height=320)

        st.subheader("5. Miesięcznie")
        st.dataframe(monthly_df, use_container_width=True)

        excel = make_excel(summary_df, pv_curve_df, bess_df, monthly_df)
        st.download_button(
            "Pobierz pełny raport XLSX",
            data=excel,
            file_name="energy_agent_pv_bess_raport.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.caption(
            "Uwaga: profil PV jest syntetyczny i służy do screeningu. "
            "BESS jest liczony jako uproszczona symulacja ładowania z nadwyżek PV i rozładowania na deficyt odbioru."
        )

except Exception as e:
    st.error(f"Błąd analizy: {e}")
