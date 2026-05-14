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


def decode_csv_bytes(raw: bytes) -> str:
    for enc in ["utf-8-sig", "cp1250", "iso-8859-2", "latin1"]:
        try:
            return raw.decode(enc)
        except Exception:
            pass
    return raw.decode("latin1", errors="ignore")


def load_pvsyst_export(text: str) -> pd.DataFrame:
    """Obsługa eksportów PVsyst/arkuszy z blokiem nagłówkowym przed tabelą godzinową."""
    lines = text.splitlines()
    data_start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}", line):
            data_start = i
            break
    if data_start is None:
        raise ValueError("Nie znaleziono początku danych godzinowych w pliku PVsyst.")

    data_text = "\n".join(lines[data_start:])
    tmp = pd.read_csv(
        io.StringIO(data_text),
        sep=";",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["date", "pv_kwh", "load_kwh", "export_kwh"],
        decimal=",",
    )

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(tmp["date"].astype(str), format="%d.%m.%Y %H:%M", errors="coerce")
    out["load_kwh"] = pd.to_numeric(tmp["load_kwh"].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(0)
    out["source_pv_kwh"] = pd.to_numeric(tmp["pv_kwh"].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(0)
    out["source_export_kwh"] = pd.to_numeric(tmp["export_kwh"].astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(0)
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return aggregate_energy_by_hour(out)


def load_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    text = decode_csv_bytes(raw)

    # Specjalny przypadek: PVsyst z metadanymi u góry i tabelą od wiersza „date;Produkcja PV...”.
    if "Produkcja PV" in text and "Zapotrzebowanie klienta" in text:
        return load_pvsyst_export(text)

    last_error = None
    for enc in ["utf-8-sig", "cp1250", "latin1"]:
        for sep in ["\t", ";", ","]:
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc)
                if df.shape[1] >= 2:
                    prof = normalize_profile(df)
                    if len(prof) > 0:
                        return prof
            except Exception as e:
                last_error = e
    raise ValueError(f"Nie udało się odczytać CSV: {last_error}")


def load_excel(uploaded_file) -> pd.DataFrame:
    """Auto-import XLSX/XLS: czyta arkusz i normalizuje profil.

    Obsługuje proste arkusze typu:
    Data | Wolumen energii elektrycznej pobranej z sieci przed bilansowaniem godzinowym
    oraz podobne eksporty, gdzie jedna kolumna jest datą/czasem, a druga energią.
    """
    raw = uploaded_file.getvalue()
    last_error = None
    try:
        xls = pd.ExcelFile(io.BytesIO(raw))
        # Najpierw szukamy arkusza, który daje sensowny profil.
        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                df = df.dropna(how="all")
                if df.shape[1] >= 2:
                    prof = normalize_profile(df)
                    if len(prof) > 0:
                        return prof
            except Exception as e:
                last_error = e
                continue
    except Exception as e:
        last_error = e
    raise ValueError(f"Nie udało się odczytać Excela: {last_error}")


def load_profile_auto(uploaded_file) -> pd.DataFrame:
    name = (uploaded_file.name or "").lower()
    if name.endswith((".xlsx", ".xls")):
        return load_excel(uploaded_file)
    return load_csv(uploaded_file)



def read_generic_csv(raw: bytes, max_skiprows: int = 40) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Czyta różne CSV do surowej tabeli dla ręcznego mapowania kolumn.
    Testuje kodowanie, separator i pierwsze wiersze metadanych.
    """
    best = None
    best_score = -1
    best_meta = {}
    for enc in ["utf-8-sig", "cp1250", "iso-8859-2", "latin1"]:
        for sep in [";", ",", "\t"]:
            for skip in range(0, max_skiprows + 1):
                try:
                    tmp = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, skiprows=skip, nrows=300)
                    if tmp.shape[1] < 2:
                        continue
                    date_score = 0
                    num_score = 0
                    for c in tmp.columns:
                        sample = tmp[c].astype(str).head(100).str.replace(r"([0-9]{2}:[0-9]{2})[A-Z]$", r"\1", regex=True)
                        date_score = max(date_score, pd.to_datetime(sample, dayfirst=True, errors="coerce").notna().sum())
                        try:
                            nums = sample.map(parse_number_pl)
                            num_score += int((nums != 0).sum())
                        except Exception:
                            pass
                    score = tmp.shape[1] * 10 + date_score * 3 + min(num_score, 300)
                    if score > best_score:
                        best_score = score
                        best_meta = {"typ": "CSV", "encoding": enc, "separator": repr(sep), "skiprows": str(skip)}
                        best = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, skiprows=skip)
                except Exception:
                    continue
    if best is None:
        raise ValueError("Nie udało się odczytać pliku CSV w trybie uniwersalnym.")
    best = best.dropna(how="all")
    best.columns = [str(c).strip() for c in best.columns]
    return best, best_meta


def read_generic_excel(raw: bytes) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Czyta XLSX/XLS do surowej tabeli dla ręcznego mapowania kolumn."""
    try:
        xls = pd.ExcelFile(io.BytesIO(raw))
    except Exception as e:
        raise ValueError(f"Nie udało się otworzyć Excela: {e}")

    best = None
    best_score = -1
    best_meta = {}
    for sheet_name in xls.sheet_names:
        for header in range(0, 20):
            try:
                tmp = pd.read_excel(xls, sheet_name=sheet_name, header=header, nrows=300)
                tmp = tmp.dropna(how="all")
                if tmp.shape[1] < 2:
                    continue
                date_score = 0
                num_score = 0
                for c in tmp.columns:
                    sample = tmp[c].astype(str).head(100).str.replace(r"([0-9]{2}:[0-9]{2})[A-Z]$", r"\1", regex=True)
                    date_score = max(date_score, pd.to_datetime(sample, dayfirst=True, errors="coerce").notna().sum())
                    try:
                        nums = sample.map(parse_number_pl)
                        num_score += int((nums != 0).sum())
                    except Exception:
                        pass
                score = tmp.shape[1] * 10 + date_score * 3 + min(num_score, 300)
                if score > best_score:
                    best_score = score
                    best_meta = {"typ": "Excel", "arkusz": sheet_name, "header_row": str(header + 1)}
                    best = pd.read_excel(xls, sheet_name=sheet_name, header=header)
            except Exception:
                continue
    if best is None:
        raise ValueError("Nie udało się odczytać Excela w trybie uniwersalnym.")
    best = best.dropna(how="all")
    best.columns = [str(c).strip() for c in best.columns]
    return best, best_meta


def read_generic_table(uploaded_file) -> Tuple[pd.DataFrame, Dict[str, str]]:
    raw = uploaded_file.getvalue()
    name = (uploaded_file.name or "").lower()
    if name.endswith((".xlsx", ".xls")):
        return read_generic_excel(raw)
    return read_generic_csv(raw)


def guess_column(cols: List[str], keywords: List[str], excludes: List[str] = None):
    excludes = excludes or []
    for c in cols:
        cl = str(c).lower()
        if any(e in cl for e in excludes):
            continue
        if any(k in cl for k in keywords):
            return c
    return cols[0] if cols else None


def normalize_profile_manual(
    raw_df: pd.DataFrame,
    date_col: str,
    load_col: str,
    time_col: str = "— brak —",
    pv_col: str = "— brak —",
    export_col: str = "— brak —",
    ppe_col: str = "— brak —",
    unit: str = "kWh",
    aggregate_hourly: bool = True,
) -> pd.DataFrame:
    out = pd.DataFrame()
    if time_col and time_col != "— brak —":
        dt_text = raw_df[date_col].astype(str).str.strip() + " " + raw_df[time_col].astype(str).str.strip()
    else:
        dt_text = raw_df[date_col].astype(str).str.strip()
    dt_text = dt_text.str.replace(r"([0-9]{2}:[0-9]{2})[A-Z]$", r"\1", regex=True)
    out["timestamp"] = pd.to_datetime(dt_text, dayfirst=True, errors="coerce")

    multiplier = 1000.0 if unit == "MWh" else 1.0
    out["load_kwh"] = raw_df[load_col].map(parse_number_pl) * multiplier
    if pv_col and pv_col != "— brak —":
        out["source_pv_kwh"] = raw_df[pv_col].map(parse_number_pl) * multiplier
    if export_col and export_col != "— brak —":
        out["source_export_kwh"] = raw_df[export_col].map(parse_number_pl) * multiplier
    if ppe_col and ppe_col != "— brak —":
        out["ppe_id"] = raw_df[ppe_col].astype(str).str.strip()

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    if aggregate_hourly:
        return aggregate_energy_by_hour(out)
    return enrich_time_columns(out)

def enrich_time_columns(out: pd.DataFrame) -> pd.DataFrame:
    out["date"] = out["timestamp"].dt.date
    out["hour"] = out["timestamp"].dt.hour
    out["month"] = out["timestamp"].dt.month
    out["weekday"] = out["timestamp"].dt.dayofweek
    out["is_sunday"] = out["weekday"].eq(6)
    return out


def aggregate_energy_by_hour(out: pd.DataFrame, ppe_col: str = None) -> pd.DataFrame:
    """Finalizuje profil energii.

    Ważne dla multi-PPE:
    - timestampy typu 00:59/01:59 są sprowadzane do początku godziny,
    - jeżeli wykryto PPE/licznik, dane są agregowane OSOBNO dla każdego PPE,
    - aplikacja nie sumuje PPE automatycznie do jednego profilu zakładu.

    Sumowanie wielu PPE powinno być osobnym, świadomym trybem, bo SC/PV/BESS
    dla każdego PPE może być inna i nie zawsze istnieje wspólny bilans energii.
    """
    out = out.copy()
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    # Profile OSD często opisują godzinę końcem interwału, np. 00:59.
    # Do symulacji PV/BESS potrzebujemy wspólnego indeksu godzinowego.
    out["timestamp"] = out["timestamp"].dt.floor("h")

    value_cols = [c for c in out.columns if c not in {"timestamp", "ppe_id"}]
    raw_rows = len(out)
    unique_hours = out["timestamp"].nunique()
    rows_per_hour = out.groupby("timestamp").size() if raw_rows else pd.Series(dtype=int)
    ppe_count = out["ppe_id"].nunique() if "ppe_id" in out.columns else None

    if "ppe_id" in out.columns and ppe_count and ppe_count > 1:
        grouped = out.groupby(["ppe_id", "timestamp"], as_index=False)[value_cols].sum()
        grouped.attrs["multi_ppe_mode"] = "separate"
    else:
        grouped = out.groupby("timestamp", as_index=False)[value_cols].sum()
        grouped.attrs["multi_ppe_mode"] = "single"

    grouped.attrs["raw_rows_before_hourly_aggregation"] = raw_rows
    grouped.attrs["unique_hours_after_aggregation"] = int(unique_hours)
    grouped.attrs["max_records_per_hour_before_aggregation"] = int(rows_per_hour.max()) if len(rows_per_hour) else 0
    grouped.attrs["ppe_count_detected"] = int(ppe_count) if ppe_count is not None else None
    grouped.attrs["multi_ppe_or_duplicate_hours_detected"] = bool((rows_per_hour.max() if len(rows_per_hour) else 0) > 1)
    grouped.attrs["multi_ppe_detected"] = bool(ppe_count and ppe_count > 1)
    return enrich_time_columns(grouped)


def normalize_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizuje różne formaty profili godzinowych do: timestamp, load_kwh (+ opcjonalnie source_pv_kwh/export).

    Obsługiwane m.in.:
    - klasyczne CSV: data/godzina + zużycie,
    - eksporty OSD/sprzedawców: „Data i godzina”, „Wartosc[kWh/kvar]”, „Rodzaj energii”,
    - pliki z produkcją PV i eksportem.
    """
    # Ujednolicenie nazw pomocniczo, ale bez utraty oryginalnych nazw kolumn
    cols = list(df.columns)
    col_l = {c: str(c).strip().lower() for c in cols}

    # 1) Kolumna daty/godziny — nie zakładamy już, że jest pierwsza.
    date_col = None
    date_keywords = ["data i godzina", "data/godzina", "datetime", "timestamp", "czas", "date", "godzina"]
    for c in cols:
        cl = col_l[c]
        if any(k in cl for k in date_keywords):
            date_col = c
            break
    if date_col is None:
        # fallback: wybierz kolumnę, która najlepiej parsuje się jako data
        best_col, best_score = None, -1
        for c in cols:
            sample = df[c].astype(str).head(200).str.replace(r"([0-9]{2}:[0-9]{2})[A-Z]$", r"\1", regex=True)
            parsed = pd.to_datetime(sample, dayfirst=True, errors="coerce")
            score = parsed.notna().sum()
            if score > best_score:
                best_col, best_score = c, score
        date_col = best_col

    # 2) Kolumna zużycia — wykluczamy kolumnę PPE „punkt poboru”, bo zawiera słowo poboru, ale nie jest energią.
    load_col = None
    strong_load_keywords = [
        "wartosc", "wartość", "kwh", "energia czynna pobrana", "zapotrzeb", "zuży", "zuzy", "load", "consumption", "pobrana"
    ]
    exclude_keywords = ["punkt poboru", "ppe", "nr punktu", "platnika", "płatnika", "kompletnosc", "kompletność", "strefa", "rodzaj energii"]
    for c in cols:
        if c == date_col:
            continue
        cl = col_l[c]
        if any(ex in cl for ex in exclude_keywords):
            continue
        if any(k in cl for k in strong_load_keywords):
            load_col = c
            break

    # Szczególny format: OSD/sprzedawca ma „Rodzaj energii” + „Wartosc[kWh/kvar]”.
    rodzaj_col = next((c for c in cols if "rodzaj energii" in col_l[c]), None)
    wartosc_col = next((c for c in cols if "wartosc" in col_l[c] or "wartość" in col_l[c]), None)
    if rodzaj_col is not None and wartosc_col is not None:
        # Jeżeli są różne rodzaje energii, bierzemy pobraną czynną. W tym pliku jest właśnie tylko ona.
        mask = df[rodzaj_col].astype(str).str.lower().str.contains("czynna") & df[rodzaj_col].astype(str).str.lower().str.contains("pobran")
        if mask.any():
            df = df.loc[mask].copy()
        load_col = wartosc_col

    if load_col is None:
        # fallback: pierwsza sensowna numeryczna kolumna poza datą i metadanymi
        best_col, best_numeric = None, -1
        for c in cols:
            if c == date_col:
                continue
            cl = col_l[c]
            if any(ex in cl for ex in exclude_keywords):
                continue
            nums = df[c].map(parse_number_pl)
            score = (nums != 0).sum()
            if score > best_numeric:
                best_col, best_numeric = c, score
        load_col = best_col

    # 3) PV i eksport, jeżeli istnieją.
    pv_col = None
    export_col = None
    for c in df.columns:
        cl = str(c).lower()
        if pv_col is None and ("produkcja pv" in cl or cl.strip() == "pv" or "production" in cl):
            pv_col = c
        if export_col is None and ("wprowadzone" in cl or "eksport" in cl or "export" in cl):
            export_col = c

    out = pd.DataFrame()
    dt_text = df[date_col].astype(str).str.strip().str.replace(r"([0-9]{2}:[0-9]{2})[A-Z]$", r"\1", regex=True)
    out["timestamp"] = pd.to_datetime(dt_text, dayfirst=True, errors="coerce")
    out["load_kwh"] = df[load_col].map(parse_number_pl)
    if pv_col is not None:
        out["source_pv_kwh"] = df[pv_col].map(parse_number_pl)
    if export_col is not None:
        out["source_export_kwh"] = df[export_col].map(parse_number_pl)

    # 4) Opcjonalna kolumna PPE/licznika — tylko diagnostyka.
    ppe_col = None
    ppe_keywords = ["ppe", "punkt poboru", "nr punktu", "kod ppe", "licznik", "meter", "metering"]
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ppe_keywords):
            ppe_col = c
            break
    if ppe_col is not None:
        out["ppe_id"] = df[ppe_col].astype(str)

    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return aggregate_energy_by_hour(out, ppe_col=ppe_col)

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
    initial_soc_pct: float = 50.0
    min_soc_pct: float = 10.0
    mode: str = "Peak shaving priority"
    pv_mode: str = "Auto"


def simulate_pv_bess(
    df: pd.DataFrame,
    pv_kwp: float,
    annual_yield: float,
    bess: BessConfig,
    peak_target_kw: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Godzinowy model BESS bez arbitrażu cenowego.

    Zasady:
    - PV w pierwszej kolejności zasila odbiór klienta.
    - BESS ładuje się wyłącznie z nadwyżki PV, zgodnie z założeniem zero export w WPW.
    - BESS może pracować w dwóch trybach:
      1) Peak shaving priority — bateria oszczędzana jest głównie na godziny przekroczenia celu mocy.
      2) Self-consumption + peak shaving — bateria redukuje pobór z sieci także poza pikami.
    - Uwzględniane są: SoC, min. SoC, moc ładowania/rozładowania, pojemność i sprawność.
    """
    sim = df.copy()
    load = sim["load_kwh"].to_numpy(dtype=float)  # dane godzinowe, kWh ~= średnia moc kW

    has_file_pv = "source_pv_kwh" in sim.columns and sim["source_pv_kwh"].sum() > 0
    has_file_export = "source_export_kwh" in sim.columns and sim["source_export_kwh"].sum() >= 0 and sim.get("source_export_kwh", pd.Series(dtype=float)).sum() > 0
    pv_mode = getattr(bess, "pv_mode", "Auto")

    if pv_mode == "Auto":
        # Dla plików PVsyst / z kolumną produkcji PV domyślnie NIE generujemy PV syntetycznie
        # i NIE skalujemy profilu. To daje SC zgodne z profilem technicznym.
        pv_mode_effective = "Z pliku — bez skalowania (PVsyst)" if has_file_pv else "Syntetyczna PV z uzysku"
    else:
        pv_mode_effective = pv_mode

    if has_file_pv and pv_mode_effective == "Z pliku — bez skalowania (PVsyst)":
        pv = sim["source_pv_kwh"].to_numpy(dtype=float)
    elif has_file_pv and pv_mode_effective == "Z pliku — skaluj do mocy PV":
        # Skalowanie jest pomocne do scenariuszy „co jeśli”, ale może dawać rozjazd z PVsyst.
        base_pv_kwp = sim["source_pv_kwh"].sum() / max(annual_yield, 1)
        pv = sim["source_pv_kwh"].to_numpy(dtype=float) * (pv_kwp / base_pv_kwp if base_pv_kwp > 0 else 1.0)
    else:
        pv = synthetic_pv_profile(sim, pv_kwp, annual_yield)

    dt_h = 1.0
    eta = np.sqrt(max(min(bess.roundtrip_eff, 1.0), 0.01))

    # Tryb PVsyst: jeżeli plik ma kolumnę eksportu/wprowadzenia do sieci, traktujemy ją jako
    # technicznie policzoną nadwyżkę. Wtedy SC przed BESS = (PV - eksport z PVsyst) / PV.
    if has_file_pv and has_file_export and pv_mode_effective == "Z pliku — bez skalowania (PVsyst)":
        file_export = np.minimum(sim["source_export_kwh"].to_numpy(dtype=float), pv)
        direct = np.maximum(pv - file_export, 0)
        pv_surplus = file_export.copy()
        residual_load = np.maximum(load - direct, 0)
    else:
        direct = np.minimum(load, pv)
        pv_surplus = np.maximum(pv - load, 0)
        residual_load = np.maximum(load - pv, 0)

    sc_before_bess = direct.sum() / pv.sum() if pv.sum() > 0 else 0

    soc = bess.capacity_kwh * bess.initial_soc_pct / 100 if bess.capacity_kwh > 0 else 0
    min_soc = bess.capacity_kwh * bess.min_soc_pct / 100 if bess.capacity_kwh > 0 else 0
    max_soc = bess.capacity_kwh

    charge = np.zeros(len(sim))
    discharge_self = np.zeros(len(sim))
    discharge_peak = np.zeros(len(sim))
    export_after_bess = np.zeros(len(sim))
    soc_series = np.zeros(len(sim))
    grid_before_bess = residual_load.copy()
    grid_after = np.zeros(len(sim))
    clipped_peak = np.zeros(len(sim))

    for i in range(len(sim)):
        # 1) Ładowanie wyłącznie z nadwyżki PV — brak arbitrażu i brak ładowania z sieci.
        if bess.capacity_kwh > 0 and bess.power_kw > 0 and pv_surplus[i] > 0:
            max_charge_from_power = bess.power_kw * dt_h
            max_charge_from_space = max(0, (max_soc - soc) / eta)
            ch = min(pv_surplus[i], max_charge_from_power, max_charge_from_space)
            charge[i] = ch
            soc += ch * eta

        grid = residual_load[i]

        # 2) Rozładowanie — tryb peak shaving priority lub pełna autokonsumpcja.
        if bess.capacity_kwh > 0 and bess.power_kw > 0 and grid > 0:
            available = max(0, (soc - min_soc) * eta)
            power_left = bess.power_kw * dt_h

            if bess.mode == "Self-consumption + peak shaving":
                # Najpierw redukujemy każdy pobór z sieci, potem pilnujemy peaku.
                dis = min(grid, available, power_left)
                discharge_self[i] = dis
                soc -= dis / eta
                grid -= dis
                power_left -= dis
                available = max(0, (soc - min_soc) * eta)

                if grid > peak_target_kw and power_left > 0 and available > 0:
                    peak_need = grid - peak_target_kw
                    dis2 = min(peak_need, available, power_left)
                    discharge_peak[i] = dis2
                    soc -= dis2 / eta
                    grid -= dis2
            else:
                # Peak shaving priority: nie rozładowujemy baterii poza pikami, żeby nie wyczyścić SoC przed szczytem.
                if grid > peak_target_kw:
                    peak_need = grid - peak_target_kw
                    dis = min(peak_need, available, power_left)
                    discharge_peak[i] = dis
                    soc -= dis / eta
                    grid -= dis

        export_after_bess[i] = max(0, pv_surplus[i] - charge[i])
        soc_series[i] = soc
        grid_after[i] = grid
        clipped_peak[i] = max(0, grid_before_bess[i] - grid_after[i])

    sim["pv_kwh"] = pv
    sim["pv_mode"] = pv_mode_effective
    sim["sc_before_bess_pct_hourly"] = sc_before_bess * 100
    sim["direct_self_kwh"] = direct
    sim["pv_surplus_kwh"] = pv_surplus
    sim["grid_before_bess_kwh"] = grid_before_bess
    sim["bess_charge_kwh"] = charge
    sim["bess_discharge_self_kwh"] = discharge_self
    sim["bess_discharge_peak_kwh"] = discharge_peak
    sim["bess_discharge_total_kwh"] = discharge_self + discharge_peak
    sim["export_potential_kwh"] = pv_surplus
    sim["export_after_bess_kwh"] = export_after_bess
    sim["grid_after_kwh"] = grid_after
    sim["peak_reduction_hourly_kwh"] = clipped_peak
    sim["soc_kwh"] = soc_series
    sim["soc_pct"] = np.where(bess.capacity_kwh > 0, soc_series / bess.capacity_kwh * 100, 0)

    pv_total = pv.sum()
    export_potential = pv_surplus.sum()
    export_after = export_after_bess.sum()
    pv_used_direct = direct.sum()
    pv_used_via_bess = charge.sum() - export_after_bess.sum() * 0  # informacyjnie: energia skierowana do BESS z PV
    used_pv = pv_total - export_after
    autokonsumpcja = used_pv / pv_total if pv_total else 0
    peak_before = load.max()
    peak_after_pv = residual_load.max()
    peak_after = grid_after.max()
    peak_reduction_kw = max(0, peak_before - peak_after)
    bess_discharge_total = (discharge_self.sum() + discharge_peak.sum())
    equivalent_cycles = bess_discharge_total / bess.capacity_kwh if bess.capacity_kwh > 0 else 0
    export_reduction = export_potential - export_after

    metrics = {
        "load_mwh": load.sum() / 1000,
        "pv_mwh": pv_total / 1000,
        "direct_self_mwh": pv_used_direct / 1000,
        "autokonsumpcja_przed_bess_pct": sc_before_bess * 100,
        "autokonsumpcja_pct": autokonsumpcja * 100,
        "pv_mode_effective": pv_mode_effective,
        "export_potential_mwh": export_potential / 1000,
        "export_after_bess_mwh": export_after / 1000,
        "export_reduction_mwh": export_reduction / 1000,
        "bess_charge_mwh": charge.sum() / 1000,
        "bess_discharge_mwh": bess_discharge_total / 1000,
        "bess_discharge_peak_mwh": discharge_peak.sum() / 1000,
        "bess_discharge_self_mwh": discharge_self.sum() / 1000,
        "bess_equivalent_cycles": equivalent_cycles,
        "peak_before_kw": peak_before,
        "peak_after_pv_kw": peak_after_pv,
        "peak_after_kw": peak_after,
        "peak_reduction_kw": peak_reduction_kw,
        "grid_after_mwh": grid_after.sum() / 1000,
        "min_soc_pct": sim["soc_pct"].min() if bess.capacity_kwh > 0 else 0,
        "avg_soc_pct": sim["soc_pct"].mean() if bess.capacity_kwh > 0 else 0,
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
            bess = BessConfig(
                capacity_kwh=cap,
                power_kw=power,
                roundtrip_eff=params["bess_eff_pct"] / 100,
                initial_soc_pct=params.get("initial_soc_pct", 50),
                min_soc_pct=params.get("min_soc_pct", 10),
                mode=params.get("bess_mode", "Peak shaving priority"),
                pv_mode=params.get("pv_mode", "Auto"),
            )
            _, m = simulate_pv_bess(df, pv, annual_yield, bess, peak_target)
            f = financials(m, pv, bess, params)
            rows.append({"PV kWp": pv, "BESS kWh": cap, "BESS kW": power, **m, **f})
    res = pd.DataFrame(rows)
    res["spełnia 80% SC"] = res["autokonsumpcja_pct"] >= min_sc
    return res.sort_values(["spełnia 80% SC", "net_cash_pln"], ascending=[False, False])






def split_profiles_by_ppe(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Zwraca słownik profili PPE -> dataframe.

    Jeśli profil nie ma kolumny ppe_id albo jest tylko jeden PPE, zwraca pusty słownik.
    Dzięki temu główny dashboard działa jak dotychczas dla pojedynczego profilu,
    a tryb multi-PPE uruchamia się wyłącznie dla wielu punktów poboru.
    """
    if df is None or df.empty or "ppe_id" not in df.columns:
        return {}

    tmp = df.copy()
    tmp["ppe_id"] = tmp["ppe_id"].astype(str).str.strip()
    tmp = tmp[tmp["ppe_id"].notna() & (tmp["ppe_id"] != "") & (tmp["ppe_id"].str.lower() != "nan")]

    ppes = sorted(tmp["ppe_id"].unique().tolist())
    if len(ppes) <= 1:
        return {}

    profiles = {}
    for ppe in ppes:
        prof = tmp[tmp["ppe_id"] == ppe].copy().sort_values("timestamp").reset_index(drop=True)
        # Zachowujemy ppe_id dla diagnostyki, ale symulacje korzystają z load_kwh/PV/export/timestamp.
        profiles[ppe] = prof
    return profiles

def _max_pv_for_sc(prof: pd.DataFrame, annual_yield: float, bess: BessConfig, peak_target: float, min_sc_pct: float, pv_max: int, metric_key: str):
    """Zwraca największą moc PV, dla której wskazany wskaźnik SC >= cel.

    metric_key:
    - autokonsumpcja_przed_bess_pct = SC klasyczne/PVsyst bez magazynu
    - autokonsumpcja_pct = SC po BESS
    """
    best_pv = 0
    best_m = None
    for pv in range(1, pv_max + 1):
        _, m = simulate_pv_bess(prof, pv, annual_yield, bess, peak_target)
        if m.get(metric_key, 0) >= min_sc_pct:
            best_pv = pv
            best_m = m
        else:
            # Dla stałego profilu i rosnącej mocy PV SC zasadniczo maleje.
            # Przerywamy dopiero po znalezieniu wariantu spełniającego, żeby nie zwrócić sztucznie 0.
            if best_pv > 0:
                break
    return best_pv, best_m


def find_pv_size_for_min_sc(
    profiles: Dict[str, pd.DataFrame],
    annual_yield: float,
    bess: BessConfig,
    peak_target: float,
    params: Dict[str, float],
    min_sc_pct: float = 80.0,
) -> pd.DataFrame:
    """Dla każdego PPE wyznacza indywidualną moc PV spełniającą SC >= cel.

    W v15 pokazujemy dwie wartości, bo to było główne źródło rozjazdów:
    1) SC bez BESS / jak w PVsyst: energia PV zużyta bezpośrednio / produkcja PV.
    2) SC po BESS: produkcja PV niewyeksportowana po pracy magazynu / produkcja PV.

    Do porównania z działem technicznym i PVsyst zwykle patrzymy na kolumnę
    „PV max dla SC bez BESS >=80%”.
    """
    rows = []
    pv_max = int(max(1, params.get("pv_max_kwp", 3000)))

    for ppe, prof in profiles.items():
        pv_before, m_before = _max_pv_for_sc(
            prof, annual_yield, bess, peak_target, min_sc_pct, pv_max, "autokonsumpcja_przed_bess_pct"
        )
        pv_after, m_after = _max_pv_for_sc(
            prof, annual_yield, bess, peak_target, min_sc_pct, pv_max, "autokonsumpcja_pct"
        )

        if m_before is None:
            _, m1 = simulate_pv_bess(prof, 1, annual_yield, bess, peak_target)
            m_before = m1
        if m_after is None:
            _, m1 = simulate_pv_bess(prof, 1, annual_yield, bess, peak_target)
            m_after = m1

        load_mwh = prof["load_kwh"].sum() / 1000 if "load_kwh" in prof.columns else np.nan
        rows.append({
            "PPE": ppe,
            "Zużycie PPE [MWh/rok]": load_mwh,
            "PV max dla SC bez BESS >=80% [kWp]": float(pv_before),
            "SC bez BESS przy tej PV [%]": m_before.get("autokonsumpcja_przed_bess_pct", np.nan),
            "Eksport bez BESS [MWh/rok]": m_before.get("export_potential_mwh", np.nan),
            "PV max dla SC po BESS >=80% [kWp]": float(pv_after),
            "SC po BESS przy tej PV [%]": m_after.get("autokonsumpcja_pct", np.nan),
            "Eksport po BESS [MWh/rok]": m_after.get("export_after_bess_mwh", np.nan),
            "Status": "OK" if pv_before > 0 else "Brak wariantu w zakresie — nawet 1 kWp nie spełnia celu",
        })
    return pd.DataFrame(rows).sort_values("PV max dla SC bez BESS >=80% [kWp]", ascending=False)


def build_ppe_summary(profiles: Dict[str, pd.DataFrame], pv_kwp: float, annual_yield: float, bess: BessConfig, peak_target: float, params: Dict[str, float]) -> pd.DataFrame:
    """Liczy szybkie KPI dla każdego PPE tymi samymi założeniami technicznymi i finansowymi."""
    rows = []
    for ppe, prof in profiles.items():
        sim_i, m = simulate_pv_bess(prof, pv_kwp, annual_yield, bess, peak_target)
        f = financials(m, pv_kwp, bess, params)
        rows.append({
            "PPE": ppe,
            "Godziny": len(prof),
            "Zużycie [MWh]": m["load_mwh"],
            "PV [MWh]": m["pv_mwh"],
            "SC przed BESS [%]": m["autokonsumpcja_przed_bess_pct"],
            "SC po BESS [%]": m["autokonsumpcja_pct"],
            "Eksport/potencjał BESS [MWh]": m["export_potential_mwh"],
            "Eksport po BESS [MWh]": m["export_after_bess_mwh"],
            "Peak przed [kW]": m["peak_before_kw"],
            "Peak po [kW]": m["peak_after_kw"],
            "Redukcja peak [kW]": m["peak_reduction_kw"],
            "Korzyść roczna [PLN]": f["annual_benefit_pln"],
            "ROI [%]": f["roi_pct"],
            "IRR [%]": f["irr_pct"],
            "DSCR": f["dscr"],
            "CAPEX [PLN]": f["capex_pln"],
        })
    return pd.DataFrame(rows).sort_values("Zużycie [MWh]", ascending=False)

st.title("Energy Agent MVP v16 — dobór PV + BESS")
st.caption("MVP: autokonsumpcja, eksport jako potencjał BESS, godzinowy model SoC magazynu, peak shaving, ROI, IRR, DSCR, SaaS/CAPEX, opłata mocowa. Bez arbitrażu cenowego.")

with st.sidebar:
    st.header("Dane wejściowe")
    uploaded = st.file_uploader("Wgraj profil dobowo-godzinowy", type=["csv", "txt", "xlsx", "xls"])
    st.subheader("Założenia techniczne")
    pv_kwp = st.number_input("Moc PV [kWp]", 10, 10000, 1200, 10)
    annual_yield = st.number_input("Uzysk PV [kWh/kWp/rok]", 700, 1300, 1050, 10)
    pv_mode = st.selectbox(
        "Sposób liczenia profilu PV / SC",
        ["Auto", "Z pliku — bez skalowania (PVsyst)", "Z pliku — skaluj do mocy PV", "Syntetyczna PV z uzysku"],
        index=0,
        help="Dla porównania z działem technicznym wybierz/pozostaw tryb PVsyst: produkcja PV i eksport są brane godzinowo z pliku, bez syntetycznego profilu."
    )
    bess_capacity = st.number_input("Pojemność BESS [kWh]", 0, 10000, 500, 50)
    bess_power = st.number_input("Moc BESS [kW]", 0, 10000, 500, 50)
    bess_eff = st.slider("Sprawność round-trip BESS [%]", 70, 98, 90)
    bess_mode = st.selectbox(
        "Tryb pracy BESS",
        ["Peak shaving priority", "Self-consumption + peak shaving"],
        help="Peak shaving priority oszczędza baterię na piki. Drugi tryb mocniej podnosi autokonsumpcję, ale może zużyć SoC przed szczytem."
    )
    initial_soc = st.slider("Startowy SoC BESS [%]", 0, 100, 50)
    min_soc = st.slider("Minimalny SoC BESS [%]", 0, 50, 10)
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
    st.info("Wgraj CSV/XLSX, żeby uruchomić analizę.")
    st.stop()

raw_bytes = uploaded.getvalue()

st.subheader("Importer profilu")
import_mode = st.radio(
    "Tryb importu",
    ["Auto — rozpoznaj format", "Manual — wskaż kolumny"],
    horizontal=True,
    help="Auto obsługuje znane formaty CSV/XLSX. Manual pozwala wskazać kolumny daty oraz zużycia."
)

try:
    if import_mode.startswith("Auto"):
        df = load_profile_auto(uploaded)
        unique_profiles = df["ppe_id"].nunique() if "ppe_id" in df.columns else 1
        st.success(f"Plik odczytany automatycznie: {len(df):,} rekordów po agregacji godzinowej, {df['load_kwh'].sum()/1000:,.2f} MWh zużycia.".replace(",", " "))
        if df.attrs.get("multi_ppe_detected"):
            st.info(
                (
                    f"Wykryto kilka PPE/liczników: {unique_profiles}. "
                    f"Aplikacja NIE sumuje ich automatycznie — poniżej pokaże wynik dla każdego PPE osobno. "
                    f"Rekordy przed agregacją: {df.attrs.get('raw_rows_before_hourly_aggregation')} | "
                    f"rekordy po agregacji PPE×godzina: {len(df)} | "
                    f"unikalne godziny w pliku: {df.attrs.get('unique_hours_after_aggregation')}"
                )
            )
        elif df.attrs.get("multi_ppe_or_duplicate_hours_detected"):
            st.info(
                (
                    f"Wykryto duplikaty/rekordy częściowe dla tych samych godzin. "
                    f"Zostały zsumowane w ramach tego samego profilu. "
                    f"Rekordy przed agregacją: {df.attrs.get('raw_rows_before_hourly_aggregation')} | "
                    f"godziny po agregacji: {df.attrs.get('unique_hours_after_aggregation')} | "
                    f"maks. rekordów w jednej godzinie: {df.attrs.get('max_records_per_hour_before_aggregation')}"
                )
            )
        if "source_pv_kwh" in df.columns:
            msg = f"Wykryto produkcję PV w pliku: {df['source_pv_kwh'].sum()/1000:,.2f} MWh".replace(",", " ")
            if "source_export_kwh" in df.columns:
                exp = df['source_export_kwh'].sum()/1000
                pvsum = df['source_pv_kwh'].sum()
                sc = (1 - df['source_export_kwh'].sum()/pvsum) * 100 if pvsum > 0 else 0
                msg += f" | eksport z pliku: {exp:,.2f} MWh | SC wg pliku/PVsyst: {sc:.1f}%".replace(",", " ")
            st.info(msg)
        with st.expander("Podgląd znormalizowanych danych"):
            st.dataframe(df.head(50), use_container_width=True)
    else:
        raw_df, meta = read_generic_table(uploaded)
        cols_raw = list(raw_df.columns)
        st.caption("Wykryto: " + ", ".join([f"{k}: {v}" for k, v in meta.items()]))
        with st.expander("Podgląd surowego pliku", expanded=True):
            st.dataframe(raw_df.head(30), use_container_width=True)

        date_guess = guess_column(cols_raw, ["data i godzina", "data/godzina", "datetime", "timestamp", "czas", "date", "data"])
        load_guess = guess_column(
            cols_raw,
            ["wartosc", "wartość", "kwh", "energia czynna pobrana", "zapotrzeb", "zuży", "zuzy", "load", "consumption", "pobrana"],
            ["punkt poboru", "ppe", "nr punktu", "rodzaj energii"]
        )
        pv_guess = guess_column(cols_raw, ["produkcja pv", "pv", "production", "generacja"], [])
        export_guess = guess_column(cols_raw, ["wprowadzone", "eksport", "export", "oddana"], [])

        none_option = "— brak —"
        c1, c2, c3 = st.columns(3)
        with c1:
            date_col = st.selectbox("Kolumna daty / daty i godziny", cols_raw, index=cols_raw.index(date_guess) if date_guess in cols_raw else 0)
            time_col = st.selectbox("Opcjonalna osobna kolumna godziny", [none_option] + cols_raw, index=0)
        with c2:
            load_col = st.selectbox("Kolumna zużycia / zapotrzebowania", cols_raw, index=cols_raw.index(load_guess) if load_guess in cols_raw else 0)
            unit = st.selectbox("Jednostka energii w pliku", ["kWh", "MWh"], index=0)
        with c3:
            pv_options = [none_option] + cols_raw
            export_options = [none_option] + cols_raw
            pv_col = st.selectbox("Opcjonalna kolumna produkcji PV", pv_options, index=pv_options.index(pv_guess) if pv_guess in pv_options else 0)
            export_col = st.selectbox("Opcjonalna kolumna eksportu", export_options, index=export_options.index(export_guess) if export_guess in export_options else 0)
            ppe_guess = guess_column(cols_raw, ["ppe", "punkt poboru", "nr punktu", "kod ppe", "licznik", "meter"], [])
            ppe_col = st.selectbox("Opcjonalna kolumna PPE/licznika", [none_option] + cols_raw, index=([none_option] + cols_raw).index(ppe_guess) if ppe_guess in cols_raw else 0)

        aggregate_hourly = st.checkbox("Agreguj dane do godzin", value=True, help="Włączone: 15-minutówki zostaną zsumowane do danych godzinowych.")
        df = normalize_profile_manual(raw_df, date_col, load_col, time_col, pv_col, export_col, ppe_col, unit, aggregate_hourly)
        st.success(f"Profil zmapowany: {len(df):,} rekordów, {df['load_kwh'].sum()/1000:,.2f} MWh zużycia.".replace(",", " "))
        with st.expander("Podgląd znormalizowanych danych"):
            st.dataframe(df.head(50), use_container_width=True)
except Exception as e:
    st.error(f"Nie udało się zaimportować profilu: {e}")
    st.info("Przełącz tryb importu na Manual i wskaż kolumny: data/godzina oraz zużycie.")
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
    "bess_mode": bess_mode,
    "initial_soc_pct": initial_soc,
    "min_soc_pct": min_soc,
    "peak_target_kw": peak_target,
    "pv_mode": pv_mode,
}

bess = BessConfig(bess_capacity, bess_power, bess_eff / 100, initial_soc, min_soc, bess_mode, pv_mode)

# Multi-PPE: pokazujemy wynik dla każdego PPE osobno i dopiero potem pozwalamy wybrać jeden PPE do szczegółowych wykresów.
ppe_profiles = split_profiles_by_ppe(df)
if ppe_profiles:
    st.subheader("Multi-PPE — wyniki osobno dla każdego punktu poboru")
    st.caption("Aplikacja nie sumuje PPE automatycznie. Każdy PPE liczony jest oddzielnie tymi samymi założeniami PV/BESS/finansowymi.")
    ppe_summary = build_ppe_summary(ppe_profiles, pv_kwp, annual_yield, bess, peak_target, params)
    st.dataframe(
        ppe_summary.style.format({
            "Zużycie [MWh]": "{:.2f}",
            "PV [MWh]": "{:.2f}",
            "SC przed BESS [%]": "{:.1f}",
            "SC po BESS [%]": "{:.1f}",
            "Eksport/potencjał BESS [MWh]": "{:.2f}",
            "Eksport po BESS [MWh]": "{:.2f}",
            "Peak przed [kW]": "{:.0f}",
            "Peak po [kW]": "{:.0f}",
            "Redukcja peak [kW]": "{:.0f}",
            "Korzyść roczna [PLN]": "{:,.0f}",
            "ROI [%]": "{:.1f}",
            "IRR [%]": "{:.1f}",
            "DSCR": "{:.2f}",
            "CAPEX [PLN]": "{:,.0f}",
        }),
        use_container_width=True
    )
    csv_ppe = ppe_summary.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Pobierz wyniki PPE do CSV", csv_ppe, "wyniki_multi_ppe.csv", "text/csv")

    st.subheader("Dobór PV per PPE — cel SC ≥ 80%")
    st.caption(
        "Tabela pokazuje dwie wartości: moc PV dla SC bez BESS/PVsyst ≥ 80% oraz osobno moc PV dla SC po BESS ≥ 80%. Do porównania z działem technicznym używaj przede wszystkim kolumny „SC bez BESS”. "
        "Dolna granica nie bierze pv_min z Optimizera — algorytm szuka progu od 1 kWp do PV max z dokładnością 1 kWp. BESS liczony jest według aktualnych ustawień."
    )
    pv_sc_table = find_pv_size_for_min_sc(ppe_profiles, annual_yield, bess, peak_target, params, min_sc_pct=80.0)
    st.dataframe(
        pv_sc_table.style.format({
            "Zużycie PPE [MWh/rok]": "{:.1f}",
            "PV max dla SC bez BESS >=80% [kWp]": "{:.0f}",
            "SC bez BESS przy tej PV [%]": "{:.1f}",
            "Eksport bez BESS [MWh/rok]": "{:.1f}",
            "PV max dla SC po BESS >=80% [kWp]": "{:.0f}",
            "SC po BESS przy tej PV [%]": "{:.1f}",
            "Eksport po BESS [MWh/rok]": "{:.1f}",
            "SC przed BESS [%]": "{:.1f}",
            "SC po BESS [%]": "{:.1f}",
            "PV [MWh/rok]": "{:.2f}",
            "Eksport po BESS [MWh/rok]": "{:.2f}",
            "Peak po [kW]": "{:.0f}",
        }),
        use_container_width=True
    )
    csv_pv_sc = pv_sc_table.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Pobierz dobór PV dla SC ≥ 80% do CSV", csv_pv_sc, "dobor_pv_sc_80_multi_ppe.csv", "text/csv")

    # Wykres odporny na brak wariantu/kolumn — Plotly wymaga, aby wszystkie hover_data istniały w tabeli.
    chart_y_col = "PV max dla SC bez BESS >=80% [kWp]"
    if chart_y_col in pv_sc_table.columns and "PPE" in pv_sc_table.columns:
        hover_cols = [
            c for c in [
                "Zużycie PPE [MWh/rok]",
                "SC bez BESS przy tej PV [%]",
                "Eksport bez BESS [MWh/rok]",
                "PV max dla SC po BESS >=80% [kWp]",
                "SC po BESS przy tej PV [%]",
                "Eksport po BESS [MWh/rok]",
            ]
            if c in pv_sc_table.columns
        ]
        fig_pv_sc = px.bar(
            pv_sc_table,
            x="PPE",
            y=chart_y_col,
            color="Status" if "Status" in pv_sc_table.columns else None,
            hover_data=hover_cols,
            title="Indywidualna moc PV przy SC bez BESS ≥ 80% — osobno dla każdego PPE"
        )
        st.plotly_chart(fig_pv_sc, use_container_width=True)
    else:
        st.warning("Nie udało się narysować wykresu doboru PV, ale tabela powyżej pozostaje źródłem wyniku.")

    fig_ppe = px.bar(
        ppe_summary,
        x="PPE",
        y="SC po BESS [%]",
        hover_data=["Zużycie [MWh]", "Eksport po BESS [MWh]", "Redukcja peak [kW]", "ROI [%]"],
        title="Autokonsumpcja po BESS — porównanie PPE"
    )
    fig_ppe.add_hline(y=80, line_dash="dash", annotation_text="Cel 80% SC")
    st.plotly_chart(fig_ppe, use_container_width=True)

    selected_ppe = st.selectbox("Wybierz PPE do szczegółowych wykresów poniżej", list(ppe_profiles.keys()))
    df = ppe_profiles[selected_ppe]
    st.info(f"Szczegółowy dashboard poniżej pokazuje wyłącznie PPE: {selected_ppe}")

sim, metrics = simulate_pv_bess(df, pv_kwp, annual_yield, bess, peak_target)
fin = financials(metrics, pv_kwp, bess, params)

kpi = {**metrics, **fin}

st.subheader("Podsumowanie wariantu")
st.caption(f"Tryb PV użyty w obliczeniach: {kpi.get('pv_mode_effective', '—')}")
cols = st.columns(7)
cols[0].metric("Zużycie", f"{kpi['load_mwh']:,.0f} MWh".replace(",", " "))
cols[1].metric("Produkcja PV", f"{kpi['pv_mwh']:,.0f} MWh".replace(",", " "))
cols[2].metric("SC przed BESS", f"{kpi['autokonsumpcja_przed_bess_pct']:.1f}%")
cols[3].metric("SC po BESS", f"{kpi['autokonsumpcja_pct']:.1f}%", "cel min. 80%")
cols[4].metric("Eksport/potencjał BESS", f"{kpi['export_potential_mwh']:,.0f} MWh".replace(",", " "))
cols[5].metric("Redukcja peak", f"{kpi['peak_reduction_kw']:.0f} kW")
cols[6].metric("Korzyść roczna", f"{kpi['annual_benefit_pln']:,.0f} PLN".replace(",", " "))

cols2 = st.columns(5)
cols2[0].metric("CAPEX", f"{kpi['capex_pln']:,.0f} PLN".replace(",", " "))
cols2[1].metric("ROI", f"{kpi['roi_pct']:.1f}%")
cols2[2].metric("IRR", "—" if pd.isna(kpi['irr_pct']) else f"{kpi['irr_pct']:.1f}%")
cols2[3].metric("DSCR", "—" if pd.isna(kpi['dscr']) else f"{kpi['dscr']:.2f}")
cols2[4].metric("SaaS net/rok", f"{kpi['saas_annual_net_pln']:,.0f} PLN".replace(",", " "))

cols3 = st.columns(5)
cols3[0].metric("Eksport po BESS", f"{kpi['export_after_bess_mwh']:,.0f} MWh".replace(",", " "))
cols3[1].metric("Eksport zredukowany", f"{kpi['export_reduction_mwh']:,.0f} MWh".replace(",", " "))
cols3[2].metric("Praca BESS", f"{kpi['bess_discharge_mwh']:,.0f} MWh".replace(",", " "))
cols3[3].metric("Cykle ekwiwalentne", f"{kpi['bess_equivalent_cycles']:.1f} / rok")
cols3[4].metric("Śr. SoC", f"{kpi['avg_soc_pct']:.0f}%")

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

# 5) Bilans pracy BESS — czy bateria pracuje bardziej pod piki, czy pod autokonsumpcję
st.subheader("Bilans pracy BESS")
bess_balance = pd.DataFrame({
    "Kategoria": ["Ładowanie z nadwyżki PV", "Rozładowanie na autokonsumpcję", "Rozładowanie na peak shaving", "Eksport po BESS"],
    "MWh": [
        kpi["bess_charge_mwh"],
        kpi["bess_discharge_self_mwh"],
        kpi["bess_discharge_peak_mwh"],
        kpi["export_after_bess_mwh"],
    ],
})
fig_bess_balance = px.bar(bess_balance, x="Kategoria", y="MWh", text="MWh", title="Energia przepływająca przez BESS i eksport po magazynie")
fig_bess_balance.update_traces(texttemplate="%{text:.0f} MWh", textposition="outside")
st.plotly_chart(fig_bess_balance, use_container_width=True)

# 6) Krzywa czasu trwania poboru — dobrze pokazuje efekt peak shavingu
st.subheader("Krzywa czasu trwania poboru — przed i po BESS")
duration_df = pd.DataFrame({
    "Godzina rankingu": np.arange(1, len(sim) + 1),
    "Przed BESS [kW]": np.sort(sim["grid_before_bess_kwh"].to_numpy())[::-1],
    "Po BESS [kW]": np.sort(sim["grid_after_kwh"].to_numpy())[::-1],
})
fig_duration = go.Figure()
fig_duration.add_scatter(x=duration_df["Godzina rankingu"], y=duration_df["Przed BESS [kW]"], name="Po PV, przed BESS")
fig_duration.add_scatter(x=duration_df["Godzina rankingu"], y=duration_df["Po BESS [kW]"], name="Po BESS")
fig_duration.add_hline(y=peak_target, line_dash="dash", annotation_text="Cel peak shaving")
fig_duration.update_layout(xaxis_title="Godziny posortowane od najwyższego poboru", yaxis_title="kW")
st.plotly_chart(fig_duration, use_container_width=True)

# 7) Próbka godzinowa — pierwsze 14 dni z baterią i poborem po optymalizacji
st.subheader("Przebieg godzinowy — próbka pierwszych 14 dni")
sample = sim.iloc[: min(24*14, len(sim))]
fig2 = go.Figure()
fig2.add_scatter(x=sample["timestamp"], y=sample["load_kwh"], name="Zużycie kWh")
fig2.add_scatter(x=sample["timestamp"], y=sample["pv_kwh"], name="PV kWh")
fig2.add_scatter(x=sample["timestamp"], y=sample["grid_after_kwh"], name="Pobór z sieci po PV+BESS")
fig2.add_scatter(x=sample["timestamp"], y=sample["bess_charge_kwh"], name="Ładowanie BESS")
fig2.add_scatter(x=sample["timestamp"], y=sample["bess_discharge_total_kwh"], name="Rozładowanie BESS")
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
