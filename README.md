# Energy Agent Online Dashboard — PV + BESS

Dashboard online do analizy profili CSV, autokonsumpcji PV, potencjału BESS, peak shavingu oraz podstawowych wskaźników finansowych.

## Co liczy MVP

- zużycie roczne,
- produkcję PV z profilu syntetycznego,
- autokonsumpcję,
- eksport jako potencjał do BESS,
- BESS bez arbitrażu,
- peak shaving,
- opłatę mocową / korzyść z redukcji mocy,
- ROI,
- IRR,
- DSCR,
- porównanie SaaS/CAPEX,
- ranking wariantów PV+BESS.

## Uruchomienie online przez Streamlit Cloud

1. Wejdź na GitHub i utwórz nowe repozytorium, np. `energy-agent-dashboard`.
2. Wrzuć do repozytorium pliki z tej paczki:
   - `app.py`
   - `requirements.txt`
   - folder `.streamlit`
   - `README.md`
3. Wejdź na Streamlit Cloud.
4. Kliknij `New app`.
5. Wybierz repozytorium z GitHuba.
6. Jako plik startowy ustaw:

```text
app.py
```

7. Kliknij `Deploy`.
8. Po chwili dostaniesz link do dashboardu.

## Ważne uwagi

- Pliki CSV użytkownik wgrywa ręcznie w dashboardzie.
- Nie trzymaj danych klientów w publicznym repozytorium.
- Na start najlepiej używać repozytorium prywatnego.
- MVP używa uproszczonego profilu PV, a nie danych PVGIS. To można dodać w kolejnym kroku.
- Logika BESS nie uwzględnia arbitrażu cenowego — zgodnie z założeniem MVP.
