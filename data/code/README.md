# Data Code

Denne mappe indeholder alle scripts med datalogik:
- patchberegning
- billeddownload
- datarensning/sortering
- segmenterings-datakontrakt og split
- finansmarkeds-downloads

## Segmentering

```bash
python -m data.code.dataset_contract --images <images_dir> --masks <masks_dir> --ext .png
python -m data.code.prepare_splits --pairs <pair_report.csv> --out <split_dir> --strategy hybrid --seed 42
python -m data.code.build_person_mixture --person-root <.../Anders_indi> --person-root <.../Bjarke_indi> --person-root <.../Kasper_indi> --out-root <mixed_out> --fraction 0.5 --seed 42
```

## Container Activity Index

Byg et land-maaneligt containeraktivitetsindeks fra `port_timeseries.csv`:

```bash
python -m data.code.build_container_activity_index \
  --port-timeseries-csv <run_dir>/port_timeseries.csv \
  --port-country-mapping-csv <port_country_mapping.csv> \
  --out-dir <output_dir>
```

Mapping-filen skal minimum have kolonnerne:

```text
port_id,country
```

Scriptet producerer:
- `port_month_activity.csv`
- `country_month_activity.csv`
- `container_activity_index_summary.json`

Anbefalet signal til finansdelen:
- `country_signal`

Her er signalet defineret som et gennemsnit af havnenes maaned-til-maaned log-aendringer:

```text
country_signal = mean_p[ log(CA_{p,t}+1) - log(CA_{p,t-1}+1) ]
```

## Markedsindekser fra Yahoo Finance

```bash
python -m data.code.fetch_market_indices --start 2017-01-01
```

Scriptet gemmer output i `data/outputs/market_indices/yahoo_finance/` som:
- `all_market_indices.csv`
- en CSV pr. land
- `download_summary.json`

Hvis Yahoo Finance mangler et symbol for et land, kan det overrides:

```bash
python -m data.code.fetch_market_indices --symbol-override Panama=DIT_SYMBOL
```

Hvis Yahoo Finance ikke virker for et land, kan scriptet automatisk laese en lokal fallback-CSV fra `data/inputs/market_indices/`.
Det er saerligt taenkt til Panama. Sri Lanka bruger som standard `JKH-N0000.CM` (John Keells Holdings) som markeds-proxy.

## Simpel Forecast-Model

Byg den simple forecast-model oven på `country_month_activity.csv` og et samlet markedsindeks-panel:

```bash
python -m data.code.build_simple_forecast_model \
  --country-month-csv <output_dir>/country_month_activity.csv \
  --market-indices-csv data/outputs/market_indices/yahoo_finance/all_market_indices.csv \
  --out-dir <output_dir> \
  --signal-column signal_change \
  --return-target raw \
  --min-country-months 12 \
  --month-fixed-effects
```

Scriptet producerer:
- `country_month_activity_baseline.csv`
- `country_month_activity_finance_ready.csv`
- `monthly_market_returns.csv`
- `monthly_market_returns_tplus1.csv`
- `forecast_dataset_baseline.csv`
- `simple_forecast_model_summary.txt`
- `simple_forecast_model_coefficients.csv`
- `signal_tercile_country_ranks.csv`
- `signal_tercile_monthly_returns.csv`
- `signal_tercile_test_summary.txt`

Den anbefalede ændringsmodel er:

```text
return_t_plus_1 ~ signal_change + C(country) + C(month)
```

med HC3-robuste standardfejl, hvor `signal_change = country_signal_zscore_t - country_signal_zscore_{t-1}`. Modellen bruger rå næste-måneds lokale markedsafkast (`return_t_plus_1`) og evaluerer ikke længere afkast relativt til USA som benchmark. Brug `--signal-column country_signal_zscore` for niveaumodellen, `--signal-column signal_pct_change` for `signal_change / (abs(signal_lag1) + c)`, `--no-month-fixed-effects` for en model uden månedseffekter, eller `--min-country-months` for at filtrere lande med for få valide observationer.

Scriptet laver også en porteføljeorienteret ranking-test: hver måned sorteres lande i lav/mellem/høj signal-tercil, og testen evaluerer næste måneds gennemsnitlige `high - low` afkast med HAC-standardfejl.
## Globalt havnesignal

Hvis I vil teste om den samlede aktivitet i alle havne kan forudsige hvert lands marked, kan forecastet koeres med et globalt havnesignal:

```bash
python -m data.code.build_simple_forecast_model \
  --country-month-csv <output_dir>/country_month_activity.csv \
  --market-indices-csv data/outputs/market_indices/yahoo_finance/all_market_indices.csv \
  --out-dir <output_dir> \
  --signal-scope global_ports \
  --signal-column signal_change \
  --return-target raw \
  --min-country-months 12 \
  --no-month-fixed-effects
```

`global_ports` vaegter hvert maaneds landesignal med `port_count_used`, saa maaneder med flere brugbare havne taeller tilsvarende mere. Brug ikke maanedseffekter i denne tilstand, fordi det globale signal er ens for alle lande i samme maaned. Scriptet slaar dem automatisk fra, hvis de er sat. Ranking-testen springes automatisk over, fordi alle lande har samme globale signal i samme maaned.
