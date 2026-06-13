# Lokale fallback-filer til markedsindekser

Denne mappe bruges af `python -m data.code.fetch_market_indices`, hvis Yahoo Finance ikke leverer brugbare data for et land.

Brug en CSV per land med filnavnet:
- `panama.csv`
- `sri_lanka.csv`

Minimumskrav til kolonner:
- `date`
- `adj_close`

Valgfrie kolonner:
- `close`
- `open`
- `high`
- `low`
- `volume`

Hvis `close` mangler, saettes den automatisk lig `adj_close`.

Eksempel:

```csv
date,adj_close,close,open,high,low,volume
2017-01-02,100.0,100.0,100.0,100.5,99.8,100000
2017-01-03,100.8,100.8,100.1,101.0,99.9,120000
```

Anbefalet brug:
- Panama: eksportér et bredt markedsindeks fra den kilde I vaelger, og gem som `panama.csv`
- Sri Lanka: eksportér ASPI-serien og gem som `sri_lanka.csv`
