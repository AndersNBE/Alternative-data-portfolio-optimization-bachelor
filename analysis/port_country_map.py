# Maps each port_slug to the country name used in fetch_market_indices.py.
# Ports whose country has no matching stock index (e.g. Marokko) will appear
# in port_timeseries.csv but be excluded from the country-level GNC signal.

PORT_COUNTRY: dict[str, str] = {
    "abu_dhabi":           "UAE",
    "algeciras":           "Spanien",
    "antwerpbrugges":      "Belgien",
    "balboa":              "Panama",
    "bremen":              "Tyskland",
    "busan":               "Sydkorea",
    "cai_mep":             "Vietnam",
    "colombo":             "Sri Lanka",
    "colon":               "Panama",
    "da_lian":             "Kina",
    "dongguan":            "Kina",
    "guangxi_beibu":       "Kina",
    "guangzhou":           "Kina",
    "hai_phong":           "Vietnam",
    "hamborg":             "Tyskland",
    "ho_chi_minh_city":    "Vietnam",
    "hong_kong":           "Kina",
    "houston":             "USA",
    "jawaharal_nehru":     "Indien",
    "jebel_ali":           "UAE",
    "kaohsiung":           "Taiwan",
    "laem_chabang":        "Thailand",
    "lianyungang":         "Kina",
    "long_beach":          "USA",
    "los_angeles":         "USA",
    "manila":              "Filippinerne",
    "mundra":              "Indien",
    "new_york_new_jersey": "USA",
    "ningbozhoushan":      "Kina",
    "piraeus":             "Grækenland",
    "port_klang":          "Malaysia",
    "qing_dao":            "Kina",
    "rizhao":              "Kina",
    "rotterdam":           "Holland",
    "santos":              "Brasilien",
    "savannah":            "USA",
    "shanghai":            "Kina",
    "shenzhen":            "Kina",
    "singapore":           "Singapore",
    "suzhou":              "Kina",
    "tanger_med":          "Marokko",
    "tanjung_pelepas":     "Malaysia",
    "tanjung_perak":       "Malaysia",
    "tanjung_priok":       "Indonesia",
    "tianjin":             "Kina",
    "tokyo":               "Japan",
    "valencia":            "Spanien",
    "xiamen":              "Kina",
    "yantai":              "Kina",
    "yingkou":             "Kina",
}


def port_to_country(port_slug: str) -> str | None:
    return PORT_COUNTRY.get(port_slug)


def all_countries() -> list[str]:
    return sorted(set(PORT_COUNTRY.values()))


def ports_for_country(country: str) -> list[str]:
    return sorted(slug for slug, c in PORT_COUNTRY.items() if c == country)
