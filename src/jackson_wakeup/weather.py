from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeocodeResult:
    name: str
    admin1: str | None
    country: str | None
    latitude: float
    longitude: float
    timezone: str | None


@dataclass(frozen=True)
class WeatherSummary:
    location: str
    latitude: float
    longitude: float
    timezone: str
    as_of_date: str
    summary_text: str


_WEATHER_CODE: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def geocode_location(*, location: str, timeout_seconds: float = 5.0) -> GeocodeResult | None:
    """Geocode a location string using Open-Meteo's free geocoding API."""

    raw = (location or "").strip()
    if not raw:
        return None

    # Open-Meteo's geocoder handles many "City, Country" queries well, but common
    # US-style "City, ST" (e.g. "Boston, MA") often returns zero results.
    # Try a small set of normalized fallbacks.
    queries: list[str] = []
    queries.append(raw)
    if "," in raw:
        head, tail = raw.split(",", 1)
        head = head.strip()
        tail = tail.strip()
        if head:
            # If tail looks like a state/province abbreviation, prefer an explicit country.
            if tail.isalpha() and len(tail) in (2, 3):
                queries.append(f"{head}, United States")
            queries.append(head)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    queries = [q for q in queries if q and not (q in seen or seen.add(q))]

    url = "https://geocoding-api.open-meteo.com/v1/search"

    for q in queries:
        params = {
            "name": q,
            "count": 1,
            "language": "en",
            "format": "json",
        }

        try:
            resp = requests.get(url, params=params, timeout=timeout_seconds)
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
            results = data.get("results") or []
            if not results:
                continue
            r0 = results[0] or {}
            name = str(r0.get("name") or "").strip()
            if not name:
                continue
            return GeocodeResult(
                name=name,
                admin1=(str(r0.get("admin1") or "").strip() or None),
                country=(str(r0.get("country") or "").strip() or None),
                latitude=float(r0.get("latitude")),
                longitude=float(r0.get("longitude")),
                timezone=(str(r0.get("timezone") or "").strip() or None),
            )
        except Exception:
            logger.debug("Open-Meteo geocoding failed for query=%r", q, exc_info=True)

    return None


def fetch_today_forecast(
    *,
    latitude: float,
    longitude: float,
    timezone: str = "auto",
    timeout_seconds: float = 7.0,
) -> dict:
    """Fetch forecast JSON from Open-Meteo."""

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,weather_code",
        "forecast_days": 1,
    }

    resp = requests.get(url, params=params, timeout=timeout_seconds)
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def build_weather_summary(*, location: str, timeout_seconds: float = 7.0) -> WeatherSummary | None:
    """Return a concise weather summary for today's forecast for a given location."""

    geo = geocode_location(location=location, timeout_seconds=timeout_seconds)
    if not geo:
        return None

    tz = geo.timezone or "auto"

    try:
        data = fetch_today_forecast(
            latitude=geo.latitude,
            longitude=geo.longitude,
            timezone=tz,
            timeout_seconds=timeout_seconds,
        )
    except Exception:
        logger.debug("Open-Meteo forecast fetch failed", exc_info=True)
        return None

    current = data.get("current") or {}
    daily = data.get("daily") or {}

    def _first(v):
        if isinstance(v, list) and v:
            return v[0]
        return None

    as_of = str(_first(daily.get("time")) or date.today().isoformat())
    tmax = _first(daily.get("temperature_2m_max"))
    tmin = _first(daily.get("temperature_2m_min"))
    precip = _first(daily.get("precipitation_sum"))
    windmax = _first(daily.get("wind_speed_10m_max"))
    wcode_daily = _first(daily.get("weather_code"))

    cur_temp = current.get("temperature_2m")
    cur_rh = current.get("relative_humidity_2m")
    cur_wind = current.get("wind_speed_10m")
    cur_wcode = current.get("weather_code")

    code = None
    for candidate in (wcode_daily, cur_wcode):
        try:
            if candidate is not None:
                code = int(candidate)
                break
        except Exception:
            continue
    cond = _WEATHER_CODE.get(code, "") if code is not None else ""

    loc_bits = [geo.name]
    if geo.admin1:
        loc_bits.append(geo.admin1)
    if geo.country:
        loc_bits.append(geo.country)
    loc_name = ", ".join(loc_bits)

    # Units are provided by Open-Meteo defaults (°C, km/h, mm). Keep it explicit.
    parts: list[str] = []
    if cond:
        parts.append(f"Condition: {cond}")
    if tmin is not None and tmax is not None:
        parts.append(f"Temp: {tmin}–{tmax} °C")
    elif tmax is not None:
        parts.append(f"High: {tmax} °C")
    if precip is not None:
        parts.append(f"Precip: {precip} mm")
    if windmax is not None:
        parts.append(f"Max wind: {windmax} km/h")
    if cur_temp is not None:
        parts.append(f"Now: {cur_temp} °C")
    if cur_rh is not None:
        parts.append(f"RH: {cur_rh} %")
    if cur_wind is not None:
        parts.append(f"Wind now: {cur_wind} km/h")

    summary = "\n".join(parts).strip()
    if not summary:
        return None

    return WeatherSummary(
        location=loc_name,
        latitude=geo.latitude,
        longitude=geo.longitude,
        timezone=tz,
        as_of_date=as_of,
        summary_text=summary,
    )
