from __future__ import annotations

import logging
import os
import asyncio
from dataclasses import dataclass
from functools import lru_cache

import requests

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocationResult:
    location: str
    source: str


def _format_location(*, city: str | None, region: str | None, country: str | None) -> str | None:
    city = (city or "").strip()
    region = (region or "").strip()
    country = (country or "").strip()

    if city and region:
        return f"{city}, {region}"
    if city and country:
        return f"{city}, {country}"
    if city:
        return city
    if region and country:
        return f"{region}, {country}"
    if region:
        return region
    if country:
        return country
    return None


def _reverse_geocode_open_meteo(*, latitude: float, longitude: float, timeout_seconds: float) -> str | None:
    """Reverse geocode lat/lon into a human-friendly location string.

    Uses Open-Meteo's free reverse geocoding endpoint (no API key).
    """

    url = (
        "https://geocoding-api.open-meteo.com/v1/reverse"
        f"?latitude={latitude}&longitude={longitude}&count=1&language=en&format=json"
    )
    try:
        resp = requests.get(url, timeout=timeout_seconds)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        results = data.get("results") or []
        if not results:
            return None
        r0 = results[0] or {}
        return _format_location(
            city=r0.get("name"),
            region=r0.get("admin1"),
            country=r0.get("country"),
        )
    except Exception:
        logger.debug("Reverse geocode failed (open-meteo)", exc_info=True)
        return None


def _is_windows() -> bool:
    return os.name == "nt"


def detect_location_via_os(*, timeout_seconds: float = 3.0) -> LocationResult | None:
    """Best-effort OS-level location.

    On Windows, attempts Windows Location Services via WinRT if available.
    This can be significantly more accurate than IP geolocation.
    """

    if not _is_windows():
        return None

    # WinRT bindings are optional. If not installed/available, fall back.
    try:
        from winsdk.windows.devices.geolocation import Geolocator  # type: ignore
    except Exception:
        return None

    async def _get_lat_lon() -> tuple[float, float] | None:
        try:
            locator = Geolocator()
            # Default desired accuracy is fine; Windows decides based on available signals.
            pos = await locator.get_geoposition_async()
            coord = pos.coordinate
            return float(coord.point.position.latitude), float(coord.point.position.longitude)
        except Exception:
            return None

    try:
        latlon = asyncio.run(asyncio.wait_for(_get_lat_lon(), timeout=timeout_seconds))
    except RuntimeError:
        # If we're already in an event loop, fall back to synchronous-ish polling.
        try:
            loop = asyncio.get_event_loop()
            latlon = loop.run_until_complete(asyncio.wait_for(_get_lat_lon(), timeout=timeout_seconds))
        except Exception:
            latlon = None
    except Exception:
        latlon = None

    if not latlon:
        return None

    lat, lon = latlon
    loc = _reverse_geocode_open_meteo(latitude=lat, longitude=lon, timeout_seconds=timeout_seconds)
    if not loc:
        # As a fallback, keep a coordinate string (still useful for weather queries).
        loc = f"{lat:.5f}, {lon:.5f}"
    return LocationResult(location=loc, source="os")


@lru_cache(maxsize=1)
def detect_location_via_ip(*, timeout_seconds: float = 3.0) -> LocationResult | None:
    """Best-effort location detection using IP geolocation.

    Notes:
    - This is NOT GPS-precise. It depends on your network/IP.
    - It is intentionally best-effort and may fail (captive portals, VPNs, rate limits).
    """

    urls = [
        "https://ipapi.co/json/",  # generally reliable over HTTPS
        "https://ipinfo.io/json",  # may return region/city; sometimes rate-limited
    ]

    for url in urls:
        try:
            resp = requests.get(url, timeout=timeout_seconds)
            resp.raise_for_status()
            data = resp.json() if resp.content else {}

            # ipapi.co
            if "ipapi.co" in url:
                loc = _format_location(
                    city=data.get("city"),
                    region=data.get("region") or data.get("region_code"),
                    country=data.get("country_name") or data.get("country"),
                )
            else:
                # ipinfo.io
                loc = _format_location(
                    city=data.get("city"),
                    region=data.get("region"),
                    country=data.get("country"),
                )

            if loc:
                return LocationResult(location=loc, source=url)
        except Exception:
            logger.debug("IP location lookup failed via %s", url, exc_info=True)

    return None


@lru_cache(maxsize=4)
def _auto_detect_location(*, timeout_seconds: float, prefer_os: bool) -> LocationResult | None:
    # Intentionally do NOT fall back to IP-based location because it can be very inaccurate.
    # If OS location isn't available, return None.
    if prefer_os:
        return detect_location_via_os(timeout_seconds=timeout_seconds)
    return None


def resolve_default_location(
    *,
    configured_default: str | None,
    auto_enabled: bool,
    timeout_seconds: float = 3.0,
    prefer_os: bool = True,
) -> LocationResult | None:
    """Resolve the effective default location.

    Rules:
    - If configured_default is a real value (not 'auto'), it wins.
    - Otherwise, if auto_enabled, attempt IP geolocation.
    - Returns None if no location is available.
    """

    configured = (configured_default or "").strip()
    if configured and configured.lower() != "auto":
        return LocationResult(location=configured, source="config")

    if not auto_enabled:
        return None

    return _auto_detect_location(timeout_seconds=timeout_seconds, prefer_os=prefer_os)
