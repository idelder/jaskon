from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Literal

import requests

from .config import AppConfig


logger = logging.getLogger(__name__)


SearchProvider = Literal["serpapi", "bing"]


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


def _serpapi_search(query: str, *, api_key: str, k: int, timeout: float) -> list[SearchResult]:
    resp = requests.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": max(1, min(k, 10)),
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    results: list[SearchResult] = []
    for item in (data.get("organic_results") or [])[:k]:
        title = str(item.get("title") or "").strip()
        url = str(item.get("link") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if title and url:
            results.append(SearchResult(title=title, url=url, snippet=snippet))
    return results


def _bing_search(
    query: str,
    *,
    api_key: str,
    endpoint: str,
    k: int,
    timeout: float,
) -> list[SearchResult]:
    resp = requests.get(
        endpoint,
        params={"q": query, "count": max(1, min(k, 10)), "mkt": "en-US"},
        headers={"Ocp-Apim-Subscription-Key": api_key},
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    values = ((data.get("webPages") or {}).get("value") or [])[:k]
    results: list[SearchResult] = []
    for item in values:
        title = str(item.get("name") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        if title and url:
            results.append(SearchResult(title=title, url=url, snippet=snippet))
    return results


def build_search_query(cfg: AppConfig, user_request: str) -> str:
    user_request = user_request.strip()
    if cfg.default_location and cfg.default_location.strip():
        # Helps with queries like "weather" or "sunset".
        return f"{user_request} {cfg.default_location.strip()}"
    return user_request


def web_search(cfg: AppConfig, user_request: str, *, query_override: str | None = None) -> list[SearchResult]:
    if not cfg.enable_web_search:
        return []

    api_key = os.getenv(cfg.web_search_api_key_env)
    if not api_key:
        logger.warning(
            "Web search enabled but missing API key env var %r; continuing without search",
            cfg.web_search_api_key_env,
        )
        return []

    query = (query_override or "").strip() or build_search_query(cfg, user_request)
    provider: str = (cfg.web_search_provider or "").strip().lower()
    timeout = cfg.web_search_timeout_seconds
    k = cfg.web_search_results_k

    logger.info("Web searching (%s): %s", provider, query)

    try:
        if provider == "bing":
            endpoint = cfg.bing_search_endpoint or "https://api.bing.microsoft.com/v7.0/search"
            return _bing_search(query, api_key=api_key, endpoint=endpoint, k=k, timeout=timeout)
        # default: serpapi
        return _serpapi_search(query, api_key=api_key, k=k, timeout=timeout)
    except Exception:
        logger.exception("Web search failed; continuing without search")
        return []
