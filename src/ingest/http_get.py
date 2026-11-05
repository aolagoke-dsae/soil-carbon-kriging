"""One HTTP GET with the retry rules both clients share.

Retried: a connection error or timeout, HTTP 429 (after its Retry-After) and
any 5xx, up to MAX_RETRIES attempts with exponential backoff capped at 30 s.
Not retried: any other status. It is returned, and the caller decides whether
the body is data; raw_store keeps it under _failed/ if it is not.

Neither source needs a token. The User-Agent names the project, so that an
operator who sees the traffic can find out who sent it.

Named http_get, not http: src/ingest is put on sys.path for the tests, and a
module called http there would shadow the standard library's, which requests
itself imports.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import requests

USER_AGENT = "soil-carbon-kriging (+https://github.com/aolagoke-dsae/soil-carbon-kriging)"
TIMEOUT = (10, 120)         # seconds: connect, read
MAX_RETRIES = 4


def session() -> requests.Session:
    """A session that sends the project's User-Agent and reuses connections."""
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def retry_after(response, default: float = 60.0) -> float:
    """Seconds to wait after a 429: Retry-After when given in seconds."""
    try:
        return float(response.headers.get("Retry-After", default))
    except (TypeError, ValueError):     # an HTTP date rather than seconds
        return default


def get(http, url: str, params: Optional[dict] = None, *,
        max_retries: int = MAX_RETRIES,
        sleep: Optional[Callable[[float], None]] = None):
    """GET url and return (response, attempt number). Raises RuntimeError
    when every attempt ended in a network error, a 429 or a 5xx. sleep
    defaults to time.sleep, looked up at call time."""
    sleep = sleep or time.sleep
    last: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        wait = float(min(2 ** attempt, 30))
        try:
            r = http.get(url, params=params, timeout=TIMEOUT)
        except requests.RequestException as e:
            last = e
        else:
            if r.status_code != 429 and r.status_code < 500:
                return r, attempt
            last = requests.HTTPError(f"HTTP {r.status_code}")
            if r.status_code == 429:
                wait = retry_after(r)
        if attempt < max_retries:
            sleep(wait)
    raise RuntimeError(f"{url}: no usable response after {max_retries} attempts ({last})")
