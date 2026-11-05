"""The shared GET: what is retried, how long it waits, what is returned.
No network is used: the session is replaced."""
import pytest
import requests

import http_get as H


class Resp:
    def __init__(self, status, headers=None):
        self.status_code, self.headers = status, headers or {}


class Session:
    """Answers each GET with the next item; an exception item is raised."""
    def __init__(self, *items):
        self.items, self.calls = list(items), 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_a_5xx_is_retried_and_the_next_response_returned():
    """Prevents: a passing 503 failing a pull that a second try would make."""
    waits = []
    r, attempt = H.get(Session(Resp(503), Resp(200)), "u", sleep=waits.append)
    assert (r.status_code, attempt, waits) == (200, 2, [2.0])


def test_a_429_waits_as_long_as_the_server_asks():
    """Prevents: hammering a rate-limited server at our own backoff."""
    waits = []
    r, _ = H.get(Session(Resp(429, {"Retry-After": "7"}), Resp(200)), "u", sleep=waits.append)
    assert r.status_code == 200 and waits == [7.0]


def test_a_retry_after_given_as_a_date_falls_back_to_a_minute():
    assert H.retry_after(Resp(429, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})) == 60.0


def test_a_404_is_returned_at_once_for_the_caller_to_judge():
    """Prevents: retrying a request that cannot succeed."""
    s = Session(Resp(404))
    r, attempt = H.get(s, "u", sleep=lambda w: pytest.fail("no wait expected"))
    assert (r.status_code, attempt, s.calls) == (404, 1, 1)


def test_network_errors_end_in_one_clear_error_after_every_attempt():
    s = Session(*[requests.ConnectionError("down")] * H.MAX_RETRIES)
    waits = []
    with pytest.raises(RuntimeError, match="no usable response"):
        H.get(s, "u", sleep=waits.append)
    assert s.calls == H.MAX_RETRIES and len(waits) == H.MAX_RETRIES - 1
    assert max(waits) <= 30


def test_every_request_names_the_project_and_has_a_timeout():
    assert "soil-carbon-kriging" in H.session().headers["User-Agent"]
    assert all(t > 0 for t in H.TIMEOUT)
