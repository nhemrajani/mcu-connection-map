"""Shared Wikipedia fetching, used by registry.py and plots.py.

One place for the polite-citizen rules: identify ourselves with a real
User-Agent (Wikipedia 403s the Python default), and back off properly when
told to slow down.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "mcu-connection-map/0.1 (github.com/nhemrajani/mcu-connection-map)"}
API = "https://en.wikipedia.org/w/api.php"


def api(**params):
    """Call the MediaWiki API with backoff.

    A 429 wants a real wait — seconds, not milliseconds — so honour
    Retry-After when it's sent and escalate steeply when it isn't.
    """
    params.setdefault("format", "json")
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(6):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA)))
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 5:
                raise
            wait = int(exc.headers.get("Retry-After") or 0) or min(60, 5 * 2 ** attempt)
            print(f"      rate-limited, waiting {wait}s")
            time.sleep(wait)
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 * (attempt + 1))


def sections(page):
    return api(action="parse", page=page, prop="sections")["parse"]["sections"]


def wikitext(page, index):
    return api(action="parse", page=page, prop="wikitext", section=index)["parse"]["wikitext"]["*"]


def page_html(page):
    return api(action="parse", page=page, prop="text")["parse"]["text"]["*"]
