# SearXNG search

Enable `web_search_searxng`, open its settings in the Web family, and enter the
instance root URL. A deployment subpath is supported, for example
`https://search.example.org/searxng`. Do not append `/search` or query parameters.
The instance must enable `json` in `search.formats`.

After saving, select **SearXNG** as the search provider in **Web tools** when
more than one search provider is configured. The model continues to use
`web_search`. Clearing the address removes SearXNG from the available providers;
it does not silently fall back if Web tools explicitly selects SearXNG.

The provider sends `GET <root>/search?q=...&format=json` and returns at most the
requested number of results, each with `title`, `url`, and `content`. Engine
failures remain visible in `unresponsive_engines`. No results with engine failures
is an error; a valid empty response without engine failures is an empty search.
HTTP failures and invalid responses are errors, not empty results.

Settings are `endpoint` (unset by default) and `timeout_seconds` (25 seconds,
range 1–120). This provider supports instances that do not require authentication.
Use an address reachable from the Akashic process; its localhost refers to its
own host or container.

Run the installed-plugin HTTP scenarios:

```sh
.venv/bin/python scripts/searxng_search_scenario.py
.venv/bin/python scripts/searxng_search_scenario.py --endpoint http://localhost:8888
```
