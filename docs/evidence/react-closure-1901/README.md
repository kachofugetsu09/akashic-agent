# Failed turn recovery (#1901)

Base: `7ae9b99178ba68921e5ec3229fec70c94df221e2`.

The screenshots use a production Vite build of `frontend/chat`, Chromium,
real WebChat HTTP/WebSocket routes, installed reply and compaction plugins,
and an isolated SQLite workspace. The external model is a deterministic
10,000-token peer. A local ingress adapter connects WebChat to the real
channel input service. This is not a production or provider acceptance run.

- `capacity-failed.png`: sending 62 KB of text fails visibly and offers abandon.
- `capacity-recovered.png`: after clicking abandon, a shorter input completes.
  Reloading the page retains the answer and the earlier messages.
- `tool-unknown.png`: an old call without a receipt shows unknown state,
  including the warning about possible external effects; it offers abandon.

Backend replay also verified that a new short input can summarize an earlier
failed input, including one from a different source. Programmatic abandon
settles an old call as interrupted without invoking the tool, permits the next
input, preserves original rows and learning attributes, and is idempotent after
new work completes. The old source still fails until explicitly abandoned;
we do not invent a successful tool result or automatically repeat its effects.

Local checks: 23 existing pytest cases, pyright (0 errors), plugin boundary,
TypeScript checking, dashboard and chat production builds. No tests were added.

Reproduction scripts and complete before/after logs remain outside the repo at
`/home/huashen/.cache/akashic-review/react-closure/evidence/1901/`:
`backend.py`, `browser_server.py`, `browser.mjs`, `verified/results.json`, and
`browser-results.json`. The fixture uses only local temporary workspaces.
