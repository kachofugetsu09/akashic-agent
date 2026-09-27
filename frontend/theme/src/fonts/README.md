# LXGW WenKai GB Screen runtime shards

These four WOFF2 subsets come from
`frontend/theme/assets/fonts/LXGWWenKaiGBScreen.woff2` v1.522. Together they
cover the source font's complete Unicode cmap. Unicode ranges let the browser
fetch only the subsets needed by a page.

Regenerate them from the repository root with FontTools 4.63.0:

```bash
python scripts/split-paper-font.py
```

The script verifies the source digest, full cmap coverage, and generated CSS
before replacing these checked-in runtime files.
