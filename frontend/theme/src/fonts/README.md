# LXGW WenKai GB Screen runtime shards

These four WOFF2 files are the Mobile WebUI-safe runtime form of
`frontend/theme/assets/fonts/LXGWWenKaiGBScreen.woff2` v1.522. Together they
cover the source font's complete Unicode cmap. Each file stays below the
Mobile WebUI 8 MiB per-file contract, while four files keep Android's eager
OTA download request count bounded.

Regenerate them from the repository root with FontTools 4.63.0:

```bash
python scripts/split-paper-font.py
```

The script verifies the source digest, full cmap coverage, shard size, and
generated CSS before replacing these checked-in runtime files.
