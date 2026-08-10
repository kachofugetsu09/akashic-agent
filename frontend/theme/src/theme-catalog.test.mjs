import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const catalog = JSON.parse(await readFile(new URL("./theme-catalog.json", import.meta.url), "utf8"));
const runtime = await readFile(new URL("./theme-runtime.ts", import.meta.url), "utf8");

const materialRoles = [
  "primary", "onPrimary", "primaryContainer", "onPrimaryContainer",
  "secondary", "onSecondary", "secondaryContainer", "onSecondaryContainer",
  "tertiary", "onTertiary", "tertiaryContainer", "onTertiaryContainer",
  "error", "onError", "errorContainer", "onErrorContainer",
  "surface", "onSurface", "surfaceVariant", "onSurfaceVariant",
  "outline", "outlineVariant", "inverseSurface", "inverseOnSurface", "inversePrimary",
  "surfaceDim", "surfaceBright", "surfaceContainerLowest", "surfaceContainerLow",
  "surfaceContainer", "surfaceContainerHigh", "surfaceContainerHighest", "surfaceTint",
];

const domainRoles = [
  "success", "onSuccess", "successContainer", "onSuccessContainer",
  "warning", "onWarning", "warningContainer", "onWarningContainer",
  "trace", "onTrace", "traceContainer", "onTraceContainer",
  "info", "onInfo", "infoContainer", "onInfoContainer",
];

test("every Akashic theme owns the complete Material and domain semantic roles", () => {
  assert.equal(catalog.version, 2);
  assert.ok(catalog.themes.some((theme) => theme.id === catalog.defaultThemeId));
  for (const theme of catalog.themes) {
    for (const role of materialRoles) assert.match(theme.material[role], /^#[0-9a-f]{6}(?:[0-9a-f]{2})?$/i, `${theme.id}.material.${role}`);
    for (const role of domainRoles) assert.match(theme.domain[role], /^#[0-9a-f]{6}(?:[0-9a-f]{2})?$/i, `${theme.id}.domain.${role}`);
  }
});

test("theme runtime publishes Material, Akashic domain, and legacy compatibility namespaces", () => {
  assert.match(runtime, /colorDeclarations\("md-sys-color"/);
  assert.match(runtime, /colorDeclarations\("ak-sys-color"/);
  assert.match(runtime, /colorDeclarations\("ak-color"/);
});
