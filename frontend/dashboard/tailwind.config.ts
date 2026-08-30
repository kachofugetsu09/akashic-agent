import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import type { Config } from "tailwindcss";
import webUiPreset from "../../packages/akashic-web-ui-v1/tailwind-preset.mjs";

// Content globs are resolved against this config's own location so scanning
// works regardless of the process cwd (npm runs from the repo root).
const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");

export default {
  content: [
    resolve(here, "index.html"),
    resolve(here, "src/**/*.{ts,tsx}"),
  ],
  presets: [webUiPreset],
} satisfies Config;
