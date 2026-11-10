import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import type { Plugin } from "vite";

const here = dirname(fileURLToPath(import.meta.url));
const catalogPath = resolve(here, "theme-catalog.json");

export function emitThemeCatalog(): Plugin {
  return {
    name: "akashic-theme-catalog",
    generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: "akashic-theme-catalog.json",
        source: readFileSync(catalogPath, "utf8"),
      });
      this.emitFile({
        type: "asset",
        fileName: "reading-fonts-OFL.txt",
        source: ["notoserifsc", "sourceserif4"]
          .map((name) => readFileSync(resolve(here, "fonts", `${name}-OFL.txt`), "utf8"))
          .join("\n\n"),
      });
    },
  };
}
