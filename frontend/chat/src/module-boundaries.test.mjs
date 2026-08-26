import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { dirname, extname, join, relative, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const sourceRoot = dirname(fileURLToPath(import.meta.url));
const moduleFiles = readdirSync(sourceRoot, { recursive: true })
  .filter((name) => [".ts", ".tsx"].includes(extname(name)))
  .map((name) => resolve(sourceRoot, name));
const moduleSet = new Set(moduleFiles);

function resolveLocalImport(importer, specifier) {
  if (!specifier.startsWith(".")) return null;
  const target = resolve(dirname(importer), specifier);
  return [target, `${target}.ts`, `${target}.tsx`, join(target, "index.ts"), join(target, "index.tsx")]
    .find((candidate) => moduleSet.has(candidate)) ?? null;
}

function localDependencies(file) {
  const source = readFileSync(file, "utf8");
  const dependencies = new Set();
  const imports = /(?:import|export)\s+(?:type\s+)?(?:[^"']*?\s+from\s+)?["']([^"']+)["']|import\s*\(\s*["']([^"']+)["']\s*\)/g;
  for (const match of source.matchAll(imports)) {
    const dependency = resolveLocalImport(file, match[1] ?? match[2]);
    if (dependency) dependencies.add(dependency);
  }
  return [...dependencies];
}

test("chat source has an acyclic local module graph", () => {
  const graph = new Map(moduleFiles.map((file) => [file, localDependencies(file)]));
  const visiting = new Set();
  const visited = new Set();

  function visit(file, path) {
    if (visiting.has(file)) {
      const cycleStart = path.indexOf(file);
      const cycle = [...path.slice(cycleStart), file].map((item) => relative(sourceRoot, item));
      assert.fail(`circular dependency: ${cycle.join(" -> ")}`);
    }
    if (visited.has(file)) return;
    visiting.add(file);
    for (const dependency of graph.get(file) ?? []) visit(dependency, [...path, file]);
    visiting.delete(file);
    visited.add(file);
  }

  for (const file of moduleFiles) visit(file, []);
});

test("entry modules are dependency roots", () => {
  const entryModules = new Set([resolve(sourceRoot, "main.tsx"), resolve(sourceRoot, "mobile-entry.tsx")]);
  const dependents = moduleFiles.flatMap((file) =>
    localDependencies(file)
      .filter((dependency) => entryModules.has(dependency))
      .map((dependency) => `${relative(sourceRoot, file)} -> ${relative(sourceRoot, dependency)}`));
  assert.deepEqual(dependents, []);
});

test("desktop entry owns bootstrap without absorbing application state", () => {
  const entry = readFileSync(resolve(sourceRoot, "main.tsx"), "utf8");
  const app = readFileSync(resolve(sourceRoot, "desktop-chat-app.tsx"), "utf8");
  assert.match(entry, /createRoot/);
  assert.match(entry, /<DesktopChatApp/);
  assert.doesNotMatch(entry, /useState|useEffect|new WebSocket|fetchChatJson/);
  assert.doesNotMatch(app, /createRoot|initializeTheme|startCrossPortThemeSync/);
  assert.match(app, /useDesktopChatController\(\)/);
  assert.match(app, /<DesktopChatView/);
  assert.doesNotMatch(app, /useState|useEffect|new WebSocket|fetchChatJson|<DesktopComposer/);
});

test("mobile entry and Browser Lab share the app mount without importing each other", () => {
  const entry = readFileSync(resolve(sourceRoot, "mobile-entry.tsx"), "utf8");
  const mount = readFileSync(resolve(sourceRoot, "mobile-native-mount.tsx"), "utf8");
  const app = readFileSync(resolve(sourceRoot, "mobile-native.tsx"), "utf8");
  const labFrame = readFileSync(resolve(sourceRoot, "mobile-lab-frame.ts"), "utf8");
  assert.match(entry, /startMobileNativeApp\(installMobileBridge\)/);
  assert.doesNotMatch(entry, /createRoot|useState|useEffect/);
  assert.match(mount, /createRoot/);
  assert.match(mount, /<MobileNativeApp/);
  assert.doesNotMatch(app, /createRoot|initializeTheme|installMobileBridge/);
  assert.match(labFrame, /from "\.\/mobile-native-mount"/);
  assert.match(labFrame, /startMobileNativeApp\(installMobileBridge\)/);
  assert.doesNotMatch(labFrame, /import\("\.\/mobile-native"\)/);
});
