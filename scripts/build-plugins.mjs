import { existsSync, readdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn, spawnSync } from "node:child_process";

const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const pluginsRoot = join(projectRoot, "plugins");
const isWindows = process.platform === "win32";
const localEsbuild = join(
  projectRoot,
  "node_modules",
  ".bin",
  isWindows ? "esbuild.cmd" : "esbuild",
);

const watchMode = process.argv.includes("--watch");

function resolveEsbuildCommand() {
  if (existsSync(localEsbuild)) {
    return [localEsbuild];
  }
  if (isWindows) {
    return ["cmd.exe", "/d", "/s", "/c", "npx", "--yes", "esbuild"];
  }
  return ["npx", "--yes", "esbuild"];
}

function listPluginPanels() {
  if (!existsSync(pluginsRoot)) {
    return [];
  }
  return readdirSync(pluginsRoot)
    .map((name) => join(pluginsRoot, name))
    .filter((path) => statSync(path, { throwIfNoEntry: false })?.isDirectory())
    .map((pluginDir) => ({
      pluginDir,
      tsPath: join(pluginDir, "dashboard_panel.ts"),
      jsPath: join(pluginDir, "dashboard_panel.js"),
    }))
    .filter((item) => existsSync(item.tsPath));
}

function buildArgs(command, panel, { watch = false } = {}) {
  return [
    ...command.slice(1),
    panel.tsPath,
    `--outfile=${panel.jsPath}`,
    "--bundle=false",
    "--platform=browser",
    "--target=es2020",
    "--format=iife",
    ...(watch ? ["--watch"] : []),
  ];
}

function buildOne(command, panel) {
  const result = spawnSync(command[0], buildArgs(command, panel), {
    cwd: projectRoot,
    stdio: "inherit",
    shell: false,
  });
  if (typeof result.status === "number" && result.status !== 0) {
    process.exitCode = result.status;
  }
  if (result.error) {
    throw result.error;
  }
}

function watchAll(command, panels) {
  const children = panels.map((panel) =>
    spawn(command[0], buildArgs(command, panel, { watch: true }), {
      cwd: projectRoot,
      stdio: "inherit",
      shell: false,
    }),
  );

  let shuttingDown = false;
  const shutdown = () => {
    if (shuttingDown) {
      return;
    }
    shuttingDown = true;
    for (const child of children) {
      killProcessTree(child);
    }
  };

  process.once("SIGINT", () => {
    shutdown();
    process.exit(130);
  });
  process.once("SIGTERM", () => {
    shutdown();
    process.exit(143);
  });

  for (const child of children) {
    child.on("error", (error) => {
      console.error(error);
      process.exitCode = 1;
      shutdown();
    });
    child.on("exit", (code, signal) => {
      if (shuttingDown) {
        return;
      }
      if (code && code !== 0) {
        process.exitCode = code;
        shutdown();
      } else if (signal) {
        process.exitCode = 1;
        shutdown();
      }
    });
  }
}

function killProcessTree(child) {
  if (child.killed || child.exitCode !== null) {
    return;
  }
  if (isWindows) {
    const killer = spawn(
      "taskkill",
      ["/PID", String(child.pid), "/T", "/F"],
      { stdio: "ignore", windowsHide: true },
    );
    killer.on("error", () => child.kill());
    return;
  }
  child.kill();
}

const esbuildCommand = resolveEsbuildCommand();
const panels = listPluginPanels();

if (panels.length === 0) {
  console.log("No plugin dashboard panels found.");
  process.exit(0);
}

if (watchMode) {
  watchAll(esbuildCommand, panels);
} else {
  for (const panel of panels) {
    buildOne(esbuildCommand, panel);
  }
}
