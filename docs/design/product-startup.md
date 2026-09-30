# Product startup

Status: implementation; local acceptance is recorded in the pull request.

`./start` owns product preparation. It reports terminal progress, builds one committed distribution, installs its default profile through the formal installer, and starts that distribution's Supervisor. Core does not discover or install business plugins.

```text
┌───────────────────────────────────────┐
│ Terminal: product preparation         │
│ dependencies → distribution → install │
└──────────────────┬────────────────────┘
                   │ start Supervisor, wait for readiness
┌──────────────────▼────────────────────┐
│ Supervisor owns port 2236              │
│ Web shell + plugin gateway ready       │
└──────────────────┬────────────────────┘
                   │ print URL and open browser
┌──────────────────▼────────────────────┐
│ existing onboarding → Chat             │
└───────────────────────────────────────┘
```

Preparation status is process-local terminal output. The launcher waits for `chatReady` and a successful Web bootstrap before opening the browser. `--no-browser` keeps URL output for headless machines and Agents. Failures return a nonzero exit code with the log path; fix the cause and run the same command again. Ctrl+C or SIGTERM stops the child process group before releasing the state lock. The launcher remains attached while Supervisor runs. Compose shows the same progress in its foreground output or Docker Desktop logs; it never tries to open a browser inside the container.

## State and versions

- `--state` contains config.toml, workspace, plugin-home, startup.json and per-launch logs.
- `startup.json` identifies the launcher-owned first installation and its software commit. It does not hold plugin choices or business progress. It remains the historical first-install marker, not the currently deployed revision; the current distribution/runtime identity is authoritative. A new distribution composes its built-in code before startup; pending data migrations still require explicit approval.
- `workspace/runtime/distribution-install.json` remains the formal install receipt. Later starts retain this receipt unchanged as legacy source evidence, compose current distribution sources with selected external inputs, and preserve disabled or uninstalled choices. See [0082](../decisions/0082-distribution-owned-plugin-composition.md).
- A nonempty installation without the launch marker is not adopted automatically. Use its original entry or a separate empty state directory.
- Native build cache is scoped to the checkout and committed revision. Failed staging directories and logs remain for inspection; completed environments are not moved because virtual environments contain absolute paths.
- Source builds reject tracked local edits. They do not silently build an older HEAD while claiming to run those edits.

The standalone Compose image builds a fetched Git commit in a builder stage. It contains the Core archive and independent plugin bundles, with no business plugin source in Core. Default tools run inside the container; the Docker socket and host filesystem are not exposed. Plugins needing external Workload infrastructure require that infrastructure to be explicitly configured; the default profile does not start Computer.

The image workflow is manually dispatched and publishes a commit-tagged GHCR image. Before this PR is merged, candidate validation sets `AKASHIC_REVISION` to the full pushed candidate SHA; upstream main does not yet contain this entry. The default Compose works through a source build without assuming that a public image tag already exists. Image publishing and production deployment are separate from this PR's local validation.

## Development entry

For direct checkout development, prepare Python and Node dependencies, build the Web assets, and explicitly install the desired plugin distribution once. The product launcher is not a watcher for dirty business-plugin source.

```bash
uv venv
uv pip install -r requirements.txt -e sdk/python
npm ci
npm run build
uv run python main.py init
# Build and install a committed distribution into the chosen workspace/plugin home.
uv run python main.py
```

The full build/install API is `scripts/build_plugin_distribution.py --help` and `scripts/install_plugin_distribution.py --help`. Use explicit config, workspace and plugins-home paths for development isolation. Product starts and [operator deployment](operator-deployment.md) use the same distribution composition policy. Migration approval, backup and actual runtime readiness remain separate from the complete selection commit.

## Acceptance

Use isolated HOME, state, plugin home, cache, browser profile and Compose project. Verify the distribution commit, default profile, formal install receipt, selected runtime and actual Web bootstrap modules. Exercise cold start, warm start, failed preparation and rerun, port collision, stop/restart, and persistence. An HTTP 200 or a running container alone does not prove that the plugins loaded.

## Updating standalone Compose

After `git pull --ff-only`, use `AKASHIC_REVISION="$(git rev-parse HEAD)" docker compose up -d --build`. The full commit changes Docker's build argument and binds the fetched source. `main` is a floating input whose fetch layer can be cached; `git pull` plus an unchanged `main` build argument does not prove that a new image was built. Keep the same Compose project and data volume; never use `down -v` for an update.
