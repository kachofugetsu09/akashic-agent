from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from scripts.akashic_release.activate import activate_release
from scripts.akashic_release.doctor import verify_release
from scripts.akashic_release.manifest import read_json, release_lock
from scripts.akashic_release.migrate import migration_plan
from scripts.akashic_release.model import ReleasePaths
from scripts.akashic_release.prepare import (
    prepare_generation,
    verify_host_prerequisites,
)
from scripts.akashic_release.source import commit_subject, resolve_target
from scripts.akashic_release.source import verify_bootstrap_checkout
from scripts.akashic_release.systemd import install_units
from scripts.akashic_release.systemd import install_operator_entrypoint
from scripts.akashic_release.systemd import verify_external_service_contract

_DEFAULT_ORIGIN = "https://github.com/kachofugetsu09/akashic-agent.git"
_DEFAULT_ROOT = Path("/srv/data/services/akashic")
_DEFAULT_ENV = Path.home() / ".config/akashic-container/runtime.env"
_DEFAULT_MISE = Path.home() / ".local/bin/mise"


_run = subprocess.run


def _confirm(commit: str, subject: str, current: str | None, *, yes: bool) -> None:
    print(f"current: {current or 'none'}")
    print(f"target:  {commit}  {subject}")
    if yes:
        return
    if not sys.stdin.isatty():
        raise RuntimeError("非交互安装必须显式传 --yes")
    if (
        input("Install this Core + Host Bridge generation? [y/N] ").strip().lower()
        != "y"
    ):
        raise RuntimeError("operator 取消安装")


def install(args: argparse.Namespace) -> dict[str, object]:
    paths = ReleasePaths(args.root.resolve())
    checkout = args.source_checkout.resolve(strict=True)
    commit = resolve_target(args.origin, args.commit, run=_run)
    verify_bootstrap_checkout(checkout, commit, args.origin, run=_run)
    active_path = paths.activation / "active.json"
    current = (
        str(read_json(active_path).get("targetCommit"))
        if active_path.exists()
        else None
    )
    _confirm(commit, commit_subject(checkout, run=_run), current, yes=args.yes)

    with release_lock(paths.run / "release.lock"):
        paths.create_layout()
        verify_host_prerequisites(mise=args.mise, run=_run)
        verify_external_service_contract(run=_run, unit_root=args.unit_root)
        manifest = prepare_generation(
            paths=paths,
            bootstrap_checkout=checkout,
            commit=commit,
            origin=args.origin,
            mise=args.mise,
            run=_run,
        )
        units_changed = install_units(
            checkout=paths.source(commit),
            backup_root=paths.backups,
            run=_run,
            unit_root=args.unit_root,
        )
        cli_changed = install_operator_entrypoint(
            checkout=paths.source(commit),
            backup_root=paths.backups,
            target=args.cli_path,
        )
        status = "prepared"
        if not args.no_activate:
            status = activate_release(
                paths=paths,
                manifest_path=paths.release(commit),
                environment_file=args.runtime_env,
                mise=args.mise,
                run=_run,
            )
    return {
        "status": status,
        "commit": commit,
        "imageId": manifest["imageId"],
        "unitsChanged": units_changed,
        "cliChanged": cli_changed,
    }


def doctor(args: argparse.Namespace) -> dict[str, object]:
    verify_release(args.runtime_env)
    return {"status": "healthy", "runtimeEnv": str(args.runtime_env)}


def rollback(args: argparse.Namespace) -> dict[str, object]:
    paths = ReleasePaths(args.root.resolve(strict=True))
    previous_path = paths.activation / "previous.json"
    previous = str(read_json(previous_path)["targetCommit"])
    _confirm(previous, "previous prepared generation", None, yes=args.yes)
    with release_lock(paths.run / "release.lock"):
        status = activate_release(
            paths=paths,
            manifest_path=paths.release(previous),
            environment_file=args.runtime_env,
            mise=args.mise,
            run=_run,
        )
    return {"status": status, "commit": previous}


def pair_mobile(args: argparse.Namespace) -> dict[str, object]:
    from scripts.akashic_release.mobile_pair import pair_mobile as run_pairing

    return run_pairing(args.runtime_env)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="akashic-release")
    subparsers = parser.add_subparsers(dest="command", required=True)

    install_parser = subparsers.add_parser("install")
    install_parser.add_argument("--source-checkout", type=Path, required=True)
    install_parser.add_argument("--commit")
    install_parser.add_argument("--origin", default=_DEFAULT_ORIGIN)
    install_parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    install_parser.add_argument("--runtime-env", type=Path, default=_DEFAULT_ENV)
    install_parser.add_argument("--mise", type=Path, default=_DEFAULT_MISE)
    install_parser.add_argument(
        "--unit-root", type=Path, default=Path("/etc/systemd/system")
    )
    install_parser.add_argument(
        "--cli-path", type=Path, default=Path.home() / ".local/bin/akashic-release"
    )
    install_parser.add_argument("--yes", action="store_true")
    install_parser.add_argument("--no-activate", action="store_true")
    install_parser.set_defaults(handler=install)

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--runtime-env", type=Path, default=_DEFAULT_ENV)
    doctor_parser.set_defaults(handler=doctor)

    rollback_parser = subparsers.add_parser("rollback")
    rollback_parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    rollback_parser.add_argument("--runtime-env", type=Path, default=_DEFAULT_ENV)
    rollback_parser.add_argument("--mise", type=Path, default=_DEFAULT_MISE)
    rollback_parser.add_argument("--yes", action="store_true")
    rollback_parser.set_defaults(handler=rollback)

    pair_parser = subparsers.add_parser("pair-mobile")
    pair_parser.add_argument("--runtime-env", type=Path, default=_DEFAULT_ENV)
    pair_parser.set_defaults(handler=pair_mobile)

    migrate_parser = subparsers.add_parser("migrate")
    migrate_parser.add_argument("--snapshot-manifest", type=Path, required=True)
    migrate_parser.set_defaults(
        handler=lambda args: migration_plan(args.snapshot_manifest)
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
