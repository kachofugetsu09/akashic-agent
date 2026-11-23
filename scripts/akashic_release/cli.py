from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from scripts.akashic_release.activate import activate_release, resume_release
from scripts.akashic_release.doctor import verify_release
from scripts.akashic_release.extra import ExtraOptions
from scripts.akashic_release.manifest import read_json, release_lock
from scripts.akashic_release.migrate import migration_plan
from scripts.akashic_release.model import ReleasePaths
from scripts.akashic_release.prepare import (
    prepare_generation,
    verify_host_prerequisites,
)
from scripts.akashic_release.source import commit_subject, resolve_target
from scripts.akashic_release.source import verify_bootstrap_checkout
from scripts.akashic_release.warm import StateCacheWarmer

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


def install(args: argparse.Namespace, *, self_deploy: dict[str, object] | None = None) -> dict[str, object]:
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

    if args.inputs is not None and args.plan is None:
        raise ValueError("--inputs 需要 --plan")
    if args.no_activate and (args.plan is not None or args.backup):
        raise ValueError("--no-activate 只准备产物；部署清单和备份选项在实际安装时提供")

    with release_lock(paths.run / "release.lock"):
        paths.create_layout()
        verify_host_prerequisites(mise=args.mise, run=_run)
        # 页缓存跨进程保留：只准备产物的一轮也预热，自部署 worker 停机前即可命中。
        warmer = StateCacheWarmer(paths.state / "workspace")
        warmer.start()
        manifest = prepare_generation(
            paths=paths,
            bootstrap_checkout=checkout,
            commit=commit,
            origin=args.origin,
            mise=args.mise,
            run=_run,
            runtime_env=args.runtime_env,
            installed_unit=args.unit_root / "akashic-host-bridge.service",
        )
        status = "prepared"
        if args.no_activate:
            warmer.wait()
        else:
            status = activate_release(
                paths=paths,
                manifest_path=paths.release(commit),
                environment_file=args.runtime_env,
                mise=args.mise,
                run=_run,
                plan=args.plan,
                inputs=args.inputs,
                backup=args.backup,
                unit_root=args.unit_root, cli_path=args.cli_path,
                self_deploy=self_deploy, warmer=warmer,
                extra=ExtraOptions(enabled=not getattr(args, "no_extra", False),
                                   repo=getattr(args, "extra_repo", None),
                                   inputs=getattr(args, "extra_inputs", None),
                                   allow_retire=getattr(args, "allow_retire", False),
                                   profile=getattr(args, "profile", None),
                                   groups=tuple(item for item in getattr(args, "groups", "").split(",") if item),
                                   assume_yes=getattr(args, "yes", False), reselect=getattr(args, "reselect", False)),
            )
    # 降级的外置插件进入 install 结果，自部署 worker 的 status 也能看到。
    runtime_check = (read_json(paths.activation / "active.json").get("runtimeCheck")
                     if status == "activated" else None)
    return {
        "status": status,
        "commit": commit,
        "imageId": manifest["imageId"],
        "backupRequested": args.backup,
        "extra": read_json(paths.activation / "active.json").get("extra") if status == "activated" else None,
        "degradedExternal": (runtime_check.get("degraded_external", {})
                             if isinstance(runtime_check, dict) else {}),
    }


def doctor(args: argparse.Namespace) -> dict[str, object]:
    verify_release(args.runtime_env)
    return {"status": "healthy", "runtimeEnv": str(args.runtime_env)}


def resume(args: argparse.Namespace) -> dict[str, object]:
    paths = ReleasePaths(args.root.resolve(strict=True))
    with release_lock(paths.run / "release.lock"):
        status = resume_release(paths=paths, attempt_path=args.attempt,
                                environment_file=args.runtime_env, mise=args.mise, run=_run,
                                unit_root=args.unit_root, cli_path=args.cli_path)
    return {"status": status, "attempt": str(args.attempt)}


# 只读：当前实例的用户 patch 相对正在运行的 preset 改了什么，以及哪些 row 已经和 preset 相同。
# preset 就是运行检出里的 bundles/；发行版市场的 row 指向检出里已没有的插件时列为 notShipped（退役或改名留下的），
# 同 DSH 的 patch 未匹配警告。
def preset_diff(args: argparse.Namespace) -> dict[str, object]:
    from agent.plugins.bundles import preset_differences
    from scripts.akashic_release.doctor import read_environment

    checkout = Path(read_environment(args.runtime_env)["AKASHIC_RUNTIME_CHECKOUT"])
    groups: dict[str, list[dict[str, object]]] = {"changed": [], "added": [], "same": []}
    for item in preset_differences(ReleasePaths(args.root).state / "workspace", checkout):
        groups[item.kind].append({
            "plugin": item.plugin, "fields": list(item.fields),
            "preset": None if item.preset is None else {"disabled": item.preset.disabled, "config": dict(item.preset.config)},
            "user": {"disabled": item.user.disabled, "config": dict(item.user.config)},
        })
    missing = [item["plugin"] for item in groups["added"]
               if str(item["plugin"]).endswith("@release")
               and not (checkout / "plugins" / str(item["plugin"]).split("@")[0]).is_dir()]
    return {"status": "ok", "preset": str(checkout / "bundles"), **groups, "notShipped": missing}


def restore_state(args: argparse.Namespace) -> dict[str, object]:
    from scripts.akashic_release.activate import restore_state as restore

    paths = ReleasePaths(args.root.resolve(strict=True))
    with release_lock(paths.run / "release.lock"):
        restore(paths=paths, attempt_path=args.attempt)
    return {"status": "restored", "attempt": str(args.attempt)}


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
    install_parser.add_argument("--yes", action="store_true", help="不确认直接部署；首次安装时也不提问（等于 --profile full）")
    install_parser.add_argument("--no-activate", action="store_true")
    install_parser.add_argument("--plan", type=Path, help="显式外置插件目标；省略时更新 Core/Bridge 与内置 preset，并先执行必要内置迁移")
    install_parser.add_argument("--inputs", type=Path, help="清单引用的 bundle/wheels 目录")
    install_parser.add_argument("--backup", action="store_true", help="停止期备份 state 和 runtime.env；默认不备份")
    install_parser.add_argument("--no-extra", action="store_true", help="不在停机前联网求 extra 插件（首装、换来源、升级）")
    install_parser.add_argument("--extra-repo", help="extra 仓库地址，默认取 AKASHIC_EXTRA_REPO 或内置常量")
    install_parser.add_argument("--extra-inputs", type=Path,
                                help="预先下载好的 extra 输入目录（此前联网部署留下的 .inputs 目录），不联网")
    install_parser.add_argument("--allow-retire", action="store_true",
                                help="确认让在用的内置插件随新发行版消失（发行版不再提供且 extra 上游也没有）；默认拒绝，自部署不传")
    install_parser.add_argument("--profile", choices=("full", "minimal", "custom"), help="首次安装的扩展包选择；不给且有终端就提问，没有终端（或 --yes）只装核心，扩展包在网页引导里挑选")
    install_parser.add_argument("--groups", default="", help="首次安装自选的包组（逗号分隔，隐含 custom）")
    install_parser.add_argument("--reselect", action="store_true", help="首次安装失败后重跑时重新选择（默认沿用上次的选择，不再提问）")
    install_parser.set_defaults(handler=install)

    doctor_parser = subparsers.add_parser("doctor")
    doctor_parser.add_argument("--runtime-env", type=Path, default=_DEFAULT_ENV)
    doctor_parser.set_defaults(handler=doctor)

    resume_parser = subparsers.add_parser("resume", help="只重试已发布版本的启动和验收")
    resume_parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    resume_parser.add_argument("--runtime-env", type=Path, default=_DEFAULT_ENV)
    resume_parser.add_argument("--mise", type=Path, default=_DEFAULT_MISE)
    resume_parser.add_argument("--attempt", type=Path, required=True)
    resume_parser.add_argument("--unit-root", type=Path, default=Path("/etc/systemd/system"))
    resume_parser.add_argument("--cli-path", type=Path, default=Path.home() / ".local/bin/akashic-release")
    resume_parser.set_defaults(handler=resume)

    restore_parser = subparsers.add_parser("restore-state", help="按部署回执里的恢复点还原选择、patch 与插件 cache 指针（先停服务）")
    restore_parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    restore_parser.add_argument("--attempt", type=Path, required=True, help="部署回执（deploy-*.json），含恢复点与发布结果")
    restore_parser.set_defaults(handler=restore_state)

    migrate_parser = subparsers.add_parser("migrate")
    migrate_parser.add_argument("--snapshot-manifest", type=Path, required=True)
    migrate_parser.set_defaults(
        handler=lambda args: migration_plan(args.snapshot_manifest)
    )
    from scripts.akashic_release.self_deploy import submit, status, worker
    submit_parser = subparsers.add_parser("submit", help="在本次 Shell 回合正常结束并送达后部署")
    submit_parser.add_argument("--commit", required=True)
    submit_parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    submit_parser.add_argument("--runtime-env", type=Path, default=_DEFAULT_ENV)
    submit_parser.add_argument("--mise", type=Path, default=_DEFAULT_MISE)
    submit_parser.add_argument("--timeout", type=int, default=600)
    submit_parser.add_argument("--backup", action="store_true")
    submit_parser.add_argument("--session", help="部署失败时注入并唤醒的会话；默认是发起调用的会话")
    submit_parser.set_defaults(handler=submit)
    diff_parser = subparsers.add_parser("preset-diff", help="只读：列出本实例相对 preset 的启停与配置覆盖")
    diff_parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    diff_parser.add_argument("--runtime-env", type=Path, default=_DEFAULT_ENV)
    diff_parser.set_defaults(handler=preset_diff)
    status_parser = subparsers.add_parser("status", help="查询宿主部署任务，不启动模型回合")
    status_parser.add_argument("request_id")
    status_parser.add_argument("--root", type=Path, default=_DEFAULT_ROOT)
    status_parser.set_defaults(handler=status)
    worker_parser = subparsers.add_parser("self-deploy-worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--request", type=Path, required=True)
    worker_parser.add_argument("--prepared", action="store_true", help=argparse.SUPPRESS)
    worker_parser.set_defaults(handler=worker)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    # 与 start 同一条规则：离线输入已经决定了要装的包，不能再给 --profile/--groups。
    if getattr(args, "extra_inputs", None) and (getattr(args, "profile", None) or getattr(args, "groups", "")):
        parser.error("--extra-inputs 是预先下载好的输入目录，已经决定了要装的包，不能同时给 --profile/--groups")
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
