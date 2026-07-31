from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from benchmark.harbor_v4flash.isolation import sha256_file
from benchmark.harbor_v4flash.git_volume import GIT_MOUNT_PATH

RUNTIME_VOLUME_PREFIX = "akasic-bench-runtime-v1-"
RUNTIME_VOLUME_SCHEMA = "akasic.benchmark-runtime.v1"
RUNTIME_MOUNT_PATH = "/opt/akashic-runtime"
RUNTIME_VENV_PATH = f"{RUNTIME_MOUNT_PATH}/venv"
RUNTIME_UV_PATH = f"{RUNTIME_MOUNT_PATH}/uv"
DEFAULT_PYTHON_VERSION = "3.13.7"
DEFAULT_BUILDER_IMAGE = "debian:bullseye-slim"
MAX_BUILDER_GLIBC_VERSION = (2, 31)
RUNTIME_TOP_LEVEL = ("manifest.json", "python", "resolved.lock", "uv", "venv")
RUNTIME_BUILD_RECIPE = {
    "id": "uv-managed-python-relocatable-venv-v3",
    "python_install": "uv-python-install-no-cache",
    "venv": "uv-venv-relocatable",
    "dependency_install": (
        "uv-pip-sync-manylinux-2.28-require-hashes-strict-no-cache"
    ),
    "builder_glibc_max": "2.31",
    "builder_root": "read-only",
    "builder_cache": "ephemeral-tmpfs",
}
_VOLUME_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_LABEL_PREFIX = "akasic.benchmark.runtime"


class RuntimeVolumeError(RuntimeError):
    pass


def _canonical_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _run(
    command: list[str],
    *,
    text: bool = True,
) -> subprocess.CompletedProcess[Any]:
    result = subprocess.run(
        command,
        check=False,
        text=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        stdout = result.stdout if isinstance(result.stdout, str) else ""
        stderr = result.stderr if isinstance(result.stderr, str) else ""
        output = "\n".join((stdout, stderr)).strip()
        raise RuntimeVolumeError(
            f"命令失败，exit={result.returncode}：{' '.join(command[:4])}\n"
            f"{output[-4000:]}"
        )
    return result


def _docker_platform() -> str:
    result = _run(
        ["docker", "version", "--format", "{{.Server.Os}}/{{.Server.Arch}}"]
    )
    platform = result.stdout.strip()
    if platform not in {"linux/amd64", "linux/arm64"}:
        raise RuntimeVolumeError(f"runtime volume 不支持 Docker 平台：{platform}")
    return platform


def _resolver_platform(platform: str) -> str:
    return {
        "linux/amd64": "x86_64-manylinux_2_28",
        "linux/arm64": "aarch64-manylinux_2_28",
    }[platform]


def _glibc_version(raw: str) -> tuple[int, int]:
    match = re.fullmatch(r"glibc ([0-9]+)\.([0-9]+)", raw)
    if match is None:
        raise RuntimeVolumeError(f"无法识别 builder glibc 版本：{raw!r}")
    return int(match.group(1)), int(match.group(2))


def _uv_identity(uv_binary: Path) -> dict[str, str]:
    uv_binary = uv_binary.resolve()
    if not uv_binary.is_file():
        raise FileNotFoundError(f"uv binary 不存在：{uv_binary}")
    version = _run([str(uv_binary), "--version"]).stdout.strip()
    if not version.startswith("uv "):
        raise RuntimeVolumeError(f"无法识别 uv 版本：{version!r}")
    return {
        "version": version,
        "digest": sha256_file(uv_binary),
    }


def _requirements_identity(source_root: Path) -> dict[str, object]:
    requirements_path = source_root.resolve() / "requirements.txt"
    if not requirements_path.is_file():
        raise FileNotFoundError(f"requirements.txt 不存在：{requirements_path}")
    base_digest = sha256_file(requirements_path)
    extras = ["tzdata"]
    return {
        "source": "requirements.txt",
        "source_digest": base_digest,
        "extras": extras,
        "digest": _canonical_digest(
            {
                "source_digest": base_digest,
                "extras": extras,
            }
        ),
    }


def _resolve_lock(
    *,
    source_root: Path,
    uv_binary: Path,
    python_version: str,
    resolver_platform: str,
    output_path: Path,
) -> None:
    """把声明依赖解析为稳定、带 distribution hash 的 runtime lock。"""

    # 1. 额外依赖独立输入，避免改写项目 requirements.txt。
    extras_path = output_path.with_name("runtime-extras.in")
    extras_path.write_text("tzdata\n", encoding="utf-8")

    # 2. 不使用持久 uv cache，输出不含临时路径或生成命令。
    _run(
        [
            str(uv_binary.resolve()),
            "pip",
            "compile",
            str(source_root.resolve() / "requirements.txt"),
            str(extras_path),
            "--python-version",
            python_version,
            "--python-platform",
            resolver_platform,
            "--generate-hashes",
            "--no-annotate",
            "--no-header",
            "--no-cache",
            "--output-file",
            str(output_path),
        ]
    )
    if not output_path.is_file() or not output_path.read_bytes():
        raise RuntimeVolumeError("uv 没有生成 resolved.lock")


def _builder_image_identity(reference: str) -> dict[str, object]:
    """冻结 builder image，并记录决定原生依赖兼容性的 glibc。"""

    # 1. 不可变 image ID 直接复用；浮动引用仍先拉取再冻结。
    if not reference.startswith("sha256:"):
        _run(["docker", "pull", reference])

    # 2. Docker image ID 和平台进入 recipe，tag 漂移会生成新 volume。
    inspected = _run(["docker", "image", "inspect", reference]).stdout
    payload: object = json.loads(inspected)
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeVolumeError("docker image inspect 必须返回一个 builder image")
    raw = payload[0]
    if not isinstance(raw, dict):
        raise RuntimeVolumeError("builder image inspect 元素必须是对象")
    image: dict[str, Any] = raw
    image_id = str(image.get("Id") or "")
    image_platform = f"{image.get('Os')}/{image.get('Architecture')}"
    if not image_id.startswith("sha256:"):
        raise RuntimeVolumeError("builder image 缺少不可变 image ID")
    glibc = _run(
        [
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--network",
            "none",
            "--entrypoint",
            "/usr/bin/getconf",
            image_id,
            "GNU_LIBC_VERSION",
        ]
    ).stdout.strip()
    _glibc_version(glibc)
    return {
        "reference": reference,
        "id": image_id,
        "repo_digests": sorted(str(value) for value in image.get("RepoDigests") or []),
        "platform": image_platform,
        "libc": glibc,
    }


def create_runtime_manifest(
    *,
    requirements: dict[str, object],
    uv: dict[str, str],
    python_version: str,
    platform: str,
    resolver_platform: str,
    builder_image: dict[str, object],
    resolved_lock_digest: str,
) -> dict[str, object]:
    """生成决定 volume 名称和 labels 的确定性 runtime manifest。"""

    # 1. recipe 只包含可复算身份，不包含时间、主机路径或 secret。
    recipe = {
        "build": RUNTIME_BUILD_RECIPE,
        "requirements": requirements,
        "uv": uv,
        "python": {
            "implementation": "cpython",
            "version": python_version,
        },
        "platform": {
            "docker": platform,
            "resolver": resolver_platform,
        },
        "builder_image": builder_image,
        "resolved_lock": {
            "path": "resolved.lock",
            "digest": resolved_lock_digest,
            "format": "requirements.txt-with-hashes",
        },
    }
    recipe_digest = _canonical_digest(recipe)
    runtime_digest = _canonical_digest(
        {
            "schema": RUNTIME_VOLUME_SCHEMA,
            "recipe_digest": recipe_digest,
        }
    )
    volume_name = (
        RUNTIME_VOLUME_PREFIX
        + runtime_digest.removeprefix("sha256:")[:24]
    )

    # 2. manifest 自身另有摘要，trial 可同时校验 recipe 与文件内容。
    base = {
        "schema": RUNTIME_VOLUME_SCHEMA,
        "volume_name": volume_name,
        "runtime_digest": runtime_digest,
        "recipe_digest": recipe_digest,
        "recipe": recipe,
        "contents": {
            "mount_path": RUNTIME_MOUNT_PATH,
            "venv_path": "venv",
            "top_level": list(RUNTIME_TOP_LEVEL),
            "contains_source": False,
            "contains_home": False,
            "contains_workspace": False,
            "contains_task_data": False,
            "contains_secrets": False,
            "contains_uv_cache": False,
            "contains_verifier_uv": True,
        },
    }
    return {
        **base,
        "manifest_digest": _canonical_digest(base),
    }


def runtime_volume_labels(manifest: dict[str, object]) -> dict[str, str]:
    recipe = manifest["recipe"]
    if not isinstance(recipe, dict):
        raise RuntimeVolumeError("runtime manifest recipe 必须是对象")
    requirements = recipe["requirements"]
    uv = recipe["uv"]
    python = recipe["python"]
    platform = recipe["platform"]
    lock = recipe["resolved_lock"]
    builder = recipe["builder_image"]
    if not all(
        isinstance(value, dict)
        for value in (requirements, uv, python, platform, lock, builder)
    ):
        raise RuntimeVolumeError("runtime manifest recipe 字段必须是对象")
    builder_glibc = builder.get("libc")
    if not isinstance(builder_glibc, str):
        raise RuntimeVolumeError("runtime manifest builder glibc 缺失")
    return {
        f"{_LABEL_PREFIX}.managed": "true",
        f"{_LABEL_PREFIX}.schema": str(manifest["schema"]),
        f"{_LABEL_PREFIX}.volume_name": str(manifest["volume_name"]),
        f"{_LABEL_PREFIX}.digest": str(manifest["runtime_digest"]),
        f"{_LABEL_PREFIX}.manifest_digest": str(manifest["manifest_digest"]),
        f"{_LABEL_PREFIX}.recipe_digest": str(manifest["recipe_digest"]),
        f"{_LABEL_PREFIX}.requirements_digest": str(requirements["digest"]),
        f"{_LABEL_PREFIX}.uv_digest": str(uv["digest"]),
        f"{_LABEL_PREFIX}.uv_version": str(uv["version"]),
        f"{_LABEL_PREFIX}.python": str(python["version"]),
        f"{_LABEL_PREFIX}.platform": str(platform["docker"]),
        f"{_LABEL_PREFIX}.resolved_lock_digest": str(lock["digest"]),
        f"{_LABEL_PREFIX}.builder_image_id": str(builder["id"]),
        f"{_LABEL_PREFIX}.builder_glibc": builder_glibc,
    }


def _inspect_volume(volume_name: str) -> dict[str, object]:
    if not _VOLUME_NAME_PATTERN.fullmatch(volume_name):
        raise RuntimeVolumeError(f"Docker volume 名称不合法：{volume_name!r}")
    result = _run(["docker", "volume", "inspect", volume_name]).stdout
    payload: object = json.loads(result)
    if not isinstance(payload, list) or len(payload) != 1:
        raise RuntimeVolumeError("docker volume inspect 必须返回一个 volume")
    raw = payload[0]
    if not isinstance(raw, dict):
        raise RuntimeVolumeError("docker volume inspect 元素必须是对象")
    return raw


def _read_volume_file(
    *,
    volume_name: str,
    image_id: str,
    relative_path: str,
) -> bytes:
    result = _run(
        [
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--network",
            "none",
            "--mount",
            (
                f"type=volume,src={volume_name},"
                f"dst={RUNTIME_MOUNT_PATH},readonly"
            ),
            "--entrypoint",
            "/bin/cat",
            image_id,
            f"{RUNTIME_MOUNT_PATH}/{relative_path}",
        ],
        text=False,
    )
    return bytes(result.stdout)


def _volume_top_level(volume_name: str, image_id: str) -> list[str]:
    result = _run(
        [
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--network",
            "none",
            "--mount",
            (
                f"type=volume,src={volume_name},"
                f"dst={RUNTIME_MOUNT_PATH},readonly"
            ),
            "--entrypoint",
            "/bin/sh",
            image_id,
            "-c",
            (
                f"find {RUNTIME_MOUNT_PATH} -mindepth 1 -maxdepth 1 "
                "-printf '%f\\n' | LC_ALL=C sort"
            ),
        ]
    )
    return [line for line in result.stdout.splitlines() if line]


def inspect_runtime_volume(
    volume_name: str,
    *,
    source_root: Path,
    uv_binary: Path,
) -> dict[str, object]:
    """校验 cache identity、只读内容清单和当前 benchmark 输入。"""

    # 1. Docker labels 决定用哪个不可变 builder image 只读检查 volume。
    volume = _inspect_volume(volume_name)
    raw_labels = volume.get("Labels")
    if not isinstance(raw_labels, dict):
        raise RuntimeVolumeError("runtime volume 缺少 labels")
    labels = {str(key): str(value) for key, value in raw_labels.items()}
    image_id = labels.get(f"{_LABEL_PREFIX}.builder_image_id", "")
    if not image_id.startswith("sha256:"):
        raise RuntimeVolumeError("runtime volume 缺少 builder image ID label")
    manifest_bytes = _read_volume_file(
        volume_name=volume_name,
        image_id=image_id,
        relative_path="manifest.json",
    )
    raw_manifest: object = json.loads(manifest_bytes)
    if not isinstance(raw_manifest, dict):
        raise RuntimeVolumeError("runtime manifest 必须是对象")
    manifest: dict[str, object] = raw_manifest

    # 2. 复算 manifest、recipe、lock 和完整目录 allowlist。
    if manifest.get("schema") != RUNTIME_VOLUME_SCHEMA:
        raise RuntimeVolumeError("runtime manifest schema 不匹配")
    without_digest = {
        key: value for key, value in manifest.items() if key != "manifest_digest"
    }
    if manifest.get("manifest_digest") != _canonical_digest(without_digest):
        raise RuntimeVolumeError("runtime manifest 摘要不匹配")
    recipe = manifest.get("recipe")
    if not isinstance(recipe, dict):
        raise RuntimeVolumeError("runtime manifest recipe 必须是对象")
    if manifest.get("recipe_digest") != _canonical_digest(recipe):
        raise RuntimeVolumeError("runtime recipe 摘要不匹配")
    runtime_digest = _canonical_digest(
        {
            "schema": RUNTIME_VOLUME_SCHEMA,
            "recipe_digest": manifest["recipe_digest"],
        }
    )
    if manifest.get("runtime_digest") != runtime_digest:
        raise RuntimeVolumeError("runtime digest 不匹配")
    expected_name = (
        RUNTIME_VOLUME_PREFIX + runtime_digest.removeprefix("sha256:")[:24]
    )
    if manifest.get("volume_name") != volume_name or volume_name != expected_name:
        raise RuntimeVolumeError("runtime volume 名称与 digest 不匹配")
    if labels != runtime_volume_labels(manifest):
        raise RuntimeVolumeError("runtime volume labels 与 manifest 不匹配")
    lock_bytes = _read_volume_file(
        volume_name=volume_name,
        image_id=image_id,
        relative_path="resolved.lock",
    )
    lock_digest = f"sha256:{hashlib.sha256(lock_bytes).hexdigest()}"
    resolved_lock = recipe.get("resolved_lock")
    if not isinstance(resolved_lock, dict) or resolved_lock.get("digest") != lock_digest:
        raise RuntimeVolumeError("runtime resolved.lock 摘要不匹配")
    if _volume_top_level(volume_name, image_id) != list(RUNTIME_TOP_LEVEL):
        raise RuntimeVolumeError("runtime volume 含 allowlist 外内容")

    # 3. 当前源码依赖、uv、Python 与 Docker 平台必须命中同一 recipe。
    platform = _docker_platform()
    if recipe.get("requirements") != _requirements_identity(source_root):
        raise RuntimeVolumeError("runtime volume requirements 与当前源码不匹配")
    if recipe.get("build") != RUNTIME_BUILD_RECIPE:
        raise RuntimeVolumeError("runtime volume builder recipe 不匹配")
    if recipe.get("uv") != _uv_identity(uv_binary):
        raise RuntimeVolumeError("runtime volume uv 与当前 builder 输入不匹配")
    python = recipe.get("python")
    if not isinstance(python, dict) or python.get("version") != DEFAULT_PYTHON_VERSION:
        raise RuntimeVolumeError("runtime volume Python 版本不匹配")
    recipe_platform = recipe.get("platform")
    if not isinstance(recipe_platform, dict) or recipe_platform != {
        "docker": platform,
        "resolver": _resolver_platform(platform),
    }:
        raise RuntimeVolumeError("runtime volume 平台不匹配")
    builder = recipe.get("builder_image")
    if not isinstance(builder, dict):
        raise RuntimeVolumeError("runtime volume builder image 缺失")
    builder_glibc = _glibc_version(str(builder.get("libc") or ""))
    if builder_glibc > MAX_BUILDER_GLIBC_VERSION:
        raise RuntimeVolumeError("runtime volume builder glibc 高于兼容上限 2.31")

    return {
        "name": volume_name,
        "driver": volume.get("Driver"),
        "scope": volume.get("Scope"),
        "created_at": volume.get("CreatedAt"),
        "labels": labels,
        "manifest": manifest,
    }


def runtime_compose_overlay(
    volume_name: str,
    *,
    task_image_id: str | None = None,
    git_volume_name: str | None = None,
) -> dict[str, object]:
    """生成 Harbor main service 的冻结 image 与 runtime 只读挂载。"""

    if not _VOLUME_NAME_PATTERN.fullmatch(volume_name):
        raise RuntimeVolumeError(f"Docker volume 名称不合法：{volume_name!r}")
    if task_image_id is not None and not task_image_id.startswith("sha256:"):
        raise RuntimeVolumeError(f"task image ID 不合法：{task_image_id!r}")
    if git_volume_name is not None and not _VOLUME_NAME_PATTERN.fullmatch(
        git_volume_name
    ):
        raise RuntimeVolumeError(f"Git volume 名称不合法：{git_volume_name!r}")
    service_volumes: list[dict[str, object]] = [
        {
            "type": "volume",
            "source": "akasic_runtime",
            "target": RUNTIME_MOUNT_PATH,
            "read_only": True,
        }
    ]
    service: dict[str, object] = {"volumes": service_volumes}
    if task_image_id is not None:
        service["image"] = task_image_id
        service["pull_policy"] = "never"
    volumes: dict[str, object] = {
        "akasic_runtime": {
            "external": True,
            "name": volume_name,
        }
    }
    if git_volume_name is not None:
        service_volumes.append(
            {
                "type": "volume",
                "source": "akasic_git",
                "target": GIT_MOUNT_PATH,
                "read_only": True,
            }
        )
        volumes["akasic_git"] = {
            "external": True,
            "name": git_volume_name,
        }
    return {
        "services": {
            "main": service,
        },
        "volumes": volumes,
    }


def build_runtime_volume(
    *,
    source_root: Path,
    uv_binary: Path,
    python_version: str,
    builder_image_reference: str,
) -> dict[str, object]:
    """显式解析依赖，并由唯一 builder 创建不可变 runtime volume。"""

    # 1. 冻结所有 recipe 输入，解析不依赖宿主 uv cache 的 lock。
    if python_version != DEFAULT_PYTHON_VERSION:
        raise RuntimeVolumeError(
            "benchmark runtime Python 已冻结为 "
            f"{DEFAULT_PYTHON_VERSION}，不接受 {python_version}"
        )
    source_root = source_root.resolve()
    uv_binary = uv_binary.resolve()
    platform = _docker_platform()
    resolver_platform = _resolver_platform(platform)
    requirements = _requirements_identity(source_root)
    uv = _uv_identity(uv_binary)
    builder_image = _builder_image_identity(builder_image_reference)
    if builder_image["platform"] != platform:
        raise RuntimeVolumeError(
            "builder image 平台不匹配："
            f"{builder_image['platform']} != {platform}"
        )
    if (
        _glibc_version(str(builder_image["libc"]))
        > MAX_BUILDER_GLIBC_VERSION
    ):
        raise RuntimeVolumeError("builder glibc 高于兼容上限 2.31")
    with tempfile.TemporaryDirectory(prefix="akasic-runtime-volume-") as raw_temp:
        temp_root = Path(raw_temp)
        lock_path = temp_root / "resolved.lock"
        _resolve_lock(
            source_root=source_root,
            uv_binary=uv_binary,
            python_version=python_version,
            resolver_platform=resolver_platform,
            output_path=lock_path,
        )
        manifest = create_runtime_manifest(
            requirements=requirements,
            uv=uv,
            python_version=python_version,
            platform=platform,
            resolver_platform=resolver_platform,
            builder_image=builder_image,
            resolved_lock_digest=sha256_file(lock_path),
        )
        volume_name = str(manifest["volume_name"])
        manifest_path = temp_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 2. 已存在 cache 只能校验复用；不覆盖身份相同但内容损坏的 volume。
        exists = subprocess.run(
            ["docker", "volume", "inspect", volume_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
        if exists:
            return inspect_runtime_volume(
                volume_name,
                source_root=source_root,
                uv_binary=uv_binary,
            )

        create_command = ["docker", "volume", "create"]
        for key, value in sorted(runtime_volume_labels(manifest).items()):
            create_command.extend(["--label", f"{key}={value}"])
        create_command.append(volume_name)
        created = _run(create_command).stdout.strip()
        if created != volume_name:
            raise RuntimeVolumeError(
                f"Docker 创建了非预期 volume：{created!r}"
            )

        # 3. builder 的根文件系统和临时 cache 均为一次性，只有目标 volume 可写。
        builder_script = (
            f"test -z \"$(find {RUNTIME_MOUNT_PATH} -mindepth 1 "
            "-maxdepth 1 -print -quit)\"\n"
            f"mkdir -p {RUNTIME_MOUNT_PATH}/python\n"
            f"/tools/uv python install \"$PYTHON_VERSION\" "
            f"--install-dir {RUNTIME_MOUNT_PATH}/python --no-cache --no-bin\n"
            f"set -- {RUNTIME_MOUNT_PATH}/python/"
            "cpython-${PYTHON_VERSION}-*/bin/python3.13\n"
            "test \"$#\" -eq 1 && test -x \"$1\"\n"
            f"/tools/uv venv --relocatable --python \"$1\" "
            f"{RUNTIME_VENV_PATH}\n"
            f"/tools/uv pip sync --python {RUNTIME_VENV_PATH}/bin/python "
            "/inputs/resolved.lock --python-platform \"$RESOLVER_PLATFORM\" "
            "--require-hashes --strict --no-cache\n"
            f"cp /tools/uv {RUNTIME_UV_PATH}\n"
            f"cp /inputs/resolved.lock {RUNTIME_MOUNT_PATH}/resolved.lock\n"
            f"cp /inputs/manifest.json {RUNTIME_MOUNT_PATH}/manifest.json\n"
            f"test \"$({RUNTIME_VENV_PATH}/bin/python -c "
            "'import platform; print(platform.python_version())')\" "
            "= \"$PYTHON_VERSION\"\n"
            f"chmod -R a-w {RUNTIME_MOUNT_PATH}\n"
        )
        try:
            _run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--read-only",
                    "--network",
                    "bridge",
                    "--tmpfs",
                    "/tmp:rw,exec,nosuid,nodev,size=8589934592",
                    "--mount",
                    (
                        f"type=volume,src={volume_name},"
                        f"dst={RUNTIME_MOUNT_PATH}"
                    ),
                    "--mount",
                    (
                        f"type=bind,src={uv_binary},"
                        "dst=/tools/uv,readonly"
                    ),
                    "--mount",
                    (
                        f"type=bind,src={lock_path},"
                        "dst=/inputs/resolved.lock,readonly"
                    ),
                    "--mount",
                    (
                        f"type=bind,src={manifest_path},"
                        "dst=/inputs/manifest.json,readonly"
                    ),
                    "--env",
                    "HOME=/tmp/home",
                    "--env",
                    "UV_CACHE_DIR=/tmp/uv-cache",
                    "--env",
                    f"PYTHON_VERSION={python_version}",
                    "--env",
                    f"RESOLVER_PLATFORM={resolver_platform}",
                    "--entrypoint",
                    "/bin/sh",
                    str(builder_image["id"]),
                    "-eu",
                    "-c",
                    builder_script,
                ]
            )
        except Exception as error:
            raise RuntimeVolumeError(
                f"runtime volume 构建失败，保留现场 {volume_name}；"
                "修复根因后需显式删除该 volume 再重建"
            ) from error

    return inspect_runtime_volume(
        volume_name,
        source_root=source_root,
        uv_binary=uv_binary,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "inspect"])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--uv-binary",
        type=Path,
        default=Path(
            os.environ.get("AKASIC_BENCH_UV", "/home/huashen/.local/bin/uv")
        ),
    )
    parser.add_argument("--volume")
    parser.add_argument("--builder-image", default=DEFAULT_BUILDER_IMAGE)
    args = parser.parse_args()

    if args.command == "build":
        report = build_runtime_volume(
            source_root=args.source_root,
            uv_binary=args.uv_binary,
            python_version=DEFAULT_PYTHON_VERSION,
            builder_image_reference=args.builder_image,
        )
    else:
        if not args.volume:
            parser.error("--volume 是 inspect 的必填参数")
        report = inspect_runtime_volume(
            args.volume,
            source_root=args.source_root,
            uv_binary=args.uv_binary,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
