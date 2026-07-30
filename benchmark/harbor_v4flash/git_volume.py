from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import uuid
from typing import Any

GIT_VOLUME_PREFIX = "akasic-bench-git-v1-"
GIT_VOLUME_SCHEMA = "akasic.benchmark-git.v1"
GIT_MOUNT_PATH = "/opt/akashic-git"
GIT_BIN_PATH = f"{GIT_MOUNT_PATH}/bin/git"
DEFAULT_GIT_BUILDER_IMAGE = "debian:bullseye-slim"
GIT_TOP_LEVEL = ("bin", "manifest.json", "metadata.json", "root")
_LABEL_PREFIX = "akasic.benchmark.git"
_VOLUME_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class GitVolumeError(RuntimeError):
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
        raise GitVolumeError(
            f"命令失败，exit={result.returncode}：{' '.join(command[:4])}\n"
            f"{output[-4000:]}"
        )
    return result


def _builder_image_identity(reference: str) -> dict[str, object]:
    """复用或拉取 Git builder，并固定到不可变 image ID。"""

    inspected = subprocess.run(
        ["docker", "image", "inspect", reference],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if inspected.returncode != 0:
        _run(["docker", "pull", reference])
    payload: object = json.loads(
        _run(["docker", "image", "inspect", reference]).stdout
    )
    if not isinstance(payload, list) or len(payload) != 1:
        raise GitVolumeError("docker image inspect 必须返回一个 builder image")
    raw = payload[0]
    if not isinstance(raw, dict):
        raise GitVolumeError("builder image inspect 元素必须是对象")
    image: dict[str, Any] = raw
    image_id = str(image.get("Id") or "")
    platform = f"{image.get('Os')}/{image.get('Architecture')}"
    if not image_id.startswith("sha256:"):
        raise GitVolumeError("Git builder image 缺少不可变 ID")
    if platform not in {"linux/amd64", "linux/arm64"}:
        raise GitVolumeError(f"Git builder 平台不受支持：{platform}")
    return {
        "reference": reference,
        "id": image_id,
        "repo_digests": sorted(
            str(value) for value in image.get("RepoDigests") or []
        ),
        "platform": platform,
    }


def create_git_manifest(
    *,
    builder_image: dict[str, object],
    metadata: dict[str, object],
    content_digest: str,
) -> dict[str, object]:
    """生成由 builder、包版本和实际内容共同决定的 Git manifest。"""

    recipe = {
        "id": "debian-portable-git-volume-v1",
        "builder_image": builder_image,
        "packages": metadata,
        "content_digest": content_digest,
    }
    recipe_digest = _canonical_digest(recipe)
    runtime_digest = _canonical_digest(
        {
            "schema": GIT_VOLUME_SCHEMA,
            "recipe_digest": recipe_digest,
        }
    )
    volume_name = GIT_VOLUME_PREFIX + runtime_digest.removeprefix("sha256:")[:24]
    base = {
        "schema": GIT_VOLUME_SCHEMA,
        "volume_name": volume_name,
        "runtime_digest": runtime_digest,
        "recipe_digest": recipe_digest,
        "recipe": recipe,
        "contents": {
            "mount_path": GIT_MOUNT_PATH,
            "git_path": "bin/git",
            "top_level": list(GIT_TOP_LEVEL),
            "contains_source": False,
            "contains_workspace": False,
            "contains_task_data": False,
            "contains_secrets": False,
        },
    }
    return {**base, "manifest_digest": _canonical_digest(base)}


def git_volume_labels(manifest: dict[str, object]) -> dict[str, str]:
    recipe = manifest.get("recipe")
    if not isinstance(recipe, dict):
        raise GitVolumeError("Git manifest recipe 必须是对象")
    builder = recipe.get("builder_image")
    packages = recipe.get("packages")
    if not isinstance(builder, dict) or not isinstance(packages, dict):
        raise GitVolumeError("Git manifest builder/packages 必须是对象")
    return {
        f"{_LABEL_PREFIX}.managed": "true",
        f"{_LABEL_PREFIX}.schema": str(manifest["schema"]),
        f"{_LABEL_PREFIX}.volume_name": str(manifest["volume_name"]),
        f"{_LABEL_PREFIX}.digest": str(manifest["runtime_digest"]),
        f"{_LABEL_PREFIX}.manifest_digest": str(manifest["manifest_digest"]),
        f"{_LABEL_PREFIX}.recipe_digest": str(manifest["recipe_digest"]),
        f"{_LABEL_PREFIX}.content_digest": str(recipe["content_digest"]),
        f"{_LABEL_PREFIX}.git_version": str(packages["git_version"]),
        f"{_LABEL_PREFIX}.builder_image_id": str(builder["id"]),
        f"{_LABEL_PREFIX}.platform": str(builder["platform"]),
    }


def _volume_file(
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
            f"type=volume,src={volume_name},dst={GIT_MOUNT_PATH},readonly",
            "--entrypoint",
            "/bin/cat",
            image_id,
            f"{GIT_MOUNT_PATH}/{relative_path}",
        ],
        text=False,
    )
    return bytes(result.stdout)


def _volume_projection(
    volume_name: str,
    image_id: str,
) -> tuple[list[str], str]:
    script = (
        f"top=$(find {GIT_MOUNT_PATH} -mindepth 1 -maxdepth 1 "
        "-printf '%f\\n' | LC_ALL=C sort); "
        f"digest=$(tar --sort=name --mtime=@0 --owner=0 --group=0 "
        f"--numeric-owner -cf - -C {GIT_MOUNT_PATH} bin metadata.json root "
        "| sha256sum | cut -d' ' -f1); "
        "printf '%s\\n--DIGEST--\\nsha256:%s\\n' \"$top\" \"$digest\""
    )
    output = _run(
        [
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--network",
            "none",
            "--mount",
            f"type=volume,src={volume_name},dst={GIT_MOUNT_PATH},readonly",
            "--entrypoint",
            "/bin/sh",
            image_id,
            "-eu",
            "-c",
            script,
        ]
    ).stdout
    top, digest = output.split("--DIGEST--\n", maxsplit=1)
    return [line for line in top.splitlines() if line], digest.strip()


def inspect_git_volume(volume_name: str) -> dict[str, object]:
    """校验 Git volume 的 labels、manifest、内容摘要和可执行入口。"""

    if not _VOLUME_NAME_PATTERN.fullmatch(volume_name):
        raise GitVolumeError(f"Docker volume 名称不合法：{volume_name!r}")
    payload: object = json.loads(
        _run(["docker", "volume", "inspect", volume_name]).stdout
    )
    if not isinstance(payload, list) or len(payload) != 1:
        raise GitVolumeError("docker volume inspect 必须返回一个 volume")
    raw = payload[0]
    if not isinstance(raw, dict):
        raise GitVolumeError("docker volume inspect 元素必须是对象")
    volume: dict[str, Any] = raw
    raw_labels = volume.get("Labels")
    if not isinstance(raw_labels, dict):
        raise GitVolumeError("Git volume 缺少 labels")
    labels = {str(key): str(value) for key, value in raw_labels.items()}
    image_id = labels.get(f"{_LABEL_PREFIX}.builder_image_id", "")
    if not image_id.startswith("sha256:"):
        raise GitVolumeError("Git volume 缺少 builder image ID label")
    manifest: object = json.loads(
        _volume_file(volume_name, image_id, "manifest.json")
    )
    if not isinstance(manifest, dict):
        raise GitVolumeError("Git manifest 必须是对象")
    without_digest = {
        key: value for key, value in manifest.items() if key != "manifest_digest"
    }
    if manifest.get("manifest_digest") != _canonical_digest(without_digest):
        raise GitVolumeError("Git manifest 摘要不匹配")
    recipe = manifest.get("recipe")
    if not isinstance(recipe, dict):
        raise GitVolumeError("Git manifest recipe 必须是对象")
    if manifest.get("recipe_digest") != _canonical_digest(recipe):
        raise GitVolumeError("Git recipe 摘要不匹配")
    expected_name = str(manifest.get("volume_name") or "")
    if expected_name != volume_name:
        raise GitVolumeError("Git volume 名称与 manifest 不匹配")
    if labels != git_volume_labels(manifest):
        raise GitVolumeError("Git volume labels 与 manifest 不匹配")
    top_level, content_digest = _volume_projection(volume_name, image_id)
    if top_level != list(GIT_TOP_LEVEL):
        raise GitVolumeError("Git volume 含 allowlist 外内容")
    if content_digest != recipe.get("content_digest"):
        raise GitVolumeError("Git volume 内容摘要不匹配")
    return {
        "name": volume_name,
        "driver": volume.get("Driver"),
        "scope": volume.get("Scope"),
        "created_at": volume.get("CreatedAt"),
        "labels": labels,
        "manifest": manifest,
    }


def _create_labeled_volume(
    volume_name: str,
    labels: dict[str, str],
) -> None:
    command = ["docker", "volume", "create"]
    for key, value in sorted(labels.items()):
        command.extend(["--label", f"{key}={value}"])
    command.append(volume_name)
    created = _run(command).stdout.strip()
    if created != volume_name:
        raise GitVolumeError(f"Docker 创建了非预期 volume：{created!r}")


def _find_reusable_git_volume(
    builder: dict[str, object],
) -> dict[str, object] | None:
    """复用同一 builder 已发布的有效 Git volume。"""

    # 1. Docker labels 是本机 cache 索引，不访问 apt 或 registry。
    image_id = str(builder["id"])
    result = _run(
        [
            "docker",
            "volume",
            "ls",
            "--filter",
            f"label={_LABEL_PREFIX}.managed=true",
            "--filter",
            f"label={_LABEL_PREFIX}.builder_image_id={image_id}",
            "--format",
            "{{.Name}}",
        ]
    )
    candidates = sorted(line for line in result.stdout.splitlines() if line)

    # 2. 只复用通过完整内容校验的 volume；损坏 cache 必须显式暴露。
    invalid: list[str] = []
    for volume_name in candidates:
        try:
            return inspect_git_volume(volume_name)
        except GitVolumeError:
            invalid.append(volume_name)
    if invalid:
        names = ", ".join(invalid)
        raise GitVolumeError(
            f"发现损坏的 Git volume cache：{names}；"
            "确认后删除，或使用 build --rebuild 重新发布"
        )
    return None


def build_git_volume(
    *,
    builder_image_reference: str = DEFAULT_GIT_BUILDER_IMAGE,
    reuse_existing: bool = True,
) -> dict[str, object]:
    """一次下载 Git/CA，发布跨 trial 只读复用的内容寻址 volume。"""

    # 1. 命中已校验的本地 cache 时，不再启动 apt。
    builder = _builder_image_identity(builder_image_reference)
    if reuse_existing:
        cached = _find_reusable_git_volume(builder)
        if cached is not None:
            return cached

    # 2. staging 保留真实包版本和依赖内容，最终身份不依赖浮动 apt 元数据。
    image_id = str(builder["id"])
    staging = f"akasic-bench-git-staging-{uuid.uuid4().hex}"
    _run(["docker", "volume", "create", staging])
    build_script = f"""
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates
mkdir -p {GIT_MOUNT_PATH}/bin {GIT_MOUNT_PATH}/root/usr/bin
mkdir -p {GIT_MOUNT_PATH}/root/usr/lib {GIT_MOUNT_PATH}/root/usr/share
mkdir -p {GIT_MOUNT_PATH}/root/etc/ssl/certs {GIT_MOUNT_PATH}/root/libdeps
cp -a /usr/bin/git {GIT_MOUNT_PATH}/root/usr/bin/git
cp -a /usr/lib/git-core {GIT_MOUNT_PATH}/root/usr/lib/git-core
cp -a /usr/share/git-core {GIT_MOUNT_PATH}/root/usr/share/git-core
cp -a /etc/ssl/certs/ca-certificates.crt {GIT_MOUNT_PATH}/root/etc/ssl/certs/
for executable in /usr/bin/git $(find /usr/lib/git-core -type f -perm /111); do
    ldd "$executable" 2>/dev/null || true
done | awk '/=> \\/|^\\// {{ for (i=1; i<=NF; i++) if ($i ~ "^/") print $i }}' \
    | LC_ALL=C sort -u | while read -r library; do
        case "$(basename "$library")" in
            libc.so.*|libpthread.so.*|libdl.so.*|librt.so.*|libm.so.*|libresolv.so.*|ld-linux-*) continue ;;
        esac
        cp -L "$library" "{GIT_MOUNT_PATH}/root/libdeps/$(basename "$library")"
    done
git_version=$(git --version)
git_package=$(dpkg-query -W -f='${{Version}}' git)
ca_package=$(dpkg-query -W -f='${{Version}}' ca-certificates)
printf '{{"git_version":"%s","git_package":"%s","ca_certificates_package":"%s"}}\\n' \
    "$git_version" "$git_package" "$ca_package" > {GIT_MOUNT_PATH}/metadata.json
cat > {GIT_BIN_PATH} <<'EOF'
#!/bin/sh
root=/opt/akashic-git/root
export GIT_EXEC_PATH="$root/usr/lib/git-core"
export GIT_TEMPLATE_DIR="$root/usr/share/git-core/templates"
export LD_LIBRARY_PATH="$root/libdeps"
export SSL_CERT_FILE="$root/etc/ssl/certs/ca-certificates.crt"
exec "$root/usr/bin/git" "$@"
EOF
chmod 0555 {GIT_BIN_PATH}
chmod -R a-w {GIT_MOUNT_PATH}
"""
    try:
        _run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "bridge",
                "--mount",
                f"type=volume,src={staging},dst={GIT_MOUNT_PATH}",
                "--entrypoint",
                "/bin/sh",
                image_id,
                "-eu",
                "-c",
                build_script,
            ]
        )
        metadata: object = json.loads(
            _volume_file(staging, image_id, "metadata.json")
        )
        if not isinstance(metadata, dict):
            raise GitVolumeError("Git metadata 必须是对象")
        _, content_digest = _volume_projection(staging, image_id)
        manifest = create_git_manifest(
            builder_image=builder,
            metadata=metadata,
            content_digest=content_digest,
        )
        volume_name = str(manifest["volume_name"])
        exists = subprocess.run(
            ["docker", "volume", "inspect", volume_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
        if not exists:
            _create_labeled_volume(volume_name, git_volume_labels(manifest))
            manifest_json = json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            )
            copy_script = f"""
cp -a /source/bin /target/bin
cp -a /source/metadata.json /target/metadata.json
cp -a /source/root /target/root
printf '%s\\n' "$MANIFEST_JSON" > /target/manifest.json
chmod -R a-w /target
"""
            _run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--read-only",
                    "--network",
                    "none",
                    "--mount",
                    f"type=volume,src={staging},dst=/source,readonly",
                    "--mount",
                    f"type=volume,src={volume_name},dst=/target",
                    "--env",
                    f"MANIFEST_JSON={manifest_json}",
                    "--entrypoint",
                    "/bin/sh",
                    image_id,
                    "-eu",
                    "-c",
                    copy_script,
                ]
            )
        return inspect_git_volume(volume_name)
    finally:
        _run(["docker", "volume", "rm", staging])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["build", "inspect"])
    parser.add_argument("--volume")
    parser.add_argument("--builder-image", default=DEFAULT_GIT_BUILDER_IMAGE)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        report = build_git_volume(
            builder_image_reference=args.builder_image,
            reuse_existing=not args.rebuild,
        )
    else:
        if not args.volume:
            parser.error("--volume 是 inspect 的必填参数")
        report = inspect_git_volume(args.volume)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
