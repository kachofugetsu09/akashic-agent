from __future__ import annotations

from pathlib import Path

from scripts.build_host_runtime_release import build_release


def prepare_core_image(
    *,
    checkout: Path,
    commit: str,
    manifest: Path,
    image_tag: str,
) -> dict[str, object]:
    """Build one exact Core image through the canonical release builder."""

    return build_release(
        repository=checkout,
        requested_commit=commit,
        image_tag=image_tag,
        output_manifest=manifest,
        base_image=(
            "archlinux@sha256:"
            "345a872f6c95e082d4b8c050af637eebb57402c6e2177b411c3acf7df84eb33b"
        ),
        arch_snapshot="2026/08/09",
    )
