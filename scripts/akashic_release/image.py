from __future__ import annotations

from pathlib import Path

from scripts.akashic_release.manifest import write_json
from scripts.build_host_runtime_release import build_distribution_release
from scripts.host_toolchain_identity import declared_toolchain_identity


def prepare_core_image(
    *,
    checkout: Path,
    commit: str,
    manifest: Path,
    image_tag: str,
) -> dict[str, object]:
    """Build the formal Core distribution and preserve Bridge identity metadata."""

    result = build_distribution_release(
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
    # The public release transaction still prepares a host Bridge checkout.
    # Keep its identity in the distribution manifest without making the image
    # builder fall back to shipping that checkout as Core source.
    result["hostToolchainIdentity"] = declared_toolchain_identity(
        commit, (checkout / "mise.toml").read_bytes()
    )
    write_json(manifest, result)
    return result
