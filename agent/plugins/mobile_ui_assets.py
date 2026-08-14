from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class MobileUiAsset:
    module: str
    module_sha256: str
    module_bytes: int
    stylesheet: str
    stylesheet_sha256: str | None
    stylesheet_bytes: int
    navigation_label: str | None
    navigation_description: str | None
    slots: tuple[str, ...]


class MobileUiQueryHandler(Protocol):
    def __call__(
        self,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> object: ...


class MobileUiRpcInvalidRequest(ValueError):
    pass


MOBILE_UI_SLOTS = frozenset(
    {
        "turn.before_reasoning",
        "turn.before_tool",
        "turn.after_answer",
        "drawer.panel",
    }
)


def resolve_mobile_ui_asset(
    plugin_dir: Path,
    *,
    module: str,
    stylesheet: str | None,
    navigation_label: str | None,
    navigation_description: str | None,
    slots: tuple[str, ...],
) -> MobileUiAsset:
    """校验并固化一组插件自有的 Mobile UI 静态资产。"""

    # 1. 在插件声明边界校验 metadata 与允许的客户端插槽
    _validate_mobile_ui_metadata(
        module=module,
        stylesheet=stylesheet,
        navigation_label=navigation_label,
        navigation_description=navigation_description,
        slots=slots,
    )

    # 2. 解析 symlink 后限制资产只能来自插件 source
    root = plugin_dir.resolve(strict=True)
    module_path = _resolve_asset_path(root, module, suffix=".js", kind="module")
    stylesheet_path = (
        None
        if stylesheet is None
        else _resolve_asset_path(root, stylesheet, suffix=".css", kind="stylesheet")
    )
    return _build_mobile_ui_asset(
        module_path,
        stylesheet_path,
        navigation_label=navigation_label,
        navigation_description=navigation_description,
        slots=slots,
    )


def _validate_mobile_ui_metadata(
    *,
    module: str,
    stylesheet: str | None,
    navigation_label: str | None,
    navigation_description: str | None,
    slots: tuple[str, ...],
) -> None:
    """校验 Mobile UI 声明中的路径类型、导航和插槽。"""

    if not isinstance(module, str) or not module:
        raise RuntimeError("插件 mobile UI module 必须是非空相对路径")
    if stylesheet is not None and not isinstance(stylesheet, str):
        raise RuntimeError("插件 mobile UI stylesheet 必须是相对路径")
    if (
        (navigation_label is None) != (navigation_description is None)
        or navigation_label is not None
        and (
            not isinstance(navigation_label, str)
            or not navigation_label.strip()
            or len(navigation_label) > 64
            or not isinstance(navigation_description, str)
            or not navigation_description.strip()
            or len(navigation_description) > 160
        )
    ):
        raise RuntimeError("插件 mobile UI navigation 无效")
    if (
        not isinstance(slots, tuple)
        or len(set(slots)) != len(slots)
        or any(
            not isinstance(slot, str) or slot not in MOBILE_UI_SLOTS
            for slot in slots
        )
    ):
        raise RuntimeError("插件 mobile UI slots 无效")


def _build_mobile_ui_asset(
    module_path: Path,
    stylesheet_path: Path | None,
    *,
    navigation_label: str | None,
    navigation_description: str | None,
    slots: tuple[str, ...],
) -> MobileUiAsset:
    """读取并按内容摘要固化一组已验证的 Mobile UI 文件。"""

    module_content = module_path.read_text(encoding="utf-8")
    stylesheet_content = (
        "" if stylesheet_path is None else stylesheet_path.read_text(encoding="utf-8")
    )
    module_encoded = module_content.encode("utf-8")
    stylesheet_encoded = stylesheet_content.encode("utf-8")
    if len(module_encoded) + len(stylesheet_encoded) > 240 * 1024:
        raise RuntimeError("插件 mobile UI 资产超过协议安全预算")
    return MobileUiAsset(
        module=module_content,
        module_sha256=hashlib.sha256(module_encoded).hexdigest(),
        module_bytes=len(module_encoded),
        stylesheet=stylesheet_content,
        stylesheet_sha256=(
            hashlib.sha256(stylesheet_encoded).hexdigest()
            if stylesheet_content
            else None
        ),
        stylesheet_bytes=len(stylesheet_encoded),
        navigation_label=(
            None if navigation_label is None else navigation_label.strip()
        ),
        navigation_description=(
            None
            if navigation_description is None
            else navigation_description.strip()
        ),
        slots=slots,
    )


def _resolve_asset_path(
    plugin_root: Path,
    relative_path: str,
    *,
    suffix: str,
    kind: str,
) -> Path:
    """解析单个 Mobile UI 资产并拒绝 source 越界。"""

    if (
        not relative_path
        or relative_path != relative_path.strip()
        or Path(relative_path).is_absolute()
    ):
        raise RuntimeError(f"插件 mobile UI {kind} 无效: {relative_path}")
    try:
        path = (plugin_root / relative_path).resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"插件 mobile UI {kind} 无效: {relative_path}"
        ) from error
    if (
        not path.is_relative_to(plugin_root)
        or path.suffix != suffix
        or not path.is_file()
    ):
        raise RuntimeError(f"插件 mobile UI {kind} 无效: {relative_path}")
    return path
