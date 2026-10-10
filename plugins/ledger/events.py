from __future__ import annotations

from dataclasses import dataclass

from plugins.ledger.contract import AttachmentKind


@dataclass(frozen=True, slots=True)
class ChannelAttachment:
    """渠道边界中带明确类型的单个附件。"""

    kind: AttachmentKind
    source: str
    filename: str | None = None
