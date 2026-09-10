"""兼容入口；Turn 效果词汇表由 `agent.plugin_contracts.turn_effects` 拥有。

本模块保留原导入路径，避免一次性改动既有调用点。新代码和插件应导入
`agent.plugin_contracts.turn_effects`。
"""

from agent.plugin_contracts.turn_effects import (
    POST_COMMIT_EFFECT_KEY,
    TURN_EFFECTS_KEY,
    PostCommitEffect,
    TurnStorage,
    post_commit_effect,
    set_post_commit_effect,
    suppresses_post_commit,
)

__all__ = [
    "POST_COMMIT_EFFECT_KEY",
    "TURN_EFFECTS_KEY",
    "PostCommitEffect",
    "TurnStorage",
    "post_commit_effect",
    "set_post_commit_effect",
    "suppresses_post_commit",
]
