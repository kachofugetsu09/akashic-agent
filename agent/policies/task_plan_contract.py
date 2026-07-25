from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, cast

TaskPlanAction = Literal[
    "none",
    "plan_create",
    "plan_inspect",
    "plan_update",
]
TaskPlanContextRequirement = Literal[
    "none",
    "long_term_memory",
    "session_history",
]
BackgroundPassthroughMode = Literal[
    "none",
    "start",
    "observe",
    "output",
    "cancel",
]

_COMPLETION_BY_ACTION: dict[TaskPlanAction, str | None] = {
    "none": None,
    "plan_create": "task_plan.create",
    "plan_inspect": "task_plan.inspect",
    "plan_update": "task_plan.update",
}
_VALID_CONTEXT_REQUIREMENTS = frozenset(
    {"none", "long_term_memory", "session_history"}
)
_KNOWN_CAPABILITIES = frozenset(
    {
        "task_plan.create",
        "task_plan.inspect",
        "task_plan.update",
        "memory.recall",
        "history.search",
    }
)

_BACKGROUND_ANCHOR_TERMS = (
    "后台任务",
    "后台 job",
    "job_id",
    "subagent",
    "spawn",
)
_BACKGROUND_CANCEL_TERMS = (
    "取消",
    "停止",
    "终止",
    "cancel",
)
_BACKGROUND_OUTPUT_TERMS = (
    "输出",
    "结果",
    "output",
)
_BACKGROUND_START_TERMS: tuple[str, ...] = ()
_BACKGROUND_OBSERVE_TERMS = (
    "状态",
    "查看",
    "查询",
)
_PLAN_CREATE_TERMS = (
    "制定计划",
    "创建计划",
    "三步计划",
    "分步骤",
    "拆成步骤",
)
_NO_PLAN_CREATE_ACTION_TERMS = (
    "不要创建计划",
    "不创建计划",
    "暂不创建计划",
    "无需创建计划",
    "不需要创建计划",
    "先不制定计划",
    "暂不制定计划",
    "无需制定计划",
    "不需要制定计划",
    "别创建计划",
    "先别创建计划",
    "暂时别创建计划",
    "别制定计划",
    "先别制定计划",
    "暂时别制定计划",
    "不用创建计划",
    "不用制定计划",
    "不必创建计划",
    "不必制定计划",
    "无须创建计划",
    "无须制定计划",
)
_NO_PLAN_CREATE_ACTION_RE = re.compile(
    r"(?:请勿|不要|不再|无需|无须|不用|不必|暂时别|暂时不|暂不|先别|先不|别|不)"
    r"\s*(?:再\s*)?(?:创建|制定)\s*"
    r"(?:(?:一个|一份)\s*)?(?:(?:新的?|全新)\s*)?计划"
)
_REQUIRED_PLAN_CREATE_ACTION_RE = re.compile(
    r"(?:不得不|不能不|必须|务必|一定要)\s*(?:创建|制定)\s*"
    r"(?:(?:一个|一份)\s*)?(?:(?:新的?|全新)\s*)?计划"
)
_BACKGROUND_CREATE_RE = re.compile(
    r"(?:创建|新建|建立)\s*(?:(?:一个|一项|一份)\s*)?"
    r"(?:(?:新的?|全新)\s*)?(?:后台任务|后台\s*job|job|subagent)",
    re.IGNORECASE,
)
_BACKGROUND_START_RE = re.compile(
    r"(?:启动|运行|执行|开始|start|run|spawn)\s*"
    r"(?:(?:一个|一项|一份)\s*)?(?:(?:新的?|全新)\s*)?"
    r"(?:后台任务|后台\s*job|background\s+job|job|subagent)",
    re.IGNORECASE,
)
_NO_BACKGROUND_START_RE = re.compile(
    r"(?:请勿|不要|不再|无需|无须|不用|不必|暂时别|暂时不|暂不|先别|先不|别|不)"
    r"\s*(?:再\s*)?(?:创建|新建|建立|启动|运行|执行|开始|start|run|spawn)"
    r"\s*(?:(?:一个|一项|一份)\s*)?(?:(?:新的?|旧的?|全新)\s*)?"
    r"(?:后台任务|后台\s*job|background\s+job|job|subagent)",
    re.IGNORECASE,
)
_REQUIRED_BACKGROUND_START_RE = re.compile(
    r"(?:不得不|不能不|必须|务必|一定要)\s*"
    r"(?:创建|新建|建立|启动|运行|执行|开始|start|run|spawn)"
    r"\s*(?:(?:一个|一项|一份)\s*)?(?:(?:新的?|旧的?|全新)\s*)?"
    r"(?:后台任务|后台\s*job|background\s+job|job|subagent)",
    re.IGNORECASE,
)
_PLAN_UPDATE_TERMS = (
    "标记",
    "完成第",
    "更新步骤",
    "更新任务",
    "跳过",
    "继续执行",
    "下一步",
)
_PLAN_INSPECT_TERMS = (
    "当前任务",
    "当前进度",
    "做到哪一步",
    "任务进度",
    "这个任务",
    "那个任务",
)
_SESSION_META_PASSTHROUGH_TERMS = (
    "用了哪些工具",
    "用过哪些工具",
    "查了哪些工具",
    "调用了哪些工具",
    "工具链",
)
_NO_RETRIEVAL_TERMS = (
    "不要查询历史或记忆",
    "不要查询历史",
    "不要查询记忆",
    "不要检索历史",
    "不要检索记忆",
    "无需查询历史",
    "无需查询记忆",
    "不需要查询历史或记忆",
    "不需要查询历史",
    "不需要查询记忆",
    "不需要检索历史",
    "不需要检索记忆",
    "只创建计划",
    "只制定计划",
)
_SESSION_HISTORY_TERMS = (
    "上次讨论",
    "之前讨论",
    "聊天记录",
    "会话历史",
    "上一轮",
    "刚才讨论",
    "session",
)
_LONG_TERM_MEMORY_TERMS = (
    "我的偏好",
    "长期偏好",
    "长期记忆",
    "记住的",
    "记忆",
)


@dataclass(frozen=True)
class TaskPlanTurnContract:
    action: TaskPlanAction
    context_requirement: TaskPlanContextRequirement
    required_capabilities: frozenset[str]
    allowed_capabilities: frozenset[str]
    retrieval_budget: int
    completion_capability: str | None
    matched_terms: tuple[str, ...] = ()
    reason: str = "no_task_plan_contract"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_capabilities",
            frozenset(self.required_capabilities),
        )
        object.__setattr__(
            self,
            "allowed_capabilities",
            frozenset(self.allowed_capabilities),
        )
        object.__setattr__(self, "matched_terms", tuple(self.matched_terms))

        if self.action not in _COMPLETION_BY_ACTION:
            raise ValueError("unsupported TaskPlan action")
        if self.context_requirement not in _VALID_CONTEXT_REQUIREMENTS:
            raise ValueError("unsupported TaskPlan context requirement")
        if type(self.retrieval_budget) is not int or self.retrieval_budget not in (0, 1):
            raise ValueError("retrieval budget must be integer zero or one")
        if not all(
            isinstance(item, str)
            for item in self.required_capabilities | self.allowed_capabilities
        ):
            raise ValueError("capabilities must be strings")
        unknown_capabilities = (
            self.required_capabilities | self.allowed_capabilities
        ) - _KNOWN_CAPABILITIES
        if unknown_capabilities:
            raise ValueError("unsupported TaskPlan capability")

        if self.action == "none":
            if (
                self.context_requirement != "none"
                or self.required_capabilities
                or self.allowed_capabilities
                or self.retrieval_budget != 0
                or self.completion_capability is not None
            ):
                raise ValueError("inactive TaskPlan contract must use canonical empty state")
            return

        if self.required_capabilities - self.allowed_capabilities:
            raise ValueError("required capabilities must be allowed")
        expected_completion = _COMPLETION_BY_ACTION.get(self.action)
        if self.completion_capability != expected_completion:
            raise ValueError("completion capability does not match TaskPlan action")
        if self.completion_capability not in self.required_capabilities:
            raise ValueError("completion capability must be required")
        if self.required_capabilities != frozenset({self.completion_capability}):
            raise ValueError("only the completion capability may be required")

        action_capabilities = {
            "plan_create": {"task_plan.create", "task_plan.inspect"},
            "plan_inspect": {"task_plan.inspect"},
            "plan_update": {"task_plan.inspect", "task_plan.update"},
        }[self.action]
        allowed_state_capabilities = self.allowed_capabilities & {
            "task_plan.create",
            "task_plan.inspect",
            "task_plan.update",
        }
        if not allowed_state_capabilities <= action_capabilities:
            raise ValueError("state capability does not match TaskPlan action")

        if self.context_requirement == "none":
            if self.retrieval_budget != 0:
                raise ValueError("no-context contract cannot have retrieval budget")
            if self.allowed_capabilities & {"memory.recall", "history.search"}:
                raise ValueError("no-context contract cannot allow retrieval")
            return
        if self.action != "plan_create" or self.retrieval_budget != 1:
            raise ValueError("context retrieval is only valid for plan creation")
        required_context_capability = (
            "memory.recall"
            if self.context_requirement == "long_term_memory"
            else "history.search"
        )
        if required_context_capability not in self.allowed_capabilities:
            raise ValueError("context capability must be allowed")
        forbidden_context_capability = (
            "history.search"
            if required_context_capability == "memory.recall"
            else "memory.recall"
        )
        if forbidden_context_capability in self.allowed_capabilities:
            raise ValueError("contract cannot mix context capability families")

    @property
    def active(self) -> bool:
        return self.action != "none"

    @classmethod
    def inactive(
        cls,
        *,
        reason: str = "no_task_plan_contract",
        matched_terms: tuple[str, ...] = (),
    ) -> TaskPlanTurnContract:
        return cls(
            action="none",
            context_requirement="none",
            required_capabilities=frozenset(),
            allowed_capabilities=frozenset(),
            retrieval_budget=0,
            completion_capability=None,
            matched_terms=matched_terms,
            reason=reason,
        )

    def to_trace_metadata(self) -> dict[str, object]:
        return {
            "active": self.active,
            "action": self.action,
            "context_requirement": self.context_requirement,
            "required_capabilities": sorted(self.required_capabilities),
            "allowed_capabilities": sorted(self.allowed_capabilities),
            "retrieval_budget": self.retrieval_budget,
            "completion_capability": self.completion_capability,
            "matched_terms": list(self.matched_terms),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class TaskPlanIntentDecision:
    contract: TaskPlanTurnContract
    background_mode: BackgroundPassthroughMode = "none"


def infer_task_plan_turn_decision(
    user_text: str,
    *,
    has_active_task: bool,
) -> TaskPlanIntentDecision:
    text = str(user_text or "")
    background_mode, background_terms = _infer_background_mode(text)
    if background_mode != "none":
        return TaskPlanIntentDecision(
            contract=TaskPlanTurnContract.inactive(
                reason=f"background_{background_mode}_passthrough",
                matched_terms=background_terms,
            ),
            background_mode=background_mode,
        )

    session_meta_terms = _matched_terms(text, _SESSION_META_PASSTHROUGH_TERMS)
    if session_meta_terms:
        return TaskPlanIntentDecision(
            contract=TaskPlanTurnContract.inactive(
                reason="session_meta_passthrough",
                matched_terms=session_meta_terms,
            )
        )

    create_terms = _matched_terms(text, _PLAN_CREATE_TERMS)
    no_create_terms = _matched_no_create_terms(text)
    update_terms = _matched_terms(text, _PLAN_UPDATE_TERMS)
    inspect_terms = _matched_terms(text, _PLAN_INSPECT_TERMS)
    if create_terms and no_create_terms:
        if update_terms and has_active_task:
            return TaskPlanIntentDecision(
                contract=_active_contract(
                    action="plan_update",
                    text=text,
                    has_active_task=has_active_task,
                    action_terms=update_terms,
                )
            )
        if inspect_terms or update_terms:
            return TaskPlanIntentDecision(
                contract=_active_contract(
                    action="plan_inspect",
                    text=text,
                    has_active_task=has_active_task,
                    action_terms=inspect_terms or update_terms,
                )
            )
        return TaskPlanIntentDecision(
            contract=TaskPlanTurnContract.inactive(
                reason="explicit_no_task_plan_action",
                matched_terms=no_create_terms,
            )
        )
    if create_terms:
        return TaskPlanIntentDecision(
            contract=_active_contract(
                action="plan_create",
                text=text,
                has_active_task=has_active_task,
                action_terms=create_terms,
            )
        )

    explicit_update_terms = tuple(
        term
        for term in update_terms
        if term != "继续执行"
        and not (term == "下一步" and "继续下一步" not in text)
    )
    if inspect_terms and not explicit_update_terms:
        return TaskPlanIntentDecision(
            contract=_active_contract(
                action="plan_inspect",
                text=text,
                has_active_task=has_active_task,
                action_terms=inspect_terms,
            )
        )

    if update_terms and has_active_task:
        return TaskPlanIntentDecision(
            contract=_active_contract(
                action="plan_update",
                text=text,
                has_active_task=has_active_task,
                action_terms=update_terms,
            )
        )

    if inspect_terms or update_terms:
        return TaskPlanIntentDecision(
            contract=_active_contract(
                action="plan_inspect",
                text=text,
                has_active_task=has_active_task,
                action_terms=inspect_terms or update_terms,
            )
        )

    return TaskPlanIntentDecision(contract=TaskPlanTurnContract.inactive())


def _active_contract(
    *,
    action: TaskPlanAction,
    text: str,
    has_active_task: bool,
    action_terms: tuple[str, ...],
) -> TaskPlanTurnContract:
    required = frozenset({_COMPLETION_BY_ACTION[action]})
    allowed = set(required)
    context_requirement: TaskPlanContextRequirement = "none"
    retrieval_budget = 0
    context_terms: tuple[str, ...] = ()

    if action == "plan_create":
        if has_active_task:
            allowed.add("task_plan.inspect")
        no_retrieval_terms = _matched_terms(text, _NO_RETRIEVAL_TERMS)
        if no_retrieval_terms:
            context_terms = no_retrieval_terms
        else:
            session_terms = _matched_terms(text, _SESSION_HISTORY_TERMS)
            memory_terms = _matched_terms(text, _LONG_TERM_MEMORY_TERMS)
            if session_terms:
                context_requirement = "session_history"
                allowed.add("history.search")
                retrieval_budget = 1
                context_terms = session_terms
            elif memory_terms:
                context_requirement = "long_term_memory"
                allowed.add("memory.recall")
                retrieval_budget = 1
                context_terms = memory_terms
    elif action == "plan_update":
        allowed.add("task_plan.inspect")

    return TaskPlanTurnContract(
        action=action,
        context_requirement=context_requirement,
        required_capabilities=frozenset(item for item in required if item is not None),
        allowed_capabilities=frozenset(item for item in allowed if item is not None),
        retrieval_budget=retrieval_budget,
        completion_capability=_COMPLETION_BY_ACTION[action],
        matched_terms=_dedupe_terms((*action_terms, *context_terms)),
        reason=f"{action}_{context_requirement}",
    )


def _infer_background_mode(
    text: str,
) -> tuple[BackgroundPassthroughMode, tuple[str, ...]]:
    anchors = _matched_terms(text, _BACKGROUND_ANCHOR_TERMS)
    if not anchors:
        return "none", ()
    for mode, operation_terms in (
        ("cancel", _BACKGROUND_CANCEL_TERMS),
        ("output", _BACKGROUND_OUTPUT_TERMS),
        ("start", _BACKGROUND_START_TERMS),
        ("observe", _BACKGROUND_OBSERVE_TERMS),
    ):
        matched = _matched_terms(text, operation_terms)
        if mode == "start":
            positive_start_text = (
                text
                if _REQUIRED_BACKGROUND_START_RE.search(text)
                else _NO_BACKGROUND_START_RE.sub("", text)
            )
            explicit_background_create = _matched_regex_terms(
                positive_start_text,
                _BACKGROUND_CREATE_RE,
            )
            explicit_background_start = _matched_regex_terms(
                positive_start_text,
                _BACKGROUND_START_RE,
            )
            matched = _dedupe_terms(
                (
                    *explicit_background_create,
                    *explicit_background_start,
                    *matched,
                )
            )
        if matched:
            return cast(BackgroundPassthroughMode, mode), _dedupe_terms(
                (*anchors, *matched)
            )
    return "observe", anchors


def _matched_terms(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    normalized = (text or "").lower()
    return tuple(term for term in terms if term.lower() in normalized)


def _matched_no_create_terms(text: str) -> tuple[str, ...]:
    if _matched_regex_terms(text, _REQUIRED_PLAN_CREATE_ACTION_RE):
        return ()
    return _dedupe_terms(
        (
            *_matched_regex_terms(text, _NO_PLAN_CREATE_ACTION_RE),
            *_matched_terms(text, _NO_PLAN_CREATE_ACTION_TERMS),
        )
    )


def _matched_regex_terms(text: str, pattern: re.Pattern[str]) -> tuple[str, ...]:
    return tuple(match.group(0) for match in pattern.finditer(text or ""))


def _dedupe_terms(terms: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(terms))
