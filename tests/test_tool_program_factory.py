from collections.abc import Callable, Mapping

import pytest

from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_WRITERS
from agent.plugin_contracts import CallRef, ContentPart, ContentReferences
from agent.plugin_composition.models import ToolCall as ModelToolCall
from plugins.tools.menu import ToolMenu
from plugins.tools.program import ToolProgramFactory


class _Reader:
    session_id = "session"


class _Writer:
    def __init__(self, session_id: str, call_ref: CallRef):
        self.session_id = session_id
        self.call_ref = call_ref
        self.expired = False

    def expire(self) -> None:
        self.expired = True


class _Writers:
    def __init__(self):
        self.bound: dict[str, object] = {}

    def bind(self, ctx: object, **kwargs: object) -> Callable[..., _Writer]:
        self.bound = kwargs

        def open_writer(session_id: str, *, call_ref: CallRef) -> _Writer:
            return _Writer(session_id, call_ref)

        return open_writer


class _Bindings:
    def describe(self, binding_id: str, _key: object) -> Mapping[str, object]:
        return {
            "tool": {
                "name": binding_id,
                "description": "test tool",
                "parameters": {"type": "object"},
            }
        }


class _Context:
    def __init__(self, writers: _Writers, bindings: _Bindings):
        self.writers = writers
        self.bindings = bindings

    def require(self, key: object) -> object:
        if key == MESSAGE_WRITERS:
            return self.writers
        if key == BINDINGS:
            return self.bindings
        raise AssertionError(f"unexpected service: {key}")


class _Catalog:
    def __init__(self):
        self.authorize = None
        self.child_permit = None

    def execution(self, authorize, *, child_permit=None):
        self.authorize = authorize
        self.child_permit = child_permit
        return object()

    def bind(self, ref, bindings, *, configuration=None):
        return ref.name


def _check_text(part: ContentPart) -> ContentReferences:
    return ContentReferences()


def test_factory_returns_real_menu_and_scoped_reply() -> None:
    writers = _Writers()
    catalog = _Catalog()
    factory = ToolProgramFactory(_Context(writers, _Bindings()), catalog)
    reader = _Reader()
    check_start = lambda: None

    async def authorize(binding_id: str, arguments: Mapping[str, object]):
        return {"binding": binding_id}

    menu = factory.create_menu(
        reader,
        "conversation",
        content={"text": _check_text},
        check_start=check_start,
        authorize=authorize,
        fixed_bindings={"example": "example"},
        child_permit=lambda: object(),
    )

    assert menu.names == frozenset({"example"})
    decoded = menu.decode(ModelToolCall("wire", "example", {}))
    assert decoded.accepted
    assert decoded.binding_id == "example"
    assert catalog.authorize is authorize
    assert catalog.child_permit is not None

    reply = factory.bind_reply(
        reader,
        "conversation",
        content={"text": _check_text},
        check_start=check_start,
    )(CallRef("call", 0))
    assert reply.message_id == "tool-result:call:0"
    assert reply.reader is reader
    assert reply.writer.session_id == reader.session_id
    assert reply.writer.call_ref == reply.call_ref
    assert writers.bound["author"] == "tool"
    assert writers.bound["source"] == "conversation"


def test_factory_requires_a_source_and_start_check() -> None:
    writers = _Writers()
    factory = ToolProgramFactory(_Context(writers, _Bindings()), _Catalog())
    with pytest.raises(ValueError, match="来源"):
        factory.bind_reply(
            _Reader(),
            "",
            content={},
            check_start=lambda: None,
        )
    with pytest.raises(TypeError, match="启动检查器"):
        factory.bind_reply(
            _Reader(),
            "conversation",
            content={},
            check_start=object(),  # type: ignore[arg-type]
        )


def test_menu_keeps_internal_binding_errors_fail_loud() -> None:
    writers = _Writers()
    catalog = _Catalog()
    factory = ToolProgramFactory(_Context(writers, _Bindings()), catalog)

    class _BrokenPresentation:
        schemas = ()
        system_prompt = ""

        def decode(self, call):
            return "missing", {}

        def configuration(self, name):
            return None

    menu = factory.create_menu(
        _Reader(),
        "conversation",
        content={},
        check_start=lambda: None,
        authorize=lambda binding_id, arguments: None,  # type: ignore[arg-type]
        fixed_bindings={"example": "example"},
        presentation=_BrokenPresentation(),
    )
    with pytest.raises(PermissionError, match="未获授"):
        menu.decode(ModelToolCall("wire", "missing", {}))
