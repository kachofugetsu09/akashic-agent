from contextlib import closing
from dataclasses import FrozenInstanceError

import pytest

from session.log import MessageConflict, MessageLog, SessionAttributes
from session.message import ContentPart, ContentReferences, Input


def test_session_admission_preserves_independent_attributes_across_restart(tmp_path):
    path = tmp_path / "sessions.db"
    with closing(MessageLog(path)) as log:
        for visibility in ("listed", "internal"):
            for learning in ("eligible", "excluded"):
                key = visibility + learning
                attributes = SessionAttributes(visibility, learning)
                assert log.ensure_session(key, attributes) == attributes
                writer = log.writer(key, author="app", source="work", body_types=(Input,),
                                    content={"text": lambda part: ContentReferences()})
                writer.append(key, Input((ContentPart("text", key),)))
                assert log.ensure_session(key, attributes) == attributes
                with pytest.raises(FrozenInstanceError):
                    attributes.learning = "eligible"
        before = dict(log.catalog().snapshot_attributes())
        with pytest.raises(MessageConflict):
            log.ensure_session("internalexcluded", SessionAttributes())
        assert dict(log.catalog().snapshot_attributes()) == before
    with closing(MessageLog(path)) as log:
        assert dict(log.catalog().snapshot_attributes()) == before
        assert log.catalog().snapshot_heads() == {key: 0 for key in before}
