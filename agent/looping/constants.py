import re

_FLOW_TRIGGER_WORDS = (
    "步骤",
    "流程",
    "下次",
    "按这个逻辑",
)
_FLOW_SEQUENCE_PATTERN = re.compile(r"先.{0,20}再")
