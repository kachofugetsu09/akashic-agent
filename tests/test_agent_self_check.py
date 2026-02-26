from agent.loop import (
    _apply_inference_prefix_by_line_numbers,
    _build_inference_tone_prompt,
    _needs_inference_tone_pass,
)


def test_inference_tone_pass_detects_risky_assertion():
    assert _needs_inference_tone_pass("看起来赛事正在进行中。") is True


def test_inference_tone_pass_ignores_safe_sentence():
    assert _needs_inference_tone_pass("我不确定，可能快开始了。") is False


def test_build_inference_tone_prompt_contains_required_rules():
    prompt = _build_inference_tone_prompt(
        "1. 赛事正在进行中。",
        "【工具结果】：仅显示开幕对阵公告",
    )
    assert "prefix_line_numbers" in prompt
    assert "只做“行级判定”" in prompt


def test_apply_inference_prefix_by_line_numbers():
    response = "1. ESL Pro League S23 正在进行中\n2. PGL Cluj-Napoca 我不确定"
    got = _apply_inference_prefix_by_line_numbers(response, [1, 2])
    assert got.splitlines()[0].startswith("1. 我推测")
    assert got.splitlines()[1].endswith("我不确定")
