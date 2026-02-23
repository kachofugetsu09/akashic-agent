"""TDD for MemoryStore questions file methods."""
import pytest
from agent.memory import MemoryStore

QUESTIONS_MD = """\
## 想了解的问题

1. 睡眠最近有改善吗？
2. 离职 Shopee 后找到新方向了吗？
3. Unity 项目还在推进吗？
4. 对 AI 写代码这件事现在怎么看？
5. Elden Ring DLC 打完了吗？
"""


def test_read_questions_empty_when_no_file(tmp_path):
    store = MemoryStore(tmp_path)
    assert store.read_questions() == ""


def test_write_and_read_questions(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_questions(QUESTIONS_MD)
    assert store.read_questions() == QUESTIONS_MD


def test_write_questions_overwrites_not_appends(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_questions("旧内容")
    store.write_questions("新内容")
    assert store.read_questions() == "新内容"
    assert "旧" not in store.read_questions()


def test_remove_single_question_by_index(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_questions(QUESTIONS_MD)
    store.remove_questions_by_indices([1])
    result = store.read_questions()
    assert "睡眠" not in result           # 问题1已删
    assert "离职 Shopee" in result        # 问题2仍在
    assert "1." in result                 # 重新编号后还有1.


def test_remove_multiple_questions(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_questions(QUESTIONS_MD)
    store.remove_questions_by_indices([1, 3])
    result = store.read_questions()
    assert "睡眠" not in result           # 问题1已删
    assert "离职 Shopee" in result        # 问题2仍在
    assert "Unity" not in result          # 问题3已删
    assert "AI 写代码" in result          # 问题4仍在


def test_remaining_questions_renumbered(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_questions(QUESTIONS_MD)
    store.remove_questions_by_indices([2])   # 删问题2
    result = store.read_questions()
    lines = [l for l in result.splitlines() if l.strip()]
    numbered = [l for l in lines if l.startswith(("1.", "2.", "3.", "4."))]
    # 剩4题，编号应连续1-4
    assert any(l.startswith("1.") for l in numbered)
    assert any(l.startswith("4.") for l in numbered)
    assert not any(l.startswith("5.") for l in numbered)


def test_remove_nonexistent_index_is_noop(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_questions(QUESTIONS_MD)
    store.remove_questions_by_indices([99])
    # 原内容不变（问题数量相同）
    result = store.read_questions()
    assert "睡眠" in result
    assert "Elden Ring" in result


def test_remove_all_questions(tmp_path):
    store = MemoryStore(tmp_path)
    store.write_questions(QUESTIONS_MD)
    store.remove_questions_by_indices([1, 2, 3, 4, 5])
    result = store.read_questions()
    # 所有问题都删了，不应有编号行
    numbered = [l for l in result.splitlines() if l.strip().startswith(("1.", "2."))]
    assert numbered == []


def test_remove_from_empty_file_is_noop(tmp_path):
    store = MemoryStore(tmp_path)
    store.remove_questions_by_indices([1])   # 不应抛异常
    assert store.read_questions() == ""
