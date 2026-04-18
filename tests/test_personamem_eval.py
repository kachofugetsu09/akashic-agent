from __future__ import annotations

from pathlib import Path

from eval.personamem.dataset import load_dataset
from eval.personamem.metrics import extract_option_label, score_results


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_personamem_loader_builds_instance(tmp_path: Path) -> None:
    questions = tmp_path / "questions.csv"
    contexts = tmp_path / "contexts.jsonl"
    _write(
        questions,
        (
            "persona_id,question_id,question_type,topic,context_length_in_tokens,"
            "context_length_in_letters,distance_to_ref_in_blocks,distance_to_ref_in_tokens,"
            "num_irrelevant_tokens,distance_to_ref_proportion_in_context,user_question_or_message,"
            "correct_answer,all_options,shared_context_id,end_index_in_shared_context\n"
            "7,q1,recall_user_shared_facts,music,10,10,1,1,0,10%,Which one?,(b),"
            "\"['(a) First', '(b) Second']\",ctx1,3\n"
        ),
    )
    _write(
        contexts,
        '{"ctx1": ['
        '{"role":"system","content":"Current user persona: Likes music."},'
        '{"role":"user","content":"User: hello"},'
        '{"role":"assistant","content":"Assistant: hi"},'
        '{"role":"user","content":"User: ignored by end index"}'
        "]}\n",
    )

    instances = load_dataset(questions, contexts)

    assert len(instances) == 1
    inst = instances[0]
    assert inst.question_id == "q1"
    assert inst.gold_label == "(b)"
    assert inst.gold_option == "(b) Second"
    assert inst.persona_profile == "Current user persona: Likes music."
    assert len(inst.haystack_sessions) == 1
    assert inst.haystack_sessions[0][0].role == "user"
    assert inst.haystack_sessions[0][0].content == "hello"
    assert inst.haystack_sessions[0][1].content == "hi"


def test_personamem_loader_splits_turns_into_sessions(tmp_path: Path) -> None:
    questions = tmp_path / "questions.csv"
    contexts = tmp_path / "contexts.jsonl"
    _write(
        questions,
        (
            "persona_id,question_id,question_type,topic,context_length_in_tokens,"
            "context_length_in_letters,distance_to_ref_in_blocks,distance_to_ref_in_tokens,"
            "num_irrelevant_tokens,distance_to_ref_proportion_in_context,user_question_or_message,"
            "correct_answer,all_options,shared_context_id,end_index_in_shared_context\n"
            "7,q2,recall_user_shared_facts,music,10,10,1,1,0,10%,Which one?,(a),"
            "\"['(a) First', '(b) Second']\",ctx2,5\n"
        ),
    )
    _write(
        contexts,
        '{"ctx2": ['
        '{"role":"user","content":"User: first u"},'
        '{"role":"assistant","content":"Assistant: first a"},'
        '{"role":"user","content":"User: second u1"},'
        '{"role":"user","content":"User: second u2"},'
        '{"role":"assistant","content":"Assistant: second a"}'
        "]}\n",
    )

    instances = load_dataset(questions, contexts)

    assert len(instances) == 1
    inst = instances[0]
    assert len(inst.haystack_sessions) == 2
    assert [turn.content for turn in inst.haystack_sessions[0]] == ["first u", "first a"]
    assert [turn.content for turn in inst.haystack_sessions[1]] == [
        "second u1",
        "second u2",
        "second a",
    ]


def test_extract_option_label_supports_label_and_text() -> None:
    options = ["Alpha", "Bravo choice", "Charlie"]

    assert extract_option_label("(b)", options) == "(b)"
    assert extract_option_label("I choose b", options) == "(b)"
    assert extract_option_label("Bravo choice", options) == "(b)"


def test_score_results_returns_accuracy() -> None:
    scores = score_results(
        [
            {
                "question_type": "recall_user_shared_facts",
                "predicted_label": "(a)",
                "is_correct": True,
                "error": None,
            },
            {
                "question_type": "recall_user_shared_facts",
                "predicted_label": None,
                "is_correct": False,
                "error": "timeout",
            },
        ]
    )

    assert scores["overall"]["accuracy"] == 0.5
    assert scores["overall"]["parsed_rate"] == 0.5
    assert scores["overall"]["errors"] == 1
