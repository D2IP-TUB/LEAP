from types import SimpleNamespace

from leap.core import Action, Table
from leap.core.actions import REGISTRY
from leap.generation.action_examples import ActionExamplesManager
from leap.generation.prompt_builder import PromptBuilder


class RecordingTokenizer:
    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return "\n".join(f"{message['role']}:{message['content']}" for message in messages)


def test_phase_two_examples_use_canonical_argument_only_syntax():
    manager = ActionExamplesManager()
    expected_first_answers = {
        "select_row": "[*]",
        "select_column": '["cardiff win", "draw"]',
        "add_column": '"attendance number", [32092, 34186, 17503]',
        "group_by": '"country"',
        "sort_by": '"position", "desc"',
    }

    for action_name, expected in expected_first_answers.items():
        examples = manager.get_examples(action_name)
        assert examples[0].answer == expected
        assert not examples[0].answer.startswith(f"f_{action_name}(")


def test_phase_two_few_shot_and_current_turns_request_arguments_only():
    tokenizer = RecordingTokenizer()
    builder = PromptBuilder(tokenizer=tokenizer, is_instruct=True)
    worker = SimpleNamespace(max_model_len=4096)
    table = Table(columns=["Name", "Age"], rows=[["Alice", "25"]])

    builder.build_cot_arguments_prompt(
        question="Who is listed?",
        table=table,
        action_name="select_column",
        action_history=[],
        worker=worker,
    )

    user_messages = [message["content"] for message in tokenizer.messages if message["role"] == "user"]
    assert user_messages
    assert all(content.rstrip().endswith("Arguments only for f_select_column:") for content in user_messages)


def test_phase_one_few_shot_chains_use_canonical_function_syntax():
    tokenizer = RecordingTokenizer()
    builder = PromptBuilder(tokenizer=tokenizer, is_instruct=True)
    worker = SimpleNamespace(max_model_len=4096)
    table = Table(columns=["Name", "Age"], rows=[["Alice", "25"]])
    previous_enabled = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["select_row", "select_column", "add_column", "group_by", "sort_by", "end"])
    try:
        builder.build_cot_action_prompt(
            question="Who is oldest?",
            table=table,
            action_history=[],
            worker=worker,
        )
    finally:
        REGISTRY._enabled_actions = previous_enabled

    user_messages = [message["content"] for message in tokenizer.messages if message["role"] == "user"]
    assistant_messages = [message["content"] for message in tokenizer.messages if message["role"] == "assistant"]
    assert user_messages
    assert assistant_messages
    for chain in assistant_messages:
        calls = [call.strip() for call in chain.split("->")]
        assert calls[-1] == "f_end()"
        assert all(Action.parse(call) is not None for call in calls)


def test_phase_one_examples_use_direct_questions_and_requested_chains():
    examples = ActionExamplesManager().get_examples("action_selection")

    assert [example.question for example in examples] == [
        "What was the last year when this team was part of the USL A-League?",
        "How many athletes are from India?",
        "When was the competition with the highest points scored played?",
        "How many standards were published in 2011?",
    ]
    assert [example.answer for example in examples] == [
        'f_add_column("year", [2001, 2002, 2005]) -> f_select_row(["row 0", "row 1"]) -> '
        'f_select_column(["year", "league"]) -> f_sort_by("year", "desc") -> f_end()',
        'f_add_column("country of athletes", ["India", "Kazakhstan", "India"]) -> '
        'f_select_row(["row 0", "row 2"]) -> f_select_column(["athlete", "country of athletes"]) -> '
        'f_group_by("country of athletes") -> f_end()',
        'f_add_column("points scored", [27, 37, 33]) -> f_select_row([*]) -> '
        'f_select_column(["when", "points scored"]) -> f_sort_by("points scored", "desc") -> f_end()',
        'f_add_column("year", [2005, 2008, 2011]) -> f_select_row(["row 2"]) -> f_select_column(["year"]) -> f_group_by("year") -> f_end()',
    ]
