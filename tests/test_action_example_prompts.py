from pathlib import Path
from types import SimpleNamespace

from leap.core import Action, Table
from leap.core.actions import REGISTRY
from leap.generation.action_examples import ActionExamplesManager, PromptCatalog
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
        "add_column": '"attendance number", ["32092", "34186", "17503"]',
        "group_by": '"country"',
        "sort_by": '"position", "desc"',
    }

    for action_name, expected in expected_first_answers.items():
        examples = manager.get_examples(action_name)
        assert examples[0].answer == expected
        assert not examples[0].answer.startswith(f"f_{action_name}(")


def test_phase_two_few_shot_and_current_turns_keep_table_and_question_shape():
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
    assert all("Table:" in content and "Question:" in content for content in user_messages)
    assert all("Arguments only for" not in content for content in user_messages)


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


def test_phase_one_prompt_has_no_add_column_evidence_when_disabled():
    tokenizer = RecordingTokenizer()
    builder = PromptBuilder(tokenizer=tokenizer, is_instruct=True)
    worker = SimpleNamespace(
        max_model_len=4096,
        use_constraints=False,
        use_global_constraints=False,
        constraint_backend="legacy_state_machine",
    )
    table = Table(columns=["Name", "Age"], rows=[["Alice", "25"]])
    previous_enabled = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["select_row", "select_column", "group_by", "sort_by", "end"])
    try:
        builder.build_cot_action_prompt(
            question="Who is oldest?",
            table=table,
            action_history=[],
            worker=worker,
        )
    finally:
        REGISTRY._enabled_actions = previous_enabled

    assert "add_column" not in "\n".join(message["content"] for message in tokenizer.messages)
    assert [message["content"] for message in tokenizer.messages if message["role"] == "assistant"] == [
        'f_select_row(["row 0", "row 1"]) -> f_select_column(["date", "league"]) -> f_sort_by("date", "desc") -> f_end()',
        'f_select_row(["row 0", "row 2"]) -> f_select_column(["athlete"]) -> f_group_by("athlete") -> f_end()',
        'f_select_row([*]) -> f_select_column(["when", "results; final score"]) -> f_sort_by("results; final score", "desc") -> f_end()',
        'f_select_row(["row 2"]) -> f_select_column(["status"]) -> f_group_by("status") -> f_end()',
    ]


def test_global_constraints_keep_add_column_prompt_variant_when_supported():
    tokenizer = RecordingTokenizer()
    builder = PromptBuilder(tokenizer=tokenizer, is_instruct=True)
    worker = SimpleNamespace(
        max_model_len=4096,
        use_constraints=True,
        use_global_constraints=True,
        constraint_backend="legacy_state_machine",
    )
    previous_enabled = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["select_row", "add_column", "end"])
    try:
        builder.build_cot_action_prompt(
            question="Who is listed?",
            table=Table(columns=["Name"], rows=[["Alice"]]),
            action_history=[],
            worker=worker,
        )
    finally:
        REGISTRY._enabled_actions = previous_enabled

    prompt_content = "\n".join(message["content"] for message in tokenizer.messages)
    assert "add_column" in prompt_content


def test_xgrammar_uses_add_column_prompt_variant():
    tokenizer = RecordingTokenizer()
    builder = PromptBuilder(tokenizer=tokenizer, is_instruct=True)
    worker = SimpleNamespace(
        max_model_len=4096,
        use_constraints=True,
        use_global_constraints=False,
        constraint_backend="xgrammar",
    )
    previous_enabled = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["select_row", "add_column", "end"])
    try:
        builder.build_cot_action_prompt(
            question="Who is listed?",
            table=Table(columns=["Name"], rows=[["Alice"]]),
            action_history=[],
            worker=worker,
        )
    finally:
        REGISTRY._enabled_actions = previous_enabled

    assert "add_column" in "\n".join(message["content"] for message in tokenizer.messages)


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


def test_prompt_assets_are_split_by_workflow():
    prompt_dir = Path(__file__).parents[1] / "configs" / "prompts"
    assert {path.name for path in prompt_dir.glob("*.yaml")} == {
        "iterative.yaml",
        "iterative_json.yaml",
        "cot.yaml",
        "cot_json.yaml",
        "direct_query.yaml",
    }
    assert not (prompt_dir.parent / "action_examples.yaml").exists()


def test_iterative_examples_match_cot_chains_and_apply_sequentially():
    catalog = PromptCatalog()
    cot_examples = catalog.cot_examples_manager.get_examples("action_selection")

    assert len(catalog.iterative_examples) == len(cot_examples) == 4
    for iterative, cot in zip(catalog.iterative_examples, cot_examples):
        assert iterative.table == cot.table
        assert iterative.question == cot.question
        assert iterative.answer == cot.answer
        assert iterative.answer_without_add_column == cot.answer_without_add_column

        for chain in (iterative.answer, iterative.answer_without_add_column):
            table = iterative.table
            for call in chain.split("->"):
                action = Action.parse(call.strip())
                assert action is not None
                assert action.is_valid_for_table(table)
                next_table = action.apply_to_table(table)
                assert next_table is not None
                table = next_table


def test_iterative_system_variants_include_second_cot_example_per_operation():
    catalog = PromptCatalog()
    with_add_column = catalog.iterative_system(add_column_available=True)
    without_add_column = catalog.iterative_system(add_column_available=False)
    selected_examples = {
        "select_row": 0,
        "select_column": 0,
        "add_column": 0,
        "group_by": 0,
        "sort_by": 1,
    }

    for action_name, example_idx in selected_examples.items():
        source = catalog.cot["examples"][action_name]["examples"][example_idx]
        operation = f"f_{action_name}({source['answer']})"
        assert source["table"].strip() in with_add_column
        assert f"Question: {source['question']}" in with_add_column
        assert f"Operation: {operation}" in with_add_column
        if action_name != "add_column":
            assert source["table"].strip() in without_add_column
            assert f"Question: {source['question']}" in without_add_column
            assert f"Operation: {operation}" in without_add_column

    assert 'Operation: f_sort_by("tackles", "asc")' in with_add_column
    assert 'Operation: f_sort_by("tackles", "asc")' in without_add_column
    assert "add_column" not in without_add_column


def test_iterative_xgrammar_prompt_includes_add_column_evidence():
    tokenizer = RecordingTokenizer()
    builder = PromptBuilder(tokenizer=tokenizer, is_instruct=True)
    worker = SimpleNamespace(
        max_model_len=32000,
        use_constraints=True,
        use_global_constraints=False,
        constraint_backend="xgrammar",
    )
    previous_enabled = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["select_row", "select_column", "add_column", "group_by", "sort_by", "end"])
    try:
        builder.build_iterative_prompt(
            question="Who is oldest?",
            table=Table(columns=["Name", "Age"], rows=[["Alice", "25"]]),
            action_history=[],
            worker=worker,
            step=0,
        )
    finally:
        REGISTRY._enabled_actions = previous_enabled

    prompt = "\n".join(message["content"] for message in tokenizer.messages)
    assert "add_column" in prompt
    assert "Return only the single next operation" in prompt
    assert any("->" in message["content"] for message in tokenizer.messages if message["role"] == "assistant")


def test_iterative_legacy_prompt_uses_add_column_variant_when_available():
    tokenizer = RecordingTokenizer()
    builder = PromptBuilder(tokenizer=tokenizer, is_instruct=True)
    worker = SimpleNamespace(
        max_model_len=32000,
        use_constraints=True,
        use_global_constraints=False,
        constraint_backend="legacy_state_machine",
    )
    previous_enabled = REGISTRY._enabled_actions
    REGISTRY.set_enabled_actions(["select_row", "select_column", "add_column", "group_by", "sort_by", "end"])
    try:
        builder.build_iterative_prompt(
            question="Who is oldest?",
            table=Table(columns=["Name", "Age"], rows=[["Alice", "25"]]),
            action_history=[],
            worker=worker,
            step=0,
        )
    finally:
        REGISTRY._enabled_actions = previous_enabled

    prompt = "\n".join(message["content"] for message in tokenizer.messages)
    assert "f_add_column" in prompt
    assert [message["content"] for message in tokenizer.messages if message["role"] == "assistant"] == [
        example.answer for example in builder.prompt_catalog.iterative_examples
    ]


def test_direct_query_prompt_uses_dedicated_catalog():
    tokenizer = RecordingTokenizer()
    builder = PromptBuilder(tokenizer=tokenizer, is_instruct=True)
    builder.build_query_prompt(
        question="Who is listed?",
        table=Table(columns=["Name"], rows=[["Alice"]]),
        action_history=["end()", "direct_query()"],
        worker=SimpleNamespace(max_model_len=4096),
    )

    catalog = builder.prompt_catalog
    assert tokenizer.messages[0] == {"role": "system", "content": catalog.direct_query["system"].rstrip()}
    assistant_messages = [message for message in tokenizer.messages if message["role"] == "assistant"]
    assert [message["content"] for message in assistant_messages] == [example.answer for example in catalog.direct_query_examples]
