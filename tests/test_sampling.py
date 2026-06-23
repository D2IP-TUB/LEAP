from leap.core.actions.action import Action
from leap.generation.sampling import SamplingLayer


def test_add_column_argument_cleaning_removes_echoed_function_call():
    raw_args = "f_add_column(display type, ['monochrome', 'color'])"

    cleaned = SamplingLayer._clean_argument_text(raw_args, "add_column")
    action = Action.parse(f"add_column({cleaned})")

    assert cleaned == "display type, ['monochrome', 'color']"
    assert action is not None
    assert action.name == "add_column"
    assert action.arguments == ("display type", ["monochrome", "color"])
