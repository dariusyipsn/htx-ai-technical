import pytest

from evals.part1_eval import matches


@pytest.mark.parametrize(
    "expected, actual, passed",
    [
        (28.03, 28.034, True),
        (28.03, 28.04, False),
        (-1.2, -1.2, True),
        (1.72, None, False),
        (None, None, True),
        (None, 1.72, False),
        (2024, 2024, True),
        (2024, 2025, False),
        (["Corporate Income Tax", "Betting Taxes"], ["betting taxes", "Corporate  Income Tax"], True),
        (["Corporate Income Tax", "Betting Taxes"], ["Corporate Income Tax"], False),
        (["Corporate Income Tax"], ["Corporate Income Tax", "Other Taxes"], False),
    ],
)
def test_matches(expected, actual, passed):
    assert matches(expected, actual) is passed
