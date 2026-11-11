import pytest

from htx_ai_technical.parsing import detect_fiscal_year, grid_to_markdown, heading_level


@pytest.mark.parametrize(
    "text, level",
    [
        ("01 Update on Financial Year 2024", 1),
        ("OPERATING REVENUE", 1),
        ("1.2 Operating Revenue", 2),
        ("2.1.3 Corporate Income Tax", 3),
        ("Key revenue measures", 3),
    ],
)
def test_heading_level(text, level):
    assert heading_level(text) == level


def test_detect_fiscal_year():
    assert detect_fiscal_year("Analysis of Revenue and Expenditure\nFinancial  Year 2024") == 2024
    assert detect_fiscal_year("no year here") is None


def test_grid_to_markdown_merges_header_rows():
    grid = [
        ["", "Estimated FY2030", "Change over Revised FY2029"],
        ["", "$billion", "% change"],
        ["Corporate Income Tax", "28.03", "(1.2)"],
    ]
    assert grid_to_markdown(grid, header_rows=2) == (
        "|  | Estimated FY2030 $billion | Change over Revised FY2029 % change |\n"
        "|---|---|---|\n"
        "| Corporate Income Tax | 28.03 | (1.2) |"
    )


def test_grid_to_markdown_pads_short_rows():
    assert grid_to_markdown([["A", "B"], ["x"]], header_rows=1) == "| A | B |\n|---|---|\n| x |  |"


def test_grid_to_markdown_empty():
    assert grid_to_markdown([], header_rows=0) == ""
