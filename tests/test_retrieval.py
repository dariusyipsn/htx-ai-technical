import pytest

from htx_ai_technical.retrieval import anchor_matcher, locate
from htx_ai_technical.schemas import Block, ParsedDocument


@pytest.mark.parametrize(
    "pattern, title, expected",
    [
        ("Budget for FY{fy}", "Table 2.1 Budget for FY2024", True),
        ("Budget for FY{fy}", "Table 2.1 Budget for FY2023", False),
        ("Budget for FY{fy}", "2.1 Budget for FY", False),
        ("Top-ups * in FY{fy}", "Table 2.4 Top-ups to Endowment and Trust Funds in FY2024", True),
        ("Top-ups * in FY{fy}", "Table 1.4 Top-ups to Statutory and Trust Funds in FY2023", False),
        ("Top-ups * in FY{fy}", "Table 2.4 Fund top-ups", False),
        ("Fiscal Position in FY{fy-2} and FY{fy-1}", "Fiscal Position in FY2022 and FY2023", True),
        ("Fiscal Position in FY{fy-2} and FY{fy-1}", "Fiscal Position in FY2023 and FY2024", False),
        ("Fiscal Position in FY{fy-2} and FY{fy-1}", "Table 3.1a: Overall Fiscal Position for FY2018 to FY2024", False),
        ("Operating Revenue", "1.2  OPERATING   revenue", True),
    ],
)
def test_anchor_matcher(pattern, title, expected):
    assert anchor_matcher(pattern, 2024)(title) is expected


DOC = ParsedDocument(
    source="fake.pdf",
    fiscal_year=2030,
    blocks=[
        Block(id="s0", type="section", title="1.2 Operating Revenue", text=""),
        Block(id="t1", type="table", title="Table 1.4 Top-ups to Statutory and Trust Funds in FY2029", text=""),
        Block(id="s1", type="section", title="Budget for FY2030", text=""),
        Block(id="t2", type="table", title="Table 2.1 Budget for FY2030", text=""),
        Block(id="s3", type="section", title="2.2 Operating Revenue", text=""),
    ],
)


def test_locate_returns_matches_in_document_order():
    assert [b.id for b in locate(DOC, "Operating Revenue", "section")] == ["s0", "s3"]


def test_locate_filters_by_block_type():
    assert [b.id for b in locate(DOC, "Budget for FY{fy}", "table")] == ["t2"]


def test_locate_only_matches_the_documents_fiscal_year():
    assert locate(DOC, "Top-ups * in FY{fy}", "table") == []
