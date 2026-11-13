import pytest

from htx_ai_technical.retrieval import anchor_matcher, list_blocks, locate, read_blocks, search
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


LOOKUP_DOC = ParsedDocument(
    source="fake.pdf",
    fiscal_year=2030,
    blocks=[
        Block(id="s0", type="section", title="Operating Revenue", pages=[5], text="Tax collections rose. Fund top-ups fell."),
        Block(
            id="s1",
            type="section",
            title="Top-ups to Funds",
            pages=[18],
            text="The Government will top up several funds. It will establish the Future Energy Fund with $5.0 billion.",
        ),
        Block(id="t2", type="table", title="Top-ups to Funds in FY2030", pages=[20], text="| Future Energy Fund | 5,000 |"),
    ],
)


def test_list_blocks_shows_every_block():
    lines = list_blocks(LOOKUP_DOC).splitlines()
    assert lines == [b.describe() for b in LOOKUP_DOC.blocks]


def test_read_blocks_returns_each_block_in_the_order_asked():
    assert read_blocks(LOOKUP_DOC, ["t2", "s0"]) == (
        "[t2] TABLE: Top-ups to Funds in FY2030 (pages 20)\n| Future Energy Fund | 5,000 |\n\n"
        "[s0] SECTION: Operating Revenue (pages 5)\nTax collections rose. Fund top-ups fell."
    )


def test_read_blocks_notes_unknown_ids_and_still_returns_the_rest():
    out = read_blocks(LOOKUP_DOC, ["x9", "t2"]).split("\n\n")
    assert out[0] == "[x9] No block with this id. Use list_blocks to see the ids."
    assert out[1].startswith("[t2] TABLE")


def test_read_blocks_needs_an_id():
    with pytest.raises(ValueError, match="at least one block id"):
        read_blocks(LOOKUP_DOC, [])


def test_search_ranks_by_keywords_matched_and_quotes_the_best_line():
    hits = search(LOOKUP_DOC, "Future Energy Fund").splitlines()
    assert [h.split("]")[0] for h in hits[::2]] == ["[s1", "[t2", "[s0"]  # s0 only has "Fund"
    assert hits[1].strip() == "It will establish the Future Energy Fund with $5.0 billion."


def test_search_without_matches():
    assert search(LOOKUP_DOC, "defence") == "No block mentions 'defence'."
