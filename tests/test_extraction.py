from htx_ai_technical.extraction import GroupResult, definitions, field_answers, locate_block
from htx_ai_technical.fields import CorporateIncomeTax, OperatingRevenueTaxes, TopUps
from htx_ai_technical.schemas import Block, ParsedDocument, TableValue, TextList

BUDGET = Block(id="t1", type="table", title="Table 2.1 Budget for FY2030", pages=[16], text="")
DOC = ParsedDocument(
    source="fake.pdf",
    fiscal_year=2030,
    blocks=[
        Block(id="s0", type="section", title="1.2 Operating Revenue", pages=[5, 6], text=""),
        BUDGET,
        Block(id="s2", type="section", title="2.2 Operating Revenue", pages=[13], text=""),
    ],
)


def test_locate_block_takes_first_anchor_match():
    assert locate_block(DOC, OperatingRevenueTaxes) == DOC.blocks[0]
    assert locate_block(DOC, CorporateIncomeTax) == BUDGET


def test_locate_block_no_match():
    assert locate_block(DOC, TopUps) is None


def test_definitions_lists_every_field():
    lines = definitions(CorporateIncomeTax).splitlines()
    assert [line.split(":")[0] for line in lines] == ["- corporate_income_tax_bn", "- corporate_income_tax_yoy_pct"]


def test_field_answers_table_value():
    output = CorporateIncomeTax(
        corporate_income_tax_bn=TableValue(value=28.03, row_label="Corporate Income Tax", column_label="Estimated FY2030", evidence="28.03"),
        corporate_income_tax_yoy_pct=TableValue(value=-1.2, row_label="Corporate Income Tax", column_label="% change", evidence="(1.2)"),
    )
    answers = field_answers(GroupResult(CorporateIncomeTax, BUDGET, output=output))

    value, prov = answers["corporate_income_tax_bn"]
    assert value == 28.03
    assert (prov.block_id, prov.pages) == ("t1", [16])
    assert (prov.row, prov.column, prov.evidence, prov.error) == ("Corporate Income Tax", "Estimated FY2030", "28.03", None)
    assert answers["corporate_income_tax_yoy_pct"][0] == -1.2


def test_field_answers_text_list():
    output = OperatingRevenueTaxes(operating_revenue_taxes=TextList(value=["Betting Taxes"], evidence=["Betting Taxes"]))
    value, prov = field_answers(GroupResult(OperatingRevenueTaxes, DOC.blocks[0], output=output))["operating_revenue_taxes"]
    assert value == ["Betting Taxes"]
    assert (prov.row, prov.column, prov.evidence) == (None, None, ["Betting Taxes"])


def test_field_answers_failed_call_keeps_block_and_error():
    result = GroupResult(CorporateIncomeTax, BUDGET, error="TimeoutError: slow")
    for value, prov in field_answers(result).values():
        assert value is None
        assert (prov.block_id, prov.error) == ("t1", "TimeoutError: slow")
