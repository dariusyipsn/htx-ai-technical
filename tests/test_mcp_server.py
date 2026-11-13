import pytest

from htx_ai_technical.mcp_server import normalize_date


@pytest.mark.parametrize(
    "text, expected",
    [
        ("16 February 2024", "2024-02-16"),
        ("15 February 2008", "2008-02-15"),
        ("February 12, 2026", "2026-02-12"),
        ("05/03/2024", "2024-03-05"),  # day first, as written in Singapore
    ],
)
def test_normalize_date(text, expected):
    assert normalize_date(text) == expected


@pytest.mark.parametrize("text", ["February 2024", "2024", "16 February", "not a date"])
def test_normalize_date_rejects_partial_or_invalid(text):
    with pytest.raises(ValueError):
        normalize_date(text)
