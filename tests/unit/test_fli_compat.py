import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fli_mcp_compat.py"


def load_compat():
    spec = importlib.util.spec_from_file_location("fli_mcp_compat", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_locale_variant_switches_language_and_country() -> None:
    compat = load_compat()
    url = "https://www.google.com/travel/flights?tfs=abc&hl=zh-CN&gl=HK&curr=CNY"

    assert compat._locale_variant(url) == "https://www.google.com/travel/flights?tfs=abc&hl=en&gl=US&curr=CNY"


def test_locale_variant_keeps_currency_and_other_params() -> None:
    compat = load_compat()
    url = "https://www.google.com/travel/flights?tfs=abc&hl=en&gl=US&curr=CNY&q=x"

    variant = compat._locale_variant(url)

    assert "curr=CNY" in variant
    assert "q=x" in variant
    assert variant.count("hl=en") == 1
    assert variant.count("gl=US") == 1


def test_locale_variant_is_noop_without_locale_params() -> None:
    compat = load_compat()
    url = "https://www.google.com/travel/flights?tfs=abc"

    assert compat._locale_variant(url) == url
