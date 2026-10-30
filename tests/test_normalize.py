from pathlib import Path

import pytest

from persian_rag.normalize import NormalizeConfig, normalize

FIXTURES = Path(__file__).parent / "fixtures"


def test_unifies_arabic_yeh_and_kaf_and_presentation_forms() -> None:
    assert normalize("يك كتاب") == "یک کتاب"
    assert normalize("على") == "علی"
    assert normalize("\ufef1\ufed9") == "یک"
    assert normalize("أحمد وإسلام") == "احمد واسلام"
    assert normalize("مؤمن") == "مومن"
    assert normalize("صلاة") == "صلاه"
    assert normalize("آب") == "آب"
    assert normalize("سؤال") == "سوال"


def test_digits_and_diacritics_are_configurable() -> None:
    raw = "۱۲۳ و 456 و ٤٥٦ و کِتاب"
    assert normalize(raw) == "۱۲۳ و ۴۵۶ و ۴۵۶ و کِتاب"
    assert normalize(raw, NormalizeConfig(digits="ascii")) == "123 و 456 و 456 و کِتاب"
    assert normalize(raw, NormalizeConfig(digits="keep")) == "۱۲۳ و 456 و ٤٥٦ و کِتاب"
    stripped = normalize(raw, NormalizeConfig(remove_diacritics=True))
    assert "کتاب" in stripped
    assert "\u0650" not in stripped


def test_zwnj_policy_and_affixes() -> None:
    assert normalize("می\u200c\u200cروم") == "می‌روم"
    assert normalize("می   رود") == "می‌رود"
    assert normalize("نمی توانم") == "نمی‌توانم"
    assert normalize("می-روم") == "می‌روم"
    assert normalize("کتاب ها") == "کتاب‌ها"
    assert normalize("کتاب هایی") == "کتاب‌هایی"
    assert normalize("ساده ترین") == "ساده‌ترین"
    assert normalize("بهتر") == "بهتر"
    assert normalize("می‌روم", NormalizeConfig(zwnj="remove")) == "میروم"
    assert normalize("می‌رود", NormalizeConfig(zwnj="space")) == "می رود"
    untouched = normalize("کتاب ها", NormalizeConfig(correct_half_spaces=False))
    assert untouched == "کتاب ها"


def test_punctuation_tatweel_and_whitespace() -> None:
    assert normalize("سلام?") == "سلام؟"
    assert normalize("واقعا!!!") == "واقعا!"
    assert normalize("صبر...") == "صبر..."
    assert normalize("ســـلام") == "سلام"
    assert normalize("سلام ،  دنیا") == "سلام، دنیا"
    assert normalize("سلام•دنیا") == "سلام دنیا"
    assert normalize("  خط\r\nبعد   ") == "خط\nبعد"
    assert normalize("") == ""


def test_normalize_is_idempotent_on_fixture() -> None:
    raw = (FIXTURES / "messy.txt").read_text(encoding="utf-8")
    configs = [
        NormalizeConfig(),
        NormalizeConfig(digits="ascii", remove_diacritics=True, punctuation="ascii"),
        NormalizeConfig(zwnj="remove", clean_punctuation=False, unify_arabic_chars=False),
    ]
    for config in configs:
        once = normalize(raw, config)
        twice = normalize(once, config)
        assert once == twice


def test_invalid_config_is_rejected() -> None:
    with pytest.raises(ValueError):
        NormalizeConfig(digits="roman")  # type: ignore[arg-type]
