import unicodedata

import pytest

from graph_llm.names.normalize import ascii_key, clean_name, is_alley_name, name_key, slugify


def test_clean_name_composes_decomposed_characters() -> None:
    decomposed = unicodedata.normalize("NFD", "Lê Lợi")

    assert decomposed != "Lê Lợi"
    assert clean_name(decomposed) == "Lê Lợi"


def test_clean_name_replaces_icelandic_eth_with_vietnamese_d() -> None:
    assert clean_name("Lê Quý Ðôn") == "Lê Quý Đôn"


def test_clean_name_collapses_whitespace() -> None:
    assert clean_name("  Nguyễn   Huệ ") == "Nguyễn Huệ"


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("Đường Cô Giang", "Cô Giang"),
        ("Đường Số 11", "Đường số 11"),
        ("Lê Quý Ðôn", "Lê Quý Đôn"),
    ],
)
def test_spellings_of_the_same_name_share_a_key(first: str, second: str) -> None:
    assert name_key(first) == name_key(second)


def test_name_key_keeps_words_after_the_prefix() -> None:
    assert name_key("Đường Số 11") == "số 11"
    assert name_key("Dẫn Cầu Calmette") == "dẫn cầu calmette"


def test_different_names_keep_different_keys() -> None:
    assert name_key("Lê Lai") != name_key("Lê Lợi")


def test_ascii_key_removes_diacritics_and_d_with_stroke() -> None:
    assert ascii_key("Điện Biên Phủ") == "dien bien phu"


def test_slugify() -> None:
    assert slugify("Cách Mạng Tháng Tám") == "cach-mang-thang-tam"
    assert slugify("Đường Lô J - Lô K Cư Xá Vĩnh Hội") == "lo-j-lo-k-cu-xa-vinh-hoi"


def test_alley_detection() -> None:
    assert is_alley_name("Hẻm 12 Pasteur")
    assert not is_alley_name("Pasteur")
