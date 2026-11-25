"""Normalization of Vietnamese street names.

Three forms of a name are used:

- the clean name: the OSM spelling in Unicode NFC with look-alike characters fixed
  and whitespace collapsed; this is what is shown to people;
- the name key: the clean name in lower case without a leading "Đường"; two OSM
  spellings with the same name key are the same name;
- the ASCII key: the name key without diacritics; it is only used for lookups,
  never to decide that two names are the same, because different Vietnamese names
  can share an ASCII form.
"""

import re
import unicodedata

# OSM editors sometimes type the Icelandic eth (U+00D0, U+00F0) instead of the
# Vietnamese D with stroke (U+0110, U+0111). They look identical but compare unequal.
LOOK_ALIKE_CHARACTERS: dict[str, str] = {
    "Ð": "Đ",
    "ð": "đ",
}

STREET_PREFIX_PATTERN = re.compile(r"^đường\s+")
ALLEY_PREFIX = "hẻm "
WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9]+")


def clean_name(raw_name: str) -> str:
    """Return the display form of a name: NFC, look-alikes fixed, single spaces."""
    name = unicodedata.normalize("NFC", raw_name)
    for look_alike, replacement in LOOK_ALIKE_CHARACTERS.items():
        name = name.replace(look_alike, replacement)
    return WHITESPACE_PATTERN.sub(" ", name).strip()


def name_key(raw_name: str) -> str:
    """Return the key that decides whether two spellings are the same name.

    The key is lower case and drops a leading "Đường", so "Đường Cô Giang" and
    "Cô Giang" share a key, as do "Đường Số 11" and "Đường số 11".
    """
    lowered = unicodedata.normalize("NFC", clean_name(raw_name).lower())
    return STREET_PREFIX_PATTERN.sub("", lowered)


def ascii_key(raw_name: str) -> str:
    """Return the name key without diacritics, for lookups only."""
    decomposed = unicodedata.normalize("NFD", name_key(raw_name))
    without_marks = "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )
    return without_marks.replace("đ", "d")


def slugify(raw_name: str) -> str:
    """Return an identifier made of ASCII letters, digits and hyphens."""
    return NON_ALPHANUMERIC_PATTERN.sub("-", ascii_key(raw_name)).strip("-")


def is_alley_name(raw_name: str) -> bool:
    """Return True if the name is an alley ("Hẻm ...")."""
    return name_key(raw_name).startswith(ALLEY_PREFIX)
