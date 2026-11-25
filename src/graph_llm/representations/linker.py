"""Find street names in text and resolve predicted names to streets.

Matching uses name keys (NFC, lower case, no leading "Đường"), so spelling
variants and aliases of the street layer are recognized. Resolving a single
predicted name also accepts a match without diacritics, because an LLM may drop
them; linking inside a question never does, to avoid false matches.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

from graph_llm.names.normalize import ascii_key, name_key
from graph_llm.questions.graph_view import GraphView
from graph_llm.schemas.street import StreetCategory


@dataclass(frozen=True)
class StreetLinker:
    """Lookups from surface forms to street ids."""

    ids_by_key: dict[str, set[str]]
    ids_by_ascii_key: dict[str, set[str]]
    patterns: list[tuple[re.Pattern[str], set[str]]]

    @classmethod
    def from_view(
        cls, view: GraphView, categories: frozenset[StreetCategory] | None = None
    ) -> StreetLinker:
        """Collect every display name, OSM spelling and alias of the streets.

        With ``categories`` only streets of those categories are linked; ``None``
        links every street.
        """
        ids_by_key: dict[str, set[str]] = defaultdict(set)
        ids_by_ascii_key: dict[str, set[str]] = defaultdict(set)
        for street in view.streets.values():
            if categories is not None and street.category not in categories:
                continue
            for form in [street.display_name, *street.osm_names, *street.aliases]:
                ids_by_key[name_key(form)].add(street.street_id)
                ids_by_ascii_key[ascii_key(form)].add(street.street_id)
        keys = sorted(ids_by_key, key=len, reverse=True)
        patterns = [
            (re.compile(rf"(?<!\w){re.escape(key)}(?!\w)"), ids_by_key[key]) for key in keys
        ]
        return cls(dict(ids_by_key), dict(ids_by_ascii_key), patterns)

    def link(self, text: str) -> set[str]:
        """Return the streets named in a text. Longer names win over names inside them."""
        remaining = name_key(text)
        found: set[str] = set()
        for pattern, street_ids in self.patterns:
            if pattern.search(remaining):
                found |= street_ids
                remaining = pattern.sub(" ", remaining)
        return found

    def resolve(self, name: str) -> set[str]:
        """Return the streets a single predicted name can stand for (empty if unknown)."""
        exact = self.ids_by_key.get(name_key(name))
        if exact:
            return set(exact)
        return set(self.ids_by_ascii_key.get(ascii_key(name), set()))
