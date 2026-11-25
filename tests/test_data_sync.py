from pathlib import Path

import pytest

from graph_llm.data_sync import (
    GraphConfigKind,
    build_manifest,
    graph_config_kind,
    json_equal_up_to_float_rounding,
    list_data_files,
    verify_manifest,
)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_manifest_lists_data_files_and_skips_placeholders(tmp_path: Path) -> None:
    write(tmp_path / "data/raw/area/graph.graphml", "graph")
    write(tmp_path / "data/raw/.gitkeep", "")
    write(tmp_path / "runs/run1/metrics.json", "{}")

    files = list_data_files(tmp_path, ["data/raw", "runs", "missing"])

    assert files == [Path("data/raw/area/graph.graphml"), Path("runs/run1/metrics.json")]


def test_verify_reports_missing_and_changed_files_only(tmp_path: Path) -> None:
    write(tmp_path / "data/raw/a.txt", "a")
    write(tmp_path / "data/raw/b.txt", "b")
    manifest = build_manifest(tmp_path, ["data/raw"], "remote:test")

    assert verify_manifest(tmp_path, manifest) == []

    (tmp_path / "data/raw/a.txt").unlink()
    write(tmp_path / "data/raw/b.txt", "changed")
    write(tmp_path / "data/raw/new.txt", "newer data is allowed")

    assert verify_manifest(tmp_path, manifest) == [
        "missing: data/raw/a.txt",
        "changed: data/raw/b.txt",
    ]


def test_every_committed_graph_config_has_a_known_kind() -> None:
    kinds = {
        path.name: graph_config_kind(path) for path in sorted(Path("configs/graphs").glob("*.yaml"))
    }

    assert kinds["hcmc_district1_v2_p1.yaml"] is GraphConfigKind.PROCESSED_GRAPH_V1
    assert kinds["hcmc_district1_v3_p2.yaml"] is GraphConfigKind.REFERENCE_GRAPH
    assert kinds["hcmc_district1_v3_p2_review.yaml"] is GraphConfigKind.REVIEW_LOG


def test_unknown_graph_config_is_rejected_not_skipped(tmp_path: Path) -> None:
    write(tmp_path / "unknown.yaml", "some_other_key: 1\n")

    with pytest.raises(ValueError, match="not a known kind"):
        graph_config_kind(tmp_path / "unknown.yaml")


def test_last_digit_float_difference_counts_as_equal() -> None:
    stored = [{"junction_id": "j1", "spread_m": 6.510835709338412, "node_ids": [1, 2]}]
    rebuilt = [{"junction_id": "j1", "spread_m": 6.510835709338413, "node_ids": [1, 2]}]

    assert json_equal_up_to_float_rounding(stored, rebuilt) is True


@pytest.mark.parametrize(
    "rebuilt",
    [
        [{"junction_id": "j1", "spread_m": 6.52, "node_ids": [1, 2]}],
        [{"junction_id": "j1", "spread_m": 6.510835709338412, "node_ids": [1, 3]}],
        [{"junction_id": "j2", "spread_m": 6.510835709338412, "node_ids": [1, 2]}],
        [{"junction_id": "j1", "spread_m": 6.510835709338412}],
        [],
    ],
)
def test_real_differences_are_not_hidden_by_the_tolerance(rebuilt: object) -> None:
    stored = [{"junction_id": "j1", "spread_m": 6.510835709338412, "node_ids": [1, 2]}]

    assert json_equal_up_to_float_rounding(stored, rebuilt) is False
