from pathlib import Path

from graph_llm.data_sync import build_manifest, list_data_files, verify_manifest


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
