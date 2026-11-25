"""Share the git-ignored data between machines through an rclone remote.

Code and the benchmark travel through git. The OSM snapshot, the processed
graph, the text corpus, the LightRAG index, run outputs and the LLM cache do
not; they are copied to an rclone remote (Google Drive) and described by a
manifest (`data/data_manifest.json`) that is committed to git. A machine that
pulls the data checks every file against the manifest, so it knows it has
exactly the data the code version expects.

Uploads use ``rclone copy``, never ``rclone sync``: nothing on the remote is
ever deleted, which matches the rule that frozen data is never overwritten.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from graph_llm.graph.processed import (
    load_processed_graph,
    load_processing_config,
    process_snapshot,
    save_processed_graph,
)
from graph_llm.graph.snapshot import compute_sha256, load_snapshot
from graph_llm.schemas.data_manifest import DataFileEntry, DataManifest

DEFAULT_REMOTE = "gdrive:graph_llm_data"
MANIFEST_PATH = Path("data/data_manifest.json")
SHARED_DIRECTORIES = ["data/raw", "data/processed", "runs", ".cache"]
IGNORED_FILE_NAMES = {".gitkeep", ".DS_Store"}


def list_data_files(root: Path, directories: list[str]) -> list[Path]:
    """Return every shared file under the given directories, relative to ``root``, sorted."""
    files: list[Path] = []
    for directory in directories:
        base = root / directory
        if not base.exists():
            continue
        files += [
            path.relative_to(root)
            for path in base.rglob("*")
            if path.is_file() and path.name not in IGNORED_FILE_NAMES
        ]
    return sorted(files)


def build_manifest(root: Path, directories: list[str], remote: str) -> DataManifest:
    """Describe the current data files by size and SHA-256."""
    entries = [
        DataFileEntry(
            path=path.as_posix(),
            size_bytes=(root / path).stat().st_size,
            sha256=compute_sha256(root / path),
        )
        for path in list_data_files(root, directories)
    ]
    return DataManifest(
        created_at=datetime.now(UTC),
        created_on_host=socket.gethostname(),
        remote=remote,
        directories=directories,
        files=entries,
    )


def verify_manifest(root: Path, manifest: DataManifest) -> list[str]:
    """Return one problem per missing or changed file; an empty list means all files match.

    Files that exist locally but are not in the manifest are not problems: a
    machine may hold newer data, such as runs made after the manifest was written.
    """
    problems = []
    for entry in manifest.files:
        path = root / entry.path
        if not path.exists():
            problems.append(f"missing: {entry.path}")
        elif path.stat().st_size != entry.size_bytes or compute_sha256(path) != entry.sha256:
            problems.append(f"changed: {entry.path}")
    return problems


def rclone_executable() -> str:
    """Return the path of rclone, looking in ~/bin as well as PATH."""
    found = shutil.which("rclone") or shutil.which("rclone", path=str(Path.home() / "bin"))
    if found is None:
        raise FileNotFoundError(
            "rclone is not installed. Install it into ~/bin from https://rclone.org/downloads/ "
            "and configure the remote with 'rclone config'."
        )
    return found


def rclone_copy(source: str, destination: str) -> None:
    """Copy new and changed files from source to destination, comparing checksums."""
    subprocess.run(
        [rclone_executable(), "copy", source, destination, "--checksum", "--stats-one-line"],
        check=True,
    )


def push_data(root: Path, directories: list[str], remote: str) -> DataManifest:
    """Upload the data directories to the remote and write the manifest locally.

    Every file currently on disk is listed. The manifest itself is uploaded too,
    and should also be committed to git.
    """
    manifest = build_manifest(root, directories, remote)
    for directory in manifest.directories:
        if (root / directory).exists():
            rclone_copy(str(root / directory), f"{remote}/{directory}")
    (root / MANIFEST_PATH).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    rclone_copy(str(root / MANIFEST_PATH.parent / MANIFEST_PATH.name), f"{remote}/data")
    return manifest


def pull_data(root: Path, manifest: DataManifest, remote: str) -> list[str]:
    """Download the data directories of a manifest and return the verification problems."""
    for directory in manifest.directories:
        rclone_copy(f"{remote}/{directory}", str(root / directory))
    return verify_manifest(root, manifest)


def check_rebuild_matches(
    raw_data_directory: Path,
    processed_data_directory: Path,
    config_path: Path,
    work_directory: Path,
) -> list[str]:
    """Rebuild a processed graph from its snapshot and compare it with the stored one.

    The rebuilt graph is written under ``work_directory``, never over the stored
    one. Returns one problem per file whose checksum differs.
    """
    config = load_processing_config(config_path)
    stored = load_processed_graph(processed_data_directory, config.graph_version).metadata
    snapshot_graph, snapshot_metadata = load_snapshot(raw_data_directory, config.source_area_id)
    rebuilt = save_processed_graph(
        process_snapshot(snapshot_graph, config), config, snapshot_metadata, work_directory
    )
    problems = []
    if rebuilt.graphml_sha256 != stored.graphml_sha256:
        problems.append(f"{config.graph_version}: rebuilt graph differs from the stored graph")
    if rebuilt.streets_sha256 != stored.streets_sha256:
        problems.append(f"{config.graph_version}: rebuilt streets differ from the stored streets")
    return problems
