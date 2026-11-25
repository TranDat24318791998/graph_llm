"""Set up this machine from git plus the shared data on the rclone remote.

Steps:
  1. Read data/data_manifest.json (from git) and check free disk space.
  2. Download the data directories from the remote and verify every file's SHA-256.
  3. Optionally rebuild every graph in configs/graphs/ (processed graph v1 and
     reference graph v2) from its snapshot and check that it is identical to the
     downloaded one (--verify-rebuild).
  4. Optionally check that an LLM endpoint answers (--llm-config).
  5. Run the test suite.

Run 'uv sync' first. The script only writes inside this repository.

Usage:
    uv run python scripts/setup_environment.py
    uv run python scripts/setup_environment.py --verify-rebuild \
        --llm-config configs/llm/vllm_server_qwen3_32b.yaml
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from graph_llm.data_sync import MANIFEST_PATH, check_graph_rebuild, pull_data
from graph_llm.llm.client import LLMClient
from graph_llm.schemas.data_manifest import DataManifest
from graph_llm.schemas.llm import LLMEndpointConfig

DISK_SPACE_MARGIN = 2.0


def check_disk_space(manifest: DataManifest) -> None:
    """Stop if free disk space is below twice the size of the data."""
    needed = sum(entry.size_bytes for entry in manifest.files) * DISK_SPACE_MARGIN
    free = shutil.disk_usage(".").free
    print(f"Data size {needed / DISK_SPACE_MARGIN / 1e6:.1f} MB, free disk {free / 1e9:.1f} GB")
    if free < needed:
        raise SystemExit("Not enough free disk space.")


def check_llm_endpoint(config_path: Path) -> None:
    """Send one short chat request to the configured endpoint, bypassing the response cache."""
    config = LLMEndpointConfig.model_validate(yaml.safe_load(config_path.read_text("utf-8")))
    with tempfile.TemporaryDirectory() as cache_directory:
        client = LLMClient(config, cache_path=Path(cache_directory) / "check.sqlite")
        result = client.chat([{"role": "user", "content": "Trả lời một từ: xin chào"}])
    print(f"LLM endpoint {config.base_url} ({config.model}) answered: {result.text.strip()[:60]}")


def main() -> None:
    """Parse arguments and run the setup steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default=None, help="Defaults to the remote in the manifest.")
    parser.add_argument("--verify-rebuild", action="store_true")
    parser.add_argument("--llm-config", type=Path, default=None)
    arguments = parser.parse_args()

    manifest = DataManifest.model_validate_json(MANIFEST_PATH.read_text(encoding="utf-8"))
    print(f"Manifest: {len(manifest.files)} files, written on {manifest.created_on_host}")
    check_disk_space(manifest)

    remote = arguments.remote or manifest.remote
    problems = pull_data(Path("."), manifest, remote)
    if problems:
        print("\n".join(problems))
        raise SystemExit(f"{len(problems)} data files do not match the manifest.")
    print("All data files match the manifest.")

    if arguments.verify_rebuild:
        for config_path in sorted(Path("configs/graphs").glob("*.yaml")):
            with tempfile.TemporaryDirectory() as work_directory:
                rebuild_check = check_graph_rebuild(
                    Path("data/raw"), Path("data/processed"), config_path, Path(work_directory)
                )
            if rebuild_check is None:
                print(f"Skipped {config_path.name}: a review log, not a graph config.")
                continue
            if rebuild_check.problems:
                raise SystemExit("\n".join(rebuild_check.problems))
            if rebuild_check.notes:
                print(f"Rebuilt {config_path.name}: equal to the downloaded graph, with notes:")
                for note in rebuild_check.notes:
                    print(f"  {note}")
            else:
                print(f"Rebuilt {config_path.name}: identical to the downloaded graph.")

    if arguments.llm_config is not None:
        check_llm_endpoint(arguments.llm_config)

    subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)
    print("Setup complete.")


if __name__ == "__main__":
    main()
