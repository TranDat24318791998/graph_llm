"""Upload the git-ignored data to the rclone remote and write data/data_manifest.json.

Run on the machine that produced the data, then commit data/data_manifest.json so
the other machine knows exactly which files belong with this code version.

Usage:
    uv run python scripts/push_data.py
    uv run python scripts/push_data.py --remote gdrive:graph_llm_data --directories data/raw
"""

import argparse
from pathlib import Path

from graph_llm.data_sync import DEFAULT_REMOTE, MANIFEST_PATH, SHARED_DIRECTORIES, push_data


def main() -> None:
    """Parse arguments, upload the data and write the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--directories", nargs="+", default=SHARED_DIRECTORIES)
    arguments = parser.parse_args()

    manifest = push_data(Path("."), arguments.directories, arguments.remote)
    total_mb = sum(entry.size_bytes for entry in manifest.files) / 1e6
    print(f"Uploaded {len(manifest.files)} files ({total_mb:.1f} MB) to {arguments.remote}")
    print(f"Wrote {MANIFEST_PATH}; commit it so other machines can verify their copy.")


if __name__ == "__main__":
    main()
