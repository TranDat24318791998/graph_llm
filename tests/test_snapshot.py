from pathlib import Path

import pytest
from pydantic import ValidationError

from graph_llm.graph.snapshot import compute_sha256, download_snapshot, load_study_area
from graph_llm.schemas.graph import BoundingBox

AREA_CONFIG_PATH = Path("configs/areas/hcmc_district1.yaml")


def test_study_area_config_is_valid() -> None:
    area = load_study_area(AREA_CONFIG_PATH)

    assert area.area_id == "hcmc_district1"
    assert area.network_type == "drive"


def test_bounding_box_with_swapped_corners_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(west=106.71, south=10.75, east=106.68, north=10.79)


def test_existing_snapshot_is_never_overwritten(tmp_path: Path) -> None:
    area = load_study_area(AREA_CONFIG_PATH)
    (tmp_path / area.area_id).mkdir()

    with pytest.raises(FileExistsError):
        download_snapshot(area, tmp_path)


def test_sha256_of_known_content(tmp_path: Path) -> None:
    file_path = tmp_path / "content.txt"
    file_path.write_bytes(b"abc")

    assert compute_sha256(file_path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
