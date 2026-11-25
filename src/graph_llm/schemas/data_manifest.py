"""Schema of the manifest that lists every git-ignored data file shared between machines."""

from datetime import datetime

from pydantic import Field

from graph_llm.schemas.base import StrictModel


class DataFileEntry(StrictModel):
    """One file of the shared data, by its path relative to the repository root."""

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str


class DataManifest(StrictModel):
    """The exact set of data files that belongs with a version of the code.

    The manifest is committed to git; the files themselves are stored on a
    remote (Google Drive through rclone) and checked against it after download.
    """

    created_at: datetime
    created_on_host: str
    remote: str
    directories: list[str]
    files: list[DataFileEntry]
