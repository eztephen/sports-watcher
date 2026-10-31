import pytest

from sports_watcher.ingest.base import SourceSpec
from sports_watcher.ingest.factory import open_source, register_source
from sports_watcher.ingest.file import FileSource
from tests.helpers.sources import StubSource


def test_factory_opens_file_and_custom_kind(sample_video):
    src = open_source(SourceSpec(kind="file", path=str(sample_video)))
    assert isinstance(src, FileSource)
    src.close()
    register_source("stub", lambda spec: StubSource())
    stub = open_source(SourceSpec(kind="stub"))
    assert isinstance(stub, StubSource) and stub.opened


def test_factory_rejects_missing_fields():
    with pytest.raises(ValueError):
        open_source(SourceSpec(kind="file"))
    with pytest.raises(ValueError):
        open_source(SourceSpec(kind="url"))
