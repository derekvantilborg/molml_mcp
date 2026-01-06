import pytest
from pathlib import Path


def test_a(session_workdir, request):
    d = session_workdir / request.node.name
    d.mkdir(exist_ok=True)
    (d / "out.txt").write_text("A")
    assert (d / "out.txt").read_text() == "A"
    