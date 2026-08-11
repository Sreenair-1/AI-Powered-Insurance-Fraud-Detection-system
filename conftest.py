"""Test-suite settings for sandboxed local runs."""

import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest


_PROJECT_ROOT = Path(__file__).resolve().parent
_TEMP_ROOT = _PROJECT_ROOT / "pytest_runtime_tmp"
_TEMP_ROOT.mkdir(parents=True, exist_ok=True)

for _name in ("TMPDIR", "TEMP", "TMP"):
    os.environ[_name] = str(_TEMP_ROOT)

tempfile.tempdir = str(_TEMP_ROOT)

_OriginalTemporaryDirectory = tempfile.TemporaryDirectory


def _safe_rmtree(cls, name, ignore_errors=False, repeated=False):
    shutil.rmtree(name, ignore_errors=True)


_OriginalTemporaryDirectory._rmtree = classmethod(_safe_rmtree)


class _PermissionTolerantTemporaryDirectory(_OriginalTemporaryDirectory):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ignore_cleanup_errors", True)
        super().__init__(*args, **kwargs)


tempfile.TemporaryDirectory = _PermissionTolerantTemporaryDirectory


@pytest.fixture
def tmp_path(request):
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name).strip("_")
    path = _TEMP_ROOT / "cases" / f"{safe_name}_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path
