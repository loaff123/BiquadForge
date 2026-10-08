"""Source package metadata and generated release-content contract."""
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_runtime_dependencies_compiler_optional():
    text=(ROOT/"pyproject.toml").read_text()
    assert 'dependencies = ["numpy>=1.24", "scipy>=1.10"]' in text
    assert 'requires-python = ">=3.10"' in text
    assert 'cmsisdsp' not in text
    assert (ROOT/"LICENSE").exists() and (ROOT/"THIRD_PARTY_NOTICES.md").exists()


def test_build_distribution_contents(tmp_path):
    pytest.importorskip("build",reason="Optional distribution test requires python-build development dependency")
    result=subprocess.run([sys.executable,"-m","build","--no-isolation","--outdir",str(tmp_path)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr[-12000:]
    wheel=next(tmp_path.glob("*.whl"));sdist=next(tmp_path.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as z:
        names=z.namelist()
        assert "biquadforge/reference_hashes.json" in names
        assert not any("tests/" in n or "internal" in n or "PLAN.md" in n or n.endswith(".so") for n in names)
        assert any(n.endswith("LICENSE") for n in names)
        metadata=z.read(next(n for n in names if n.endswith("METADATA"))).decode()
        assert "Requires-Dist: numpy" in metadata and "Requires-Dist: scipy" in metadata
        assert "Requires-Dist: cmsisdsp" not in metadata
    with tarfile.open(sdist) as t:
        names=t.getnames()
        assert any(n.endswith("tests/reference/cmsisdsp-1.10.3/LICENSE") for n in names)
        assert any(n.endswith("arm_biquad_cascade_df1_q15.c") for n in names)
        assert any(n.endswith("examples/reject_issue127.json") for n in names)
        assert not any("internal" in n and "internal_residual.json" not in n for n in names)


def test_ci_is_read_only_and_uses_fixed_official_actions():
    workflow=(ROOT/".github/workflows/ci.yml").read_text()
    assert 'on: [push, pull_request]' in workflow
    assert 'permissions:\n  contents: read' in workflow
    assert 'persist-credentials: false' in workflow
    assert 'actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683' in workflow
    assert 'actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065' in workflow
