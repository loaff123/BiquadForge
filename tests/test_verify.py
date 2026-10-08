import hashlib
import json
from pathlib import Path
import shutil
import pytest
from biquadforge import load_spec,qualify,write_pack
from biquadforge.verify import verify_pack

ROOT=Path(__file__).parent/"reference"/"cmsisdsp-1.10.3"

@pytest.fixture
def pack(tmp_path,specfile,minimal):
    minimal["suite"]={"excitation_samples":16,"settling_samples":32,"tail_samples":16}
    return write_pack(qualify(load_spec(specfile(minimal))),tmp_path/"pack")


def test_actual_official_c_replay(pack):
    if not shutil.which("cc"): pytest.skip("Host C compiler unavailable")
    r=verify_pack(pack,ROOT)
    assert r.status=="verified",r.as_dict()
    assert r.evidence["replay"]["vectors"]==5
    assert r.evidence["replay"]["schedules"]==7
    assert r.evidence["replay"]["state_comparisons"]==5*7*4
    assert json.loads((pack/"verification.json").read_text())["status"]=="verified"


def test_compiler_unavailable(pack):
    r=verify_pack(pack,ROOT,"definitely-no-biquadforge-compiler")
    assert r.status=="compiler_unavailable"
    assert r.evidence["host_c"]=="unverified"

@pytest.mark.parametrize("target",["filter.c","vectors.h","replay.c","vectors.json","manifest.json"])
def test_modified_pack_refused_before_compiler(pack,monkeypatch,target):
    p=pack/target
    if target=="manifest.json":
        m=json.loads(p.read_text());m["schema_version"]=99;p.write_text(json.dumps(m))
    elif target=="vectors.json":
        m=json.loads(p.read_text());m["vectors"][0]["output"][0]+=1;p.write_text(json.dumps(m))
    else: p.write_text(p.read_text()+"\nmalicious extra C\n")
    # Attack updates the easily forgeable hash too; regeneration must still refuse.
    mp=pack/"manifest.json";m=json.loads(mp.read_text())
    m["artifact_sha256"][target]=hashlib.sha256(p.read_bytes()).hexdigest();mp.write_text(json.dumps(m))
    monkeypatch.setattr("biquadforge.verify.subprocess.run",lambda *a,**k:pytest.fail("Compiler must not run"))
    assert verify_pack(pack,ROOT).status=="invalid_pack"


def test_incomplete_pack_refused(pack,monkeypatch):
    (pack/"INCOMPLETE").write_text("interrupted")
    monkeypatch.setattr("biquadforge.verify.subprocess.run",lambda *a,**k:pytest.fail("Compiler must not run"))
    assert verify_pack(pack,ROOT).status=="invalid_pack"


def test_transitive_header_hash(pack,tmp_path,monkeypatch):
    source=tmp_path/"cmsis";shutil.copytree(ROOT,source)
    (source/"Include"/"arm_math_memory.h").write_text("untrusted")
    monkeypatch.setattr("biquadforge.verify.subprocess.run",lambda *a,**k:pytest.fail("Compiler must not run"))
    assert verify_pack(pack,source).status=="source_mismatch"


def test_unlisted_shadow_header_never_compiled(pack,tmp_path):
    if not shutil.which("cc"): pytest.skip("Host C compiler unavailable")
    source=tmp_path/"cmsis";shutil.copytree(ROOT,source)
    (source/"Include"/"stdint.h").write_text('#error "Untrusted shadow header must not be copied"\n')
    assert verify_pack(pack,source).status=="verified"


def test_compile_failure_separate_from_unavailable(pack,monkeypatch):
    import subprocess
    def fail(*args,**kwargs):
        return subprocess.CompletedProcess(args[0],1,"","controlled compiler failure")
    monkeypatch.setattr("biquadforge.verify.subprocess.run",fail)
    assert verify_pack(pack,ROOT).status=="compile_failed"


def test_replay_failure_has_own_status(pack,monkeypatch):
    if not shutil.which("cc"): pytest.skip("Host C compiler unavailable")
    import subprocess
    real=subprocess.run
    def wrapped(args,**kwargs):
        if Path(args[0]).name in ("replay","replay.exe"):
            return subprocess.CompletedProcess(args,1,"","controlled replay failure")
        return real(args,**kwargs)
    monkeypatch.setattr("biquadforge.verify.subprocess.run",wrapped)
    assert verify_pack(pack,ROOT).status=="replay_failed"
