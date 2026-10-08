"""Fresh independent-review regressions; each reproduced before the fix pass."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
from biquadforge import Q15Cascade, load_spec, qualify, write_pack, verify_pack
from biquadforge.spec import BiquadForgeError
from biquadforge.candidates import generate_candidates, quantize_candidate

ROOT=(Path(__file__).parent/"reference"/"cmsisdsp-1.10.3").resolve()

@pytest.fixture
def pack(tmp_path,specfile,minimal):
    minimal["suite"]={"excitation_samples":8,"settling_samples":32,"tail_samples":8}
    return write_pack(qualify(load_spec(specfile(minimal))),tmp_path/"pack")

@pytest.mark.parametrize("kind",["report","readme","limitations","verification","count","plausible_count","diagnostics"])
def test_r1_forged_assurances_refused_before_compile(pack,monkeypatch,kind):
    m=json.loads((pack/"manifest.json").read_text())
    if kind=="report":(pack/"report.md").write_text("Quiet; safe for hardware.\n")
    elif kind=="readme":
        (pack/"README.md").write_text("Hardware verified.\n")
        m["artifact_sha256"]["README.md"]=hashlib.sha256((pack/"README.md").read_bytes()).hexdigest()
    elif kind=="limitations":m["limitations"]=[]
    elif kind=="verification":m["verification"]={"status":"verified","arm_hardware":"verified"}
    elif kind=="count":m["search"]["accepted_count"]=1000000
    elif kind=="plausible_count":m["search"]["accepted_count"]-=1
    else:m["diagnostic_tails"]=[{"ordinal":0,"vectors":[]}]
    (pack/"manifest.json").write_text(json.dumps(m))
    monkeypatch.setattr("biquadforge.verify.subprocess.run",lambda *a,**k:pytest.fail("Compiler must not run on forged assurances"))
    assert verify_pack(pack,ROOT).status=="invalid_pack"


def test_r1_nonbest_claimed_selection_refused(tmp_path,specfile,minimal,monkeypatch):
    minimal["suite"]={"excitation_samples":8,"settling_samples":32,"tail_samples":8}
    spec=load_spec(specfile(minimal));r=qualify(spec)
    p=quantize_candidate(spec,generate_candidates(spec).descriptors[1])
    forged=replace(r,chosen=replace(r.chosen,prepared=p,ordinal=1))
    pack=write_pack(forged,tmp_path/"notbest")
    monkeypatch.setattr("biquadforge.verify.subprocess.run",lambda *a,**k:pytest.fail("Compiler must not run on false selection claim"))
    assert verify_pack(pack,ROOT).status=="invalid_pack"

@pytest.mark.parametrize("token",["1e-400","-1e-400","1e-9999","-1e-9999"])
def test_r2_nonzero_lexical_underflow_rejected(specfile,minimal,token):
    p=specfile(minimal);p.write_text(p.read_text().replace("0.5",token))
    with pytest.raises(BiquadForgeError):load_spec(p)

@pytest.mark.parametrize("token,expected",[("0e-400",0.0),("-0e-400",0.0),("5e-324",np.nextafter(0.,1.))])
def test_r2_exact_zero_and_representable_subnormal_preserved(specfile,minimal,token,expected):
    p=specfile(minimal);p.write_text(p.read_text().replace("0.5",token))
    assert load_spec(p).original_sos[0][0]==expected


def test_r2_csv_underflow_rejected(specfile,minimal,tmp_path):
    (tmp_path/"v.csv").write_text("1e-400\n")
    minimal["suite"]={"vectors":[{"name":"tiny","path":"v.csv","format":"csv","encoding":"normalized_float"}]}
    with pytest.raises(BiquadForgeError):load_spec(specfile(minimal))


def test_r3_invalid_recheck_replaces_stale_success(pack):
    if not shutil.which("cc"):pytest.skip("Host compiler unavailable")
    assert verify_pack(pack,ROOT).status=="verified"
    (pack/"filter.c").write_text((pack/"filter.c").read_text()+"\nchanged\n")
    assert verify_pack(pack,ROOT).status=="invalid_pack"
    assert json.loads((pack/"verification.json").read_text())["status"]=="invalid_pack"

@pytest.mark.parametrize("filename",["manifest.json","vectors.json"])
@pytest.mark.parametrize("value",[[],None,"bad",17])
def test_r4_malformed_json_roots_structured(pack,monkeypatch,filename,value):
    (pack/filename).write_text(json.dumps(value))
    monkeypatch.setattr("biquadforge.verify.subprocess.run",lambda *a,**k:pytest.fail("Compiler must not run"))
    assert verify_pack(pack,ROOT).status=="invalid_pack"

@pytest.mark.parametrize("field",["resolved_spec","selected","search","tool"])
def test_r4_nested_malformed_containers_structured(pack,monkeypatch,field):
    m=json.loads((pack/"manifest.json").read_text());m[field]=[]
    (pack/"manifest.json").write_text(json.dumps(m))
    monkeypatch.setattr("biquadforge.verify.subprocess.run",lambda *a,**k:pytest.fail("Compiler must not run"))
    assert verify_pack(pack,ROOT).status=="invalid_pack"

@pytest.mark.parametrize("bad",[[True,1],[1,np.bool_(False)],[np.array(True),1],np.array([],dtype=float),np.array([],dtype=bool)])
def test_r5_mixed_booleans_rejected_without_state_mutation(bad):
    q=Q15Cascade([[16384,0,0,0,8192,0]],0);q.process([2,4]);before=q.state
    with pytest.raises(BiquadForgeError):q.process(bad)
    assert np.array_equal(q.state,before)

@pytest.mark.parametrize("boolean",[True,np.bool_(True),np.array(True)])
def test_r5_mixed_boolean_coefficients_rejected(boolean):
    with pytest.raises(BiquadForgeError):Q15Cascade([[boolean,0,0,0,0,0]],0)


def test_r6_relative_compiler_path_with_spaces(pack,tmp_path,monkeypatch):
    cc=shutil.which("cc")
    if not cc:pytest.skip("Host compiler unavailable")
    local=tmp_path/"my cc";local.symlink_to(Path(cc).resolve())
    monkeypatch.chdir(tmp_path)
    r=verify_pack(pack,ROOT,"./my cc")
    assert r.status=="verified",r.as_dict()
    assert Path(r.evidence["compiler"]).is_absolute()


def test_r3_success_binds_to_validated_manifest(pack):
    if not shutil.which("cc"):pytest.skip("Host compiler unavailable")
    assert verify_pack(pack,ROOT).status=="verified"
    initial=json.loads((pack/"verification.json").read_text())
    assert initial["evidence"]["manifest_sha256"]==hashlib.sha256((pack/"manifest.json").read_bytes()).hexdigest()
