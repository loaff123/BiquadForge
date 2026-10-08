import hashlib
import json
from pathlib import Path
import pytest
from biquadforge import load_spec, qualify
from biquadforge.pack import write_pack
from biquadforge.spec import BiquadForgeError

@pytest.fixture
def result(specfile, minimal):
    minimal["suite"] = {"excitation_samples":8,"settling_samples":16,"tail_samples":8}
    return qualify(load_spec(specfile(minimal)))


def test_pack_files_deterministic_and_hashes(tmp_path, result):
    a=write_pack(result,tmp_path/"a"); b=write_pack(result,tmp_path/"b")
    names={p.name for p in a.iterdir()}
    assert names == {"filter.h","filter.c","manifest.json","report.md","vectors.json","vectors.h","replay.c","README.md"}
    for name in names:
        assert (a/name).read_bytes() == (b/name).read_bytes()
        assert str(tmp_path).encode() not in (a/name).read_bytes()
    manifest=json.loads((a/"manifest.json").read_text())
    for name,digest in manifest["artifact_sha256"].items():
        assert hashlib.sha256((a/name).read_bytes()).hexdigest()==digest
    assert manifest["verification"]["status"]=="not_requested"
    assert "arm_biquad_cascade_df1_q15" in (a/"filter.c").read_text()
    assert "arm_biquad_cascade_df1_init_q15" in (a/"filter.c").read_text()
    assert "state[4]" in (a/"filter.h").read_text()
    assert "bqf_demo_context" in (a/"filter.h").read_text()
    data=json.loads((a/"vectors.json").read_text())
    assert len(data["schedules"])==7 and all("final_state" in v for v in data["vectors"])

@pytest.mark.parametrize("kind",["empty","nonempty","file","symlink"])
def test_preexisting_destination_preserved(tmp_path,result,kind):
    out=tmp_path/"out"
    if kind in ("empty","nonempty"):
        out.mkdir()
        if kind=="nonempty": (out/"mine").write_text("keep")
    elif kind=="file": out.write_text("keep")
    else: out.symlink_to(tmp_path/"absent")
    with pytest.raises(BiquadForgeError): write_pack(result,out)
    if kind=="file": assert out.read_text()=="keep"
    if kind=="nonempty": assert (out/"mine").read_text()=="keep"
    if kind=="symlink": assert out.is_symlink()


def test_source_alias_preserved(specfile,result):
    path=specfile(); before=path.read_bytes()
    with pytest.raises(BiquadForgeError): write_pack(result,path.parent/"."/path.name)
    assert path.read_bytes()==before

@pytest.mark.parametrize("fail_call",[1,2,5,11,16])
def test_handled_interruption_never_looks_complete(tmp_path,result,monkeypatch,fail_call):
    import biquadforge.pack as pack
    original=pack._write_exclusive; calls=0
    def fail(path,data):
        nonlocal calls
        calls+=1
        if calls==fail_call: raise OSError("injected interruption")
        original(path,data)
    monkeypatch.setattr(pack,"_write_exclusive",fail)
    out=tmp_path/"out"
    with pytest.raises(BiquadForgeError): write_pack(result,out)
    assert (out/"INCOMPLETE").exists() or not (out/"manifest.json").exists()


def test_rejection_has_no_deployment_sources(tmp_path,specfile,minimal):
    minimal["sos"]=[[1,0,0,1,-1.01,0]]
    p=write_pack(qualify(load_spec(specfile(minimal))),tmp_path/"reject")
    assert {x.name for x in p.iterdir()}=={"manifest.json","report.md","README.md"}


def test_internal_residual_is_prominent(tmp_path,specfile,minimal):
    minimal["suite"]={"excitation_samples":8,"settling_samples":64,"tail_samples":8}
    minimal["sos"]=[[.5,0,0,1,-.5,0],[0,0,0,1,0,0]]
    minimal["search"]={"candidate_budget":1,"post_shifts":[0]}
    p=write_pack(qualify(load_spec(specfile(minimal))),tmp_path/"internal")
    report=(p/"report.md").read_text()
    assert report.index("observed_nonzero_stage_tail") < report.index("Search")
    assert "nonzero_periodic" in report


def test_rejected_report_never_claims_quiet_tail(tmp_path,specfile,minimal):
    minimal["suite"]={"excitation_samples":8,"settling_samples":64,"tail_samples":8}
    minimal["sos"]=[[.5,0,0,1,-.5,0]]
    minimal["limits"]["max_final_tail_q15"]=0
    p=write_pack(qualify(load_spec(specfile(minimal))),tmp_path/"tailreject")
    text=(p/"report.md").read_text()
    assert "No nonzero stage tail was observed" not in text
    assert "nonzero_periodic" in text


def test_final_marker_removal_failure_stays_incomplete(tmp_path,result,monkeypatch):
    original=Path.unlink
    def fail_marker(path,*args,**kwargs):
        if path.name=="INCOMPLETE":raise OSError("injected finalization failure")
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,"unlink",fail_marker)
    out=tmp_path/"finalize"
    with pytest.raises(BiquadForgeError):write_pack(result,out)
    assert (out/"manifest.json").exists() and (out/"INCOMPLETE").exists()
