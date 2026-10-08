import json
import os
import subprocess
import sys


def cli(*args):
    return subprocess.run([sys.executable,"-m","biquadforge",*map(str,args)],capture_output=True,text=True,env={**os.environ,"PYTHONPATH":"src"})


def test_build_statuses(tmp_path,specfile,minimal):
    minimal["suite"]={"excitation_samples":4,"settling_samples":8,"tail_samples":4}
    p=specfile(minimal);r=cli("build",p,"--out",tmp_path/"yes")
    assert r.returncode==0 and json.loads(r.stdout)["status"]=="accepted_for_suite"
    minimal["sos"]=[[1,0,0,1,0,0]];minimal["search"]={"post_shifts":[0]}
    r=cli("build",specfile(minimal),"--out",tmp_path/"no")
    assert r.returncode==2 and json.loads(r.stdout)["status"]=="rejected"
    minimal["search"]={"candidate_budget":1}
    r=cli("build",specfile(minimal),"--out",tmp_path/"budget")
    assert r.returncode==3
    minimal["search"]={"post_shifts":[14]}
    r=cli("build",specfile(minimal),"--out",tmp_path/"unsupported")
    assert r.returncode==4 and json.loads(r.stderr)["status"]=="unsupported"
    r=cli("build",tmp_path/"absent","--out",tmp_path/"error")
    assert r.returncode==1 and "Traceback" not in r.stderr


def test_verify_malformed_root_no_traceback(tmp_path):
    pack=tmp_path/"malformed";pack.mkdir()
    (pack/"manifest.json").write_text("[]")
    (pack/"vectors.json").write_text("{}")
    r=cli("verify",pack,"--cmsis-root",tmp_path/"missing-source")
    assert r.returncode==6
    assert json.loads(r.stdout)["status"]=="invalid_pack"
    assert "Traceback" not in r.stderr
