"""Explicit, optional host-C replay against a pinned official-source snapshot."""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import numpy as np
from scipy import signal
from .spec import BiquadForgeError, Vector, load_spec, parse_json, samples_digest, validate_name
from .candidates import Descriptor, CandidateReject, generate_candidates, quantize_candidate
from .fixedpoint import simulate_batch
from .qualification import BuildResult, _metric, _evaluate, analyze_tail, qualify
from .pack import SCHEDULES, canonical_json, render_c, render_pack

FLAGS = ["-std=c11", "-O2", "-D__GNUC_PYTHON__", "-DARM_MATH_AUTOVECTORIZE"]
KERNEL = "Source/FilteringFunctions/arm_biquad_cascade_df1_q15.c"
INIT = "Source/FilteringFunctions/arm_biquad_cascade_df1_init_q15.c"


@dataclass(frozen=True)
class VerificationResult:
    status: str
    evidence: dict

    def as_dict(self):
        return {"status": self.status, "evidence": self.evidence}


def reference_provenance():
    return json.loads(files("biquadforge").joinpath("reference_hashes.json").read_text())


def _read(path, limit=128*1024*1024):
    with path.open("rb") as f:
        raw=f.read(limit+1)
    if len(raw)>limit:
        raise ValueError("Pack file exceeds verification admission limit")
    return raw


def _hash(value):
    return hashlib.sha256(value).hexdigest()


def _digest(value):
    if not isinstance(value,str) or not re.fullmatch("[0-9a-f]{64}",value):
        raise ValueError("Invalid digest")
    return value


def _validate_pack(pack):
    """Do not execute any pack-provided code: regenerate it from checked numbers."""
    if (pack/"INCOMPLETE").exists() or (pack/"INCOMPLETE").is_symlink():
        raise ValueError("Pack has an INCOMPLETE marker")
    manifest_bytes=_read(pack/"manifest.json",32*1024*1024)
    manifest=parse_json(manifest_bytes)
    vectors_data=parse_json(_read(pack/"vectors.json"))
    if not isinstance(manifest,dict) or not isinstance(vectors_data,dict):
        raise ValueError("Manifest and vector roots must be JSON objects")
    for field in ("resolved_spec","selected","search","tool"):
        if not isinstance(manifest.get(field),dict):
            raise ValueError(f"Manifest {field} must be an object")
    tool_record=manifest["tool"]
    if set(tool_record)!={"name","version","python","numpy","scipy"} or tool_record["name"]!="biquadforge" or tool_record["version"]!="0.1.0":
        raise ValueError("Unsupported tool identity")
    if any(not isinstance(tool_record[k],str) or not re.fullmatch(r"[0-9][A-Za-z0-9.+_-]{0,79}",tool_record[k]) for k in ("python","numpy","scipy")):
        raise ValueError("Invalid declared numeric-backend version")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"]!=1:
        raise ValueError("Unsupported manifest schema")
    if manifest.get("status") not in ("accepted_for_suite","accepted_for_suite_with_observations"):
        raise ValueError("A diagnostic/rejected pack cannot be verified as deployment")
    if vectors_data.get("schema_version")!=1 or vectors_data.get("schedules")!=SCHEDULES:
        raise ValueError("Invalid replay format/schedules")
    saved=manifest["resolved_spec"]; selected=manifest["selected"]
    if not isinstance(saved.get("vectors"),list) or not isinstance(selected.get("prepared"),dict) or not isinstance(selected["prepared"].get("descriptor"),dict):
        raise ValueError("Invalid nested manifest containers")
    spec_data={"schema_version":1,"name":saved["name"],"sample_rate_hz":saved["sample_rate_hz"],
               "sos":saved["original_sos"],"suite":saved["suite"],"limits":saved["limits"],
               "search":saved["search"],"grid":{"points":saved["grid_points"]}}
    with tempfile.TemporaryDirectory(prefix="biquadforge-validate-") as td:
        path=Path(td)/"spec.json";path.write_bytes(canonical_json(spec_data)); spec=load_spec(path)
    if saved["normalized_sos"]!=[list(v) for v in spec.normalized_sos]:
        raise ValueError("Normalized SOS is inconsistent")
    values=vectors_data["vectors"];metadata=saved["vectors"]
    if not isinstance(values,list) or not 5<=len(values)<=21 or len(values)!=len(metadata):
        raise ValueError("Invalid realized suite")
    original_names=[v.name for v in spec.vectors]
    if [v["name"] for v in values[:5]]!=original_names:
        raise ValueError("Mandatory signed/noise suite is missing")
    new_vectors=[];names=set();total=0
    for i,(v,meta) in enumerate(zip(values,metadata)):
        name=validate_name(v["name"])
        if name in names or name!=meta["name"]: raise ValueError("Invalid vector names")
        names.add(name)
        samples=v["input"];total+=len(samples)
        if not isinstance(samples,list) or not samples or total>2_000_000:
            raise ValueError("Invalid input sample count")
        if any(type(x) is not int or not -32768<=x<=32767 for x in samples):
            raise ValueError("Invalid Q15 input")
        silence=spec.suite.settling_samples+spec.suite.tail_samples
        if len(samples)<=silence or any(samples[-silence:]):
            raise ValueError("Stored vector lacks the specified zero tail")
        if i<5:
            if len(samples)!=spec.suite.excitation_samples+silence:
                raise ValueError("Mandatory vector length mismatch")
            if i<4 and samples!=list(spec.vectors[i].samples):
                raise ValueError("Mandatory signed excitation changed")
            if i==4 and any(abs(x)>spec.suite.amplitude_q15 for x in samples):
                raise ValueError("Stored noise exceeds specified amplitude")
        elif len(samples)-silence>65536:
            raise ValueError("Custom vector exceeds admission limit")
        if meta["encoding"] not in ("q15","normalized_float") or v["encoding"]!=meta["encoding"]:
            raise ValueError("Invalid encoding")
        clips=meta["clipped_input_count"]
        if type(clips) is not int or not 0<=clips<=len(samples)-silence or v["clipped_input_count"]!=clips:
            raise ValueError("Invalid clipping count")
        if _digest(meta["source_sha256"])!=v["source_sha256"]:raise ValueError("Source digest mismatch")
        if samples_digest(samples)!=v["input_sha256"]:raise ValueError("Input digest mismatch")
        new_vectors.append(Vector(name,tuple(samples),meta["encoding"],meta["source_sha256"],clips))
    spec=replace(spec,vectors=tuple(new_vectors),source_sha256=_digest(saved["source_sha256"]))
    # Stored realized noise is authoritative for replay; future PRNG versions need not reproduce it.
    descriptor=selected["prepared"]["descriptor"]
    d=Descriptor(tuple(descriptor["order"]),tuple(descriptor["exponents"]),descriptor["post_shift"])
    if any(type(v) is not int for v in (*d.order,*d.exponents,d.post_shift)):
        raise ValueError("Noninteger candidate descriptor")
    cs=generate_candidates(spec)
    ordinal=selected["ordinal"]
    if type(ordinal) is not int or not 0<=ordinal<len(cs.descriptors) or cs.descriptors[ordinal]!=d:
        raise ValueError("Selected descriptor not in declared search")
    audit=manifest["search"]
    evaluated=min(len(cs.descriptors),spec.search.candidate_budget)
    if audit["planned_total"]!=len(cs.descriptors) or audit["evaluated"]!=evaluated or audit["remaining"]!=len(cs.descriptors)-evaluated or audit["complete"]!=(evaluated==len(cs.descriptors)) or ordinal>=evaluated or audit["descriptor_sha256"]!=cs.digest or audit["descriptors"]!=json.loads(canonical_json([asdict(x) for x in cs.descriptors])):
        raise ValueError("Search accounting/descriptor mismatch")
    p=quantize_candidate(spec,d)
    if isinstance(p,CandidateReject) or json.loads(canonical_json(asdict(p)))!=selected["prepared"]:
        raise ValueError("Quantized candidate differs from normalized source transformation")
    metrics=[];recomputed=[];observations=[]
    if not audit["complete"]:observations.append("search_truncated")
    if spec.limits.max_final_tail_q15>1:observations.append("explicit_relaxed_final_tail_limit")
    if any(v.clipped_input_count for v in spec.vectors):observations.append("input_conversion_clipped")
    if any(all(c[j] == 0 for j in (0,2,3)) and any(spec.normalized_sos[index][j] != 0 for j in (0,1,2)) for c,index in zip(p.coefficients,d.order)):
        observations.append("quantized_zero_numerator")
    for vector,old in zip(spec.vectors,values):
        x=np.array(vector.samples,dtype=np.int16)
        truth=signal.sosfilt(np.array(spec.normalized_sos),x.astype(float)/32768)
        if not np.all(np.isfinite(truth)):raise ValueError("Nonfinite reference")
        sim=simulate_batch(np.array([p.coefficients]),[d.post_shift],x,spec.suite.tail_samples,True)
        m=_metric(sim.outputs[0],truth,sim,0,vector.name);metrics.append(m)
        tail=analyze_tail(x,sim.stage_outputs[0],spec.suite.tail_samples)
        if tail["nonzero_tail_stages"] and "observed_nonzero_stage_tail" not in observations:observations.append("observed_nonzero_stage_tail")
        if tail["cycle"]["status"]=="nonzero_periodic" and "observed_nonzero_periodic_state" not in observations:observations.append("observed_nonzero_periodic_state")
        fresh={"name":vector.name,"input":x.tolist(),"output":sim.outputs[0].tolist(),"input_sha256":samples_digest(x),"output_sha256":samples_digest(sim.outputs[0]),"final_state":sim.states[0].tolist(),"source_sha256":vector.source_sha256,"encoding":vector.encoding,"clipped_input_count":vector.clipped_input_count,"metrics":m,"tail":tail}
        if fresh!=old:raise ValueError("Stored finite output, state, metric or tail is inconsistent")
        recomputed.append(fresh)
    ev=_evaluate(p,ordinal,metrics,spec.limits)
    if ev.failures or json.loads(canonical_json(asdict(ev)))!=selected:
        raise ValueError("Selected candidate does not pass the stored finite requirements")
    status="accepted_for_suite_with_observations" if observations else "accepted_for_suite"
    if manifest["observations"]!=observations or manifest["status"]!=status:
        raise ValueError("Residual or other prominent observation was removed")
    expected=render_c(spec.name,p.coefficients,d.post_shift,recomputed)
    expected["vectors.json"]=canonical_json({"schema_version":1,"schedules":SCHEDULES,"vectors":recomputed})
    expected_keys={*expected,"README.md"}
    if set(manifest["artifact_sha256"])!=expected_keys:raise ValueError("Unexpected artifact names")
    for filename,data in expected.items():
        if _read(pack/filename)!=data or manifest["artifact_sha256"][filename]!=_hash(data):
            raise ValueError(f"Generated artifact differs: {filename}")
    if _hash(_read(pack/"README.md"))!=manifest["artifact_sha256"]["README.md"]:
        raise ValueError("README digest mismatch")
    # All user-facing assurances and audit facts must be reproduced, not trusted.
    # This deliberately repeats the bounded search; its latency is an explicit tradeoff.
    recomputed_result=qualify(spec)
    canonical_files=render_pack(recomputed_result)
    canonical_manifest=parse_json(canonical_files["manifest.json"])
    canonical_manifest["tool"]=tool_record  # Declared provenance, not authenticated history.
    if canonical_json(manifest)!=canonical_json(canonical_manifest):
        raise ValueError("Manifest claims differ from full bounded qualification")
    for filename,data in canonical_files.items():
        if filename!="manifest.json" and _read(pack/filename)!=data:
            raise ValueError(f"Canonical artifact differs: {filename}")
    return expected,spec,manifest,_hash(manifest_bytes)


def _record(pack, result):
    fd,path=tempfile.mkstemp(prefix=".verification-",suffix=".json",dir=pack)
    try:
        with os.fdopen(fd,"wb") as f:f.write(canonical_json(result.as_dict()))
        os.replace(path,pack/"verification.json")
    finally:
        if os.path.exists(path):os.unlink(path)
    return result


def verify_pack(pack: str | Path, cmsis_root: str | Path, compiler: str="cc") -> VerificationResult:
    pack,root=Path(pack),Path(cmsis_root)
    unverified={"host_c":"unverified","arm_hardware":"unverified"}
    try:
        generated,spec,manifest,manifest_digest=_validate_pack(pack)
    except (OSError,ValueError,TypeError,KeyError,IndexError,OverflowError) as e:
        invalid=VerificationResult("invalid_pack",{**unverified,"reason":str(e)})
        if pack.is_dir():
            try:
                return _record(pack,invalid)
            except OSError as record_error:
                return VerificationResult("invalid_pack",{**invalid.evidence,"evidence_persisted":False,"record_error":str(record_error)})
        return invalid
    provenance=reference_provenance();snapshots={}
    try:
        for relative,digest in provenance["files"].items():
            data=_read(root/relative,2*1024*1024)
            if _hash(data)!=digest:raise ValueError("Pinned source hash mismatch: "+relative)
            snapshots[relative]=data
    except (OSError,ValueError) as e:
        return _record(pack,VerificationResult("source_mismatch",{**unverified,"reason":str(e)}))
    executable=shutil.which(compiler)
    if executable is None:
        return _record(pack,VerificationResult("compiler_unavailable",{**unverified,"compiler_requested":compiler}))
    executable=str(Path(executable).resolve())
    # Copy only checked bytes. Unlisted or subsequently changed local headers cannot enter compilation.
    with tempfile.TemporaryDirectory(prefix="biquadforge-replay-") as td:
        work=Path(td); source=work/"cmsis"
        for rel,data in snapshots.items():
            dest=source/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        for name,data in generated.items():
            (work/name).write_bytes(data)
        binary=work/("replay.exe" if os.name=="nt" else "replay")
        args=[executable,*FLAGS,"-I",str(source/"Include"),"-I",str(source/"PrivateInclude"),str(work/"filter.c"),str(work/"replay.c"),str(source/KERNEL),str(source/INIT),"-o",str(binary)]
        env={"PATH":os.defpath,"LANG":"C","LC_ALL":"C"}
        if "SystemRoot" in os.environ:env["SystemRoot"]=os.environ["SystemRoot"]
        try:
            compile_run=subprocess.run(args,cwd=work,env=env,capture_output=True,text=True,timeout=120)
            if compile_run.returncode:
                return _record(pack,VerificationResult("compile_failed",{**unverified,"returncode":compile_run.returncode,"stderr":compile_run.stderr[-12000:]}))
        except (OSError,subprocess.TimeoutExpired) as e:
            return _record(pack,VerificationResult("compile_failed",{**unverified,"reason":str(e)}))
        try:
            replay=subprocess.run([str(binary)],cwd=work,env=env,capture_output=True,text=True,timeout=120)
            if replay.returncode:raise ValueError(f"Replay exit {replay.returncode}: {replay.stderr[-12000:]}")
            summary=json.loads(replay.stdout)
            expected={"status":"verified","vectors":len(spec.vectors),"schedules":7,"sample_comparisons":7*sum(len(v.samples) for v in spec.vectors),"state_comparisons":7*len(spec.vectors)*4*len(spec.normalized_sos)}
            if summary!=expected:raise ValueError("Replay summary differs from expected comparison counts")
        except (OSError,subprocess.TimeoutExpired,ValueError) as e:
            return _record(pack,VerificationResult("replay_failed",{**unverified,"reason":str(e)}))
        try:
            version=subprocess.run([executable,"--version"],env=env,capture_output=True,text=True,timeout=10).stdout.splitlines()[0]
        except (OSError,subprocess.TimeoutExpired,IndexError):version="unavailable"
        evidence={"host_c":"verified_standard_scalar_only","arm_hardware":"unverified","source":provenance,
                  "compiler":executable,"compiler_version":version,"flags":FLAGS,"source_snapshot":"only pinned checked files copied into isolated compile directory",
                  "replay":summary,"schedules":SCHEDULES,"stdout_sha256":_hash(replay.stdout.encode()),
                  "manifest_sha256":manifest_digest,"artifact_sha256":manifest["artifact_sha256"],
                  "validation_scope":"Full declared bounded-search qualification, canonical manifest/report/README, regenerated code, exact outputs and final states. Stored realized inputs are authoritative; backend/source metadata are declared provenance, not authentication."}
    return _record(pack,VerificationResult("verified",evidence))
