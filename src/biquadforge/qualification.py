"""Selection and reports are explicitly restricted to the realized finite suite."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict, dataclass
import numpy as np
from scipy import signal
from .spec import BuildSpec, BiquadForgeError, samples_digest
from .candidates import CandidateReject, PreparedCandidate, generate_candidates, quantize_candidate, reference_response
from .fixedpoint import simulate_batch


@dataclass(frozen=True)
class Evaluation:
    prepared: PreparedCandidate
    ordinal: int
    metrics: tuple[dict, ...]
    failures: tuple[str, ...]
    max_rmse: float
    max_abs_error: float
    saturations: int
    max_final_tail_q15: int
    max_internal_tail_q15: int

    @property
    def rank(self):
        return (self.max_rmse, self.max_abs_error, self.prepared.grid_error, self.saturations, self.ordinal)


@dataclass(frozen=True)
class BuildResult:
    status: str
    spec: BuildSpec
    search: dict
    chosen: Evaluation | None
    diagnostics: tuple[Evaluation, ...]
    observations: tuple[str, ...]
    replay_vectors: tuple[dict, ...]
    diagnostic_tails: tuple[dict, ...] = ()


def analyze_tail(inputs, stage_outputs, tail_samples):
    """Full-state repeats under zero input prove a periodic continuation, nothing global."""
    x, stages = np.asarray(inputs), np.asarray(stage_outputs)
    start = len(x)-tail_samples
    if start < 1 or np.any(x[start:] != 0):
        raise BiquadForgeError("Tail analysis requires zero input and preceding state", "tail")
    peaks = np.max(np.abs(stages[:,start:].astype(np.int64)),axis=1)
    seen = {}
    cycle = {"status": "inconclusive"}
    for t in range(start,len(x)):
        state = []
        for k in range(len(stages)):
            ins = x if k == 0 else stages[k-1]
            out = stages[k]
            state.extend([int(ins[t]),int(ins[t-1]),int(out[t]),int(out[t-1])])
        key = tuple(state)
        if key in seen:
            cycle = {"status": "nonzero_periodic" if any(state) else "zero_fixed_point",
                     "first_sample_index": seen[key], "repeat_sample_index": t,
                     "period_samples": t-seen[key], "state": state}
            break
        seen[key] = t
    return {"nonzero_tail_stages": [int(i+1) for i,p in enumerate(peaks) if p != 0],
            "cycle": cycle,
            "scope": "Exact full integer state within this finite zero-input tail; absence of a repeat is inconclusive."}


def _metric(output, truth, batch, i, name):
    error = output.astype(np.float64)/32768 - truth
    return {"name": name, "max_abs_error": float(np.max(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error*error))),
            "stage_saturations": batch.saturations[i].tolist(),
            "peak_preclamp_q15": batch.peaks[i].tolist(),
            "tail_peaks_q15": batch.tail_peaks[i].tolist(),
            "final_state": batch.states[i].tolist()}


def _evaluate(prepared, ordinal, metrics, limits):
    rmse = max(m["rmse"] for m in metrics)
    maximum = max(m["max_abs_error"] for m in metrics)
    sats = sum(sum(m["stage_saturations"]) for m in metrics)
    final = max(m["tail_peaks_q15"][-1] for m in metrics)
    internal = max(max(m["tail_peaks_q15"]) for m in metrics)
    fail = []
    if prepared.grid_error > limits.max_grid_error: fail.append("grid_error")
    if maximum > limits.max_abs_error: fail.append("max_abs_error")
    if rmse > limits.max_rmse: fail.append("rmse")
    if sats > limits.max_saturations: fail.append("saturations")
    if final > limits.max_final_tail_q15: fail.append("final_tail")
    if limits.max_internal_tail_q15 is not None and internal > limits.max_internal_tail_q15: fail.append("internal_tail")
    return Evaluation(prepared,ordinal,tuple(metrics),tuple(fail),rmse,maximum,sats,final,internal)


def qualify(spec: BuildSpec) -> BuildResult:
    cs = generate_candidates(spec)
    planned = len(cs.descriptors)
    count = min(planned,spec.search.candidate_budget)
    search = {"planned_total": planned, "evaluated": count, "remaining": planned-count,
              "budget": spec.search.candidate_budget, "complete": count == planned,
              "descriptor_sha256": cs.digest,
              "descriptors": [asdict(d) for d in cs.descriptors],
              "center_observations": list(cs.center_observations),
              "representable_stable": 0, "accepted_count": 0, "failure_counts": {},
              "early_rejection_examples": []}
    if cs.source_reasons:
        search["failure_counts"] = {r:1 for r in cs.source_reasons}
        return BuildResult("rejected",spec,search,None,(),cs.source_reasons,())
    original = reference_response(spec)
    inputs = [np.array(v.samples,dtype=np.int16) for v in spec.vectors]
    truths = []
    for x in inputs:
        with np.errstate(all="ignore"):
            truth = signal.sosfilt(np.array(spec.normalized_sos),x.astype(np.float64)/32768)
        if not np.all(np.isfinite(truth)):
            raise BiquadForgeError("Original finite-vector reference is nonfinite", "sos")
        truths.append(truth)
    failure_counts = Counter()
    best = None; diagnostics = []
    prepared_buffer = []
    def run_batch(buffer):
        nonlocal best, diagnostics
        coefficients = np.array([p.coefficients for _,p in buffer],dtype=np.int16)
        shifts = [p.descriptor.post_shift for _,p in buffer]
        metrics = [[] for _ in buffer]
        for vector,x,truth in zip(spec.vectors,inputs,truths):
            sim = simulate_batch(coefficients,shifts,x,spec.suite.tail_samples)
            for i in range(len(buffer)):
                metrics[i].append(_metric(sim.outputs[i],truth,sim,i,vector.name))
        for (ordinal,p),m in zip(buffer,metrics):
            ev = _evaluate(p,ordinal,m,spec.limits)
            if ev.failures:
                failure_counts.update(ev.failures)
                diagnostics.append(ev)
                diagnostics.sort(key=lambda e:(len(e.failures), e.max_rmse, e.max_abs_error, e.ordinal))
                diagnostics = diagnostics[:3]
            else:
                search["accepted_count"] += 1
                if best is None or ev.rank < best.rank:
                    best = ev
    for ordinal,d in enumerate(cs.descriptors[:count]):
        p = quantize_candidate(spec,d,original)
        if isinstance(p,CandidateReject):
            reason = p.reason
        else:
            search["representable_stable"] += 1
            reason = "grid_error" if p.grid_error > spec.limits.max_grid_error else None
        if reason:
            failure_counts[reason] += 1
            if len(search["early_rejection_examples"]) < 12:
                search["early_rejection_examples"].append({"ordinal":ordinal,"descriptor":asdict(d),"reason":reason})
            continue
        prepared_buffer.append((ordinal,p))
        if len(prepared_buffer) == spec.search.batch_size:
            run_batch(prepared_buffer); prepared_buffer = []
    if prepared_buffer:
        run_batch(prepared_buffer)
    search["failure_counts"] = dict(sorted(failure_counts.items()))
    observations = []
    if not search["complete"]: observations.append("search_truncated")
    if spec.limits.max_final_tail_q15 > 1: observations.append("explicit_relaxed_final_tail_limit")
    if any(v.clipped_input_count for v in spec.vectors): observations.append("input_conversion_clipped")
    # Enrich every retained full diagnostic with exact finite-tail evidence too.
    replay = []
    if best is not None:
        p = best.prepared
        if any(all(c[j] == 0 for j in (0,2,3)) and any(spec.normalized_sos[index][j] != 0 for j in (0,1,2)) for c,index in zip(p.coefficients,p.descriptor.order)):
            observations.append("quantized_zero_numerator")
        for vector,x,truth in zip(spec.vectors,inputs,truths):
            sim = simulate_batch(np.array([p.coefficients]),[p.descriptor.post_shift],x,spec.suite.tail_samples,True)
            metric = _metric(sim.outputs[0],truth,sim,0,vector.name)
            tail = analyze_tail(x,sim.stage_outputs[0],spec.suite.tail_samples)
            if tail["nonzero_tail_stages"] and "observed_nonzero_stage_tail" not in observations:
                observations.append("observed_nonzero_stage_tail")
            if tail["cycle"]["status"] == "nonzero_periodic" and "observed_nonzero_periodic_state" not in observations:
                observations.append("observed_nonzero_periodic_state")
            replay.append({"name":vector.name,"input":x.tolist(),"output":sim.outputs[0].tolist(),
                           "input_sha256":samples_digest(x),"output_sha256":samples_digest(sim.outputs[0]),
                           "final_state":sim.states[0].tolist(),"source_sha256":vector.source_sha256,
                           "encoding":vector.encoding,"clipped_input_count":vector.clipped_input_count,
                           "metrics":metric,"tail":tail})
        status = "accepted_for_suite_with_observations" if observations else "accepted_for_suite"
    else:
        status = "rejected" if search["complete"] else "search_budget_exhausted"
    diagnostic_tails = []
    for diagnostic in diagnostics:
        dp = diagnostic.prepared
        details = []
        for vector, x in zip(spec.vectors, inputs):
            sim = simulate_batch(np.array([dp.coefficients]), [dp.descriptor.post_shift], x, spec.suite.tail_samples, True)
            details.append({"name": vector.name, "tail": analyze_tail(x, sim.stage_outputs[0], spec.suite.tail_samples),
                            "tail_peaks_q15": sim.tail_peaks[0].tolist(), "final_state": sim.states[0].tolist()})
        diagnostic_tails.append({"ordinal": diagnostic.ordinal, "vectors": details})
    return BuildResult(status,spec,search,best,tuple(diagnostics),tuple(observations),tuple(replay),tuple(diagnostic_tails))
