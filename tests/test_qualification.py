import dataclasses
import numpy as np
import pytest
from biquadforge.spec import load_spec
from biquadforge.qualification import qualify, analyze_tail


def quick(minimal):
    minimal["suite"] = {"excitation_samples": 32, "settling_samples": 128, "tail_samples": 32}
    return minimal


def test_strict_quiet_fir_acceptance(specfile, minimal):
    r = qualify(load_spec(specfile(quick(minimal))))
    assert r.status == "accepted_for_suite"
    assert r.search["complete"] and r.search["evaluated"] == r.search["planned_total"] == 4
    assert r.chosen.prepared.descriptor.post_shift == 0
    assert r.chosen.max_abs_error <= 1/32768
    assert len(r.replay_vectors) == 5
    assert all(v["metrics"]["tail_peaks_q15"] == [0] for v in r.replay_vectors)
    assert all(v["tail"]["cycle"]["status"] == "zero_fixed_point" for v in r.replay_vectors)


def test_budget_outcomes(specfile, minimal):
    quick(minimal); minimal["search"] = {"candidate_budget": 1}
    r = qualify(load_spec(specfile(minimal)))
    assert r.status == "accepted_for_suite_with_observations"
    assert "search_truncated" in r.observations
    minimal["sos"][0][0] = 1
    r = qualify(load_spec(specfile(minimal)))
    assert r.status == "search_budget_exhausted"
    assert r.search["evaluated"] == 1 and r.search["remaining"] == 3
    assert r.search["failure_counts"]["coefficient_range"] == 1
    minimal["search"] = {"post_shifts": [0]}
    assert qualify(load_spec(specfile(minimal))).status == "rejected"


def test_internal_residual_not_hidden_by_quiet_output(specfile, minimal):
    quick(minimal)
    minimal["sos"] = [[.5,0,0,1,-.5,0], [0,0,0,1,0,0]]
    minimal["search"] = {"candidate_budget": 1, "post_shifts": [0]}
    r = qualify(load_spec(specfile(minimal)))
    assert r.status == "accepted_for_suite_with_observations"
    neg = next(v for v in r.replay_vectors if v["name"] == "negative_impulse")
    assert neg["metrics"]["tail_peaks_q15"] == [1,0]
    assert neg["tail"]["cycle"]["status"] == "nonzero_periodic"
    assert neg["tail"]["cycle"]["period_samples"] == 1
    assert "observed_nonzero_stage_tail" in r.observations
    minimal["limits"]["max_internal_tail_q15"] = 0
    rr = qualify(load_spec(specfile(minimal)))
    assert rr.chosen is None
    assert rr.search["failure_counts"]["internal_tail"] == 1


def test_issue127_rejects_default_strict_tail(specfile, minimal):
    minimal["sos"] = [[.0009566029866080141,.0019132059732160282,.0009566029866080141,1.,-1.9352943868599919,.9391207988064236]]
    minimal["limits"].update(max_grid_error=.1,max_abs_error=.1,max_rmse=.1)
    r = qualify(load_spec(specfile(minimal)))
    assert r.status == "rejected" and r.chosen is None
    assert r.search["failure_counts"]["final_tail"] >= 1
    assert r.diagnostics and r.diagnostics[0].max_final_tail_q15 > 1


def test_error_and_rmse_limits_per_vector(specfile, minimal):
    quick(minimal); minimal["limits"]["max_abs_error"] = 0
    r = qualify(load_spec(specfile(minimal)))
    assert r.status == "rejected" and r.search["failure_counts"]["max_abs_error"] > 0
    minimal["limits"]["max_abs_error"] = .01
    minimal["limits"]["max_rmse"] = 0
    r = qualify(load_spec(specfile(minimal)))
    assert r.search["failure_counts"]["rmse"] > 0


def test_source_rejection(specfile, minimal):
    minimal["sos"] = [[1,0,0,1,-1.01,0]]
    r = qualify(load_spec(specfile(minimal)))
    assert r.status == "rejected" and r.search["evaluated"] == 0
    assert "source_linear_poles_not_strictly_stable" in r.observations


def test_tail_no_repeat_inconclusive():
    stages = np.array([[4,3,2,1]],dtype=np.int16)
    tail = analyze_tail(np.zeros(4,dtype=np.int16), stages, 3)
    assert tail["cycle"]["status"] == "inconclusive"
    assert tail["nonzero_tail_stages"] == [1]


def test_batch_size_does_not_change_selected(specfile, minimal):
    s = load_spec(specfile(quick(minimal)))
    a = qualify(s); b = qualify(dataclasses.replace(s,search=dataclasses.replace(s.search,batch_size=1)))
    assert a.chosen == b.chosen and a.replay_vectors == b.replay_vectors


def test_clipping_and_relaxed_tail_prominent(specfile, minimal, tmp_path):
    quick(minimal); (tmp_path/"v.json").write_text("[2, -2]")
    minimal["suite"]["vectors"] = [{"name":"clipped","path":"v.json","format":"json","encoding":"normalized_float"}]
    minimal["limits"]["max_final_tail_q15"] = 2
    r = qualify(load_spec(specfile(minimal)))
    assert "input_conversion_clipped" in r.observations
    assert "explicit_relaxed_final_tail_limit" in r.observations
    v = r.replay_vectors[-1]
    assert v["input"][:2] == [32767,-32768]
    assert v["metrics"]["max_abs_error"] <= 1/32768


def test_rejected_diagnostic_keeps_exact_cycle_evidence(specfile,minimal):
    minimal["suite"]={"excitation_samples":8,"settling_samples":64,"tail_samples":8}
    minimal["sos"]=[[.5,0,0,1,-.5,0]]
    minimal["limits"]["max_final_tail_q15"]=0
    r=qualify(load_spec(specfile(minimal)))
    assert r.status=="rejected"
    assert r.diagnostic_tails
    assert any(v["tail"]["cycle"]["status"]=="nonzero_periodic" for d in r.diagnostic_tails for v in d["vectors"])


def test_quantized_away_nonzero_numerator_disclosed(specfile,minimal):
    minimal["suite"]={"excitation_samples":4,"settling_samples":8,"tail_samples":4}
    minimal["sos"]=[[1e-10,0,0,1,0,0]]
    r=qualify(load_spec(specfile(minimal)))
    assert r.status=="accepted_for_suite_with_observations"
    assert "quantized_zero_numerator" in r.observations
