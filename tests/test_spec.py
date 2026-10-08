import json
import dataclasses
import numpy as np
import pytest
from biquadforge.spec import load_spec, BiquadForgeError


def test_materialized_defaults(specfile):
    s = load_spec(specfile())
    assert s.limits.max_final_tail_q15 == 1
    assert s.search.post_shifts == (0, 1, 2, 3)
    assert len(s.vectors) == 5
    assert {v.name for v in s.vectors} == {"positive_impulse", "negative_impulse", "positive_step", "negative_step", "noise"}
    assert all(len(v.samples) == 10240 for v in s.vectors)
    assert s.normalized_sos == ((0.5, 0., 0., 1., 0., 0.),)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.name = "change"

@pytest.mark.parametrize("change,status", [
    ({"unknown": 3}, "error"), ({"schema_version": 2}, "unsupported"),
    ({"schema_version": True}, "error"), ({"sample_rate_hz": float("nan")}, "error"),
    ({"sample_rate_hz": 0}, "error"), ({"name": "for"}, "error"),
    ({"name": "__a"}, "error"), ({"sos": [[1, 0, 0, 0, 0, 0]]}, "error"),
    ({"sos": [[1, 0, 0, 1, 0, 0]] * 5}, "unsupported"),
    ({"search": {"post_shifts": [14]}}, "unsupported"),
    ({"search": {"post_shifts": [0, 0]}}, "error"),
    ({"suite": {"seed": True}}, "error"),
    ({"sos": [[1e308, 0, 0, 1e-308, 0, 0]]}, "error"),
    ({"sos": [[1e-308, 0, 0, 1e308, 0, 0]]}, "error"),
    ({"limits": {"max_grid_error": .1}}, "error"),
])
def test_invalid_fields(specfile, minimal, change, status):
    minimal.update(change)
    with pytest.raises(BiquadForgeError) as e:
        load_spec(specfile(minimal))
    assert e.value.status == status


def test_duplicates_and_nonfinite_json(tmp_path):
    p = tmp_path / "x.json"
    for text in ['{"schema_version":1,"schema_version":1}', '{"x":NaN}']:
        p.write_text(text)
        with pytest.raises(BiquadForgeError):
            load_spec(p)


def test_vectors_float_ties_clipping_and_csv(specfile, minimal, tmp_path):
    (tmp_path / "v.json").write_text(json.dumps([.5/32768, 1.5/32768, -.5/32768, -1.5/32768, 2.0, -2.0]))
    (tmp_path / "v.csv").write_text("-32768\n0\n32767\n")
    minimal["suite"] = {"vectors": [{"name": "custom", "path": "v.json", "format": "json", "encoding": "normalized_float"}, {"name": "csv", "path": "v.csv", "format": "csv", "encoding": "q15"}]}
    s = load_spec(specfile(minimal))
    a, b = s.vectors[-2:]
    assert a.samples[:6] == (0, 2, 0, -2, 32767, -32768)
    assert a.clipped_input_count == 2
    assert b.samples[:3] == (-32768, 0, 32767)
    assert len(a.source_sha256) == 64
    assert a.samples[-1024:] == (0,) * 1024

@pytest.mark.parametrize("contents,fmt", [('1,2\n', 'csv'), ('1.0\n', 'csv'), ('[1.0]', 'json'), ('[true]', 'json'), ('[32768]', 'json')])
def test_bad_q15_vectors(specfile, minimal, tmp_path, contents, fmt):
    (tmp_path / "v").write_text(contents)
    minimal["suite"] = {"vectors": [{"name": "custom", "path": "v", "format": fmt, "encoding": "q15"}]}
    with pytest.raises(BiquadForgeError):
        load_spec(specfile(minimal))


def test_sample_budget_before_large_allocations(specfile, minimal):
    minimal["suite"] = {"excitation_samples": 16384, "settling_samples": 65536, "tail_samples": 4096}
    assert len(load_spec(specfile(minimal)).vectors[0].samples) == 86016


def test_seed_repeatable(specfile):
    assert load_spec(specfile()).vectors == load_spec(specfile()).vectors


def test_float_clipping_counts_after_rounding(specfile, minimal, tmp_path):
    (tmp_path / "v.json").write_text(json.dumps([-1-0.25/32768, -1-0.75/32768]))
    minimal["suite"] = {"vectors": [{"name": "edges", "path": "v.json", "format": "json", "encoding": "normalized_float"}]}
    v = load_spec(specfile(minimal)).vectors[-1]
    assert v.samples[:2] == (-32768,-32768)
    assert v.clipped_input_count == 1


def test_realized_suite_over_two_million_is_unsupported(specfile,minimal,tmp_path):
    (tmp_path/"long.json").write_text(json.dumps([0]*65536))
    minimal["suite"]={"excitation_samples":16384,"settling_samples":65536,"tail_samples":4096,
                      "vectors":[{"name":f"v{i}","path":"long.json","format":"json","encoding":"q15"} for i in range(16)]}
    with pytest.raises(BiquadForgeError) as e:load_spec(specfile(minimal))
    assert e.value.status=="unsupported" and e.value.field=="suite"


def test_oversized_input_file_is_unsupported(tmp_path):
    p=tmp_path/"huge.json"
    with p.open("wb") as f:f.truncate(16*1024*1024+1)
    with pytest.raises(BiquadForgeError) as e:load_spec(p)
    assert e.value.status=="unsupported"
