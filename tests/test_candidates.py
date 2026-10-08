import dataclasses
import numpy as np
import pytest
from scipy import signal
from biquadforge.candidates import generate_candidates, quantize_candidate, Descriptor, jury
from biquadforge.spec import load_spec


def test_original_first_and_single_section_dedup(specfile):
    s = load_spec(specfile()); cs = generate_candidates(s)
    assert len(cs.descriptors) == 4
    assert cs.descriptors[0] == Descriptor((0,), (0,), 0)
    assert cs.descriptors[-1].post_shift == 3
    assert len(cs.digest) == 64
    assert not cs.source_reasons


def test_coefficient_ties_even_and_layout(specfile, minimal):
    minimal["sos"] = [[.5/32768, 1.5/32768, -1.5/32768, 1, -.25, .125]]
    s = load_spec(specfile(minimal))
    r = quantize_candidate(s, Descriptor((0,), (0,), 0))
    assert r.coefficients == ((0, 0, 2, -2, 8192, -4096),)
    assert all(v > 0 for v in r.jury_margins[0])


def test_representability_asymmetric_boundary(specfile, minimal):
    minimal["sos"] = [[1, 0, 0, 1, 0, 0]]
    s = load_spec(specfile(minimal))
    assert quantize_candidate(s, Descriptor((0,), (0,), 0)).reason == "coefficient_range"
    assert quantize_candidate(s, Descriptor((0,), (0,), 1)).coefficients[0][0] == 16384
    minimal["sos"][0][0] = -1
    s = load_spec(specfile(minimal))
    assert quantize_candidate(s, Descriptor((0,), (0,), 0)).coefficients[0][0] == -32768


def test_gain_candidates_preserve_float_transfer_and_count(specfile, minimal):
    minimal["sos"] = signal.butter(8, .05, output="sos").tolist()
    s = load_spec(specfile(minimal)); cs = generate_candidates(s)
    assert len(cs.descriptors) <= 5184
    assert len(cs.descriptors) == len(set(cs.descriptors))
    assert cs.descriptors[0].order == (0,1,2,3)
    assert cs.descriptors[0].exponents == (0,0,0,0)
    assert any(any(d.exponents) for d in cs.descriptors)
    w = np.linspace(0, np.pi, 333)
    base = signal.sosfreqz(s.normalized_sos, worN=w)[1]
    for d in cs.descriptors[::173]:
        assert sum(d.exponents) == 0
        a = np.array(s.normalized_sos)[list(d.order)]; a[:,:3] *= np.exp2(d.exponents)[:,None]
        assert np.max(np.abs(signal.sosfreqz(a, worN=w)[1] - base)) < 1e-12
    s2 = dataclasses.replace(s, search=dataclasses.replace(s.search,batch_size=1,candidate_budget=1))
    assert generate_candidates(s2).digest == cs.digest


def test_original_source_unstable(specfile, minimal):
    minimal["sos"] = [[.1,0,0,1,-1.01,0]]
    assert generate_candidates(load_spec(specfile(minimal))).source_reasons == ("source_linear_poles_not_strictly_stable",)


def test_transformation_overflow_rejected(specfile):
    s = load_spec(specfile())
    assert quantize_candidate(s,Descriptor((0,),(65,),0)).reason == "transformation_range"


def test_jury_strict_boundaries_and_random_roots():
    assert not jury(32768, 0, 0)[0]
    assert not jury(-32768, 0, 0)[0]
    assert not jury(0, -32768, 0)[0]
    rng = np.random.default_rng(80013)
    for _ in range(10000):
        p = int(rng.integers(0,4)); a,b = map(int,rng.integers(-32768,32768,2)); q=2**(15-p)
        actual, margins = jury(a,b,p)
        roots = np.roots([1,-a/q,-b/q])
        if min(abs(v) for v in margins) > 1:
            assert actual == (np.max(np.abs(roots)) < 1)
