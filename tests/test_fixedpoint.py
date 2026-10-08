import numpy as np
import pytest
from biquadforge.fixedpoint import Q15Cascade, simulate_batch, _run_scalar
from biquadforge.spec import BiquadForgeError
from oracle import oracle


def test_runtime_floor_not_ties_even():
    q = Q15Cascade([[16384, 0, 0, 0, 0, 0]], 0)
    assert q.process([-1, 1]).tolist() == [-1, 0]
    q.reset()
    assert q.state.tolist() == [[0, 0, 0, 0]]


def test_boundary_is_not_saturation():
    x = np.array([32767, -32768], dtype=np.int16)
    r = simulate_batch(np.array([[[16384, 0, 0, 0, 0, 0]]]), [1], x, 2)
    assert r.outputs[0].tolist() == x.tolist()
    assert r.saturations[0, 0] == 0
    r = simulate_batch(np.array([[[32767, 0, 0, 0, 0, 0]]]), [1], x, 2)
    assert r.saturations[0, 0] == 2
    assert r.peaks[0, 0] > 32768

@pytest.mark.parametrize("stages", [1, 2, 4])
@pytest.mark.parametrize("shift", [0, 1, 2, 3])
def test_independent_oracle_chunk_state_batch(stages, shift):
    rng = np.random.default_rng(stages*13 + shift)
    coeff = rng.integers(-32768, 32768, (stages, 6), dtype=np.int16); coeff[:, 1] = 0
    for length in [0, 1, 2, 3, 17, 193]:
        x = rng.integers(-32768, 32768, length*2, dtype=np.int16)[::2]
        expected, state, counts, peaks, traces = oracle(coeff, shift, x)
        q = Q15Cascade(coeff, shift)
        pieces = [q.process(x[i:i+7]) for i in range(0, length, 7)]
        actual = np.concatenate(pieces).tolist() if pieces else q.process([]).tolist()
        assert actual == expected
        assert q.state.tolist() == state
        r = simulate_batch(coeff[None], [shift], x, min(8, length), True)
        assert r.outputs[0].tolist() == expected
        assert r.states[0].tolist() == state
        assert r.saturations[0].tolist() == counts
        assert r.peaks[0].tolist() == peaks
        assert r.stage_outputs[0].tolist() == traces
        assert r.tail_peaks[0].tolist() == [max(map(abs, y[-min(8, length):]), default=0) for y in traces]


def test_internal_extreme_states_and_snapshot():
    c = np.array([[32767, 0, -32768, 1, 32767, -32768]], dtype=np.int16)
    initial = np.array([[32767, -32768, 32767, -32768]], dtype=np.int16)
    expected = oracle(c, 3, [32767, -32768], initial)
    got = _run_scalar(c, 3, np.array([32767, -32768], dtype=np.int16), initial)
    assert got[0].tolist() == expected[0]
    assert got[1].tolist() == expected[1]
    assert initial.tolist() == [[32767, -32768, 32767, -32768]]
    q = Q15Cascade(c, 3)
    snapshot = q.state; snapshot[:] = 123
    assert not np.any(q.state)
    q.process([1]); before = q.state
    assert q.process([]).size == 0
    assert np.array_equal(before, q.state)

@pytest.mark.parametrize("co,shift", [([[1,1,0,0,0,0]],0), ([[1.,0,0,0,0,0]],0), ([[32768,0,0,0,0,0]],0), ([[1,0,0,0,0,0]],14), ([[1,0,0,0,0,0]],15), ([[1,0,0,0,0,0]],True)])
def test_invalid_coefficient_or_shift(co, shift):
    with pytest.raises(BiquadForgeError):
        Q15Cascade(co, shift)

@pytest.mark.parametrize("x", [[1.0], [True], [32768], [[1]], [2**100]])
def test_invalid_input(x):
    with pytest.raises(BiquadForgeError):
        Q15Cascade([[1,0,0,0,0,0]],0).process(x)


def test_batch_noncontiguous_no_mutation():
    rng = np.random.default_rng(31)
    co = rng.integers(-32768,32768,(7,2,6),dtype=np.int16); co[:,:,1]=0
    x = rng.integers(-32768,32768,113,dtype=np.int16)
    oldco, oldx = co.copy(), x.copy()
    a = simulate_batch(co, [0,1,2,3,0,1,2], x, 5)
    for i in range(7):
        b = simulate_batch(co[i:i+1], [[0,1,2,3,0,1,2][i]], x, 5)
        assert np.array_equal(a.outputs[i], b.outputs[0])
    assert np.array_equal(co,oldco) and np.array_equal(x,oldx)
