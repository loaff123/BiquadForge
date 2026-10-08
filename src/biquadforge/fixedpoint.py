"""Exact standard scalar recurrence for the deliberately bounded postShift 0..3."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .spec import BiquadForgeError


def _integers(values, ndim, field, empty=False):
    def reject_boolean(value, depth=0):
        if depth > 8:
            raise BiquadForgeError("Array nesting exceeds the admitted dimensions", field)
        if isinstance(value, (bool, np.bool_)) or (isinstance(value, np.ndarray) and value.dtype.kind == "b"):
            raise BiquadForgeError("Boolean values are not Q15 integers", field)
        if isinstance(value, (list, tuple)):
            for item in value:
                reject_boolean(item, depth+1)
    reject_boolean(values)
    bare_empty = empty and isinstance(values, (list, tuple)) and len(values) == 0
    try:
        a = np.asarray(values)
    except (ValueError, TypeError, OverflowError) as e:
        raise BiquadForgeError(f"Invalid integer array: {e}", field) from e
    if a.ndim != ndim or (a.dtype.kind not in "iu" and not bare_empty):
        raise BiquadForgeError("Expected an integer array of the documented dimension", field)
    if a.size and (np.any(a < -32768) or np.any(a > 32767)):
        raise BiquadForgeError("Integer is outside Q15 range", field)
    return a.astype(np.int64, copy=True)


def _shift(p):
    if type(p) is not int or not 0 <= p <= 3:
        raise BiquadForgeError("Only integer postShift 0..3 supported", "post_shift", "unsupported")
    return p


def _coefficients(c, batched=False):
    co = _integers(c, 3 if batched else 2, "coefficients")
    if co.shape[-1] != 6 or not 1 <= co.shape[-2] <= 4 or np.any(co[..., 1] != 0) or (batched and co.shape[0] < 1):
        raise BiquadForgeError("Expected 1..4 rows [b0,0,b1,b2,A1,A2]", "coefficients")
    return co


def _run_scalar(coefficients, shift, samples, state):
    """Internal state-injectable kernel for tests; public state always begins zero."""
    data = [int(v) for v in samples]
    new_state = np.array(state, dtype=np.int16, copy=True)
    for k, row in enumerate(coefficients):
        b0, _, b1, b2, a1, a2 = map(int, row)
        x1, x2, y1, y2 = map(int, new_state[k])
        output = []
        for x0 in data:
            acc = b0*x0 + b1*x1 + b2*x2 + a1*y1 + a2*y2
            raw = acc >> (15-shift)
            y0 = min(32767, max(-32768, raw))
            output.append(y0)
            x2, x1, y2, y1 = x1, x0, y1, y0
        new_state[k] = [x1, x2, y1, y2]
        data = output
    return np.asarray(data, dtype=np.int16), new_state


class Q15Cascade:
    """Owns zero-initialized, continuous state. process() never implicitly resets."""
    def __init__(self, coefficients, post_shift):
        self._coefficients = _coefficients(coefficients)
        self._post_shift = _shift(post_shift)
        self._state = np.zeros((len(self._coefficients), 4), dtype=np.int16)

    @property
    def state(self):
        return self._state.copy()

    def reset(self):
        self._state.fill(0)

    def process(self, inputs):
        x = _integers(inputs, 1, "inputs", empty=True)
        result, self._state = _run_scalar(self._coefficients, self._post_shift, x, self._state)
        return result


@dataclass
class BatchSimulation:
    outputs: np.ndarray
    states: np.ndarray
    saturations: np.ndarray
    peaks: np.ndarray
    tail_peaks: np.ndarray
    stage_outputs: np.ndarray | None


def simulate_batch(coefficients, shifts, inputs, tail_samples=0, capture_stages=False):
    """Zero-state candidates, one common input vector; no caller arrays are mutated."""
    co = _coefficients(coefficients, batched=True)
    if len(shifts) != len(co):
        raise BiquadForgeError("One postShift required per candidate", "post_shifts")
    sh = np.array([15-_shift(int(p) if isinstance(p, np.integer) else p) for p in shifts])
    x = _integers(inputs, 1, "inputs", empty=True)
    count, stages, _ = co.shape
    length = len(x)
    if type(tail_samples) is not int or not 0 <= tail_samples <= length:
        raise BiquadForgeError("Invalid tail length", "tail_samples")
    data = np.broadcast_to(x, (count, length))
    states = np.zeros((count, stages, 4), dtype=np.int16)
    sats = np.zeros((count, stages), dtype=np.int64)
    peaks = np.zeros_like(sats); tails = np.zeros_like(sats)
    traces = np.zeros((count, stages, length), dtype=np.int16) if capture_stages else None
    for k in range(stages):
        c = co[:, k]
        # Feed-forward terms are vectorized; feedback remains a time recurrence.
        ff = data * c[:, 0, None]
        if length > 1:
            ff[:, 1:] += data[:, :-1] * c[:, 2, None]
        if length > 2:
            ff[:, 2:] += data[:, :-2] * c[:, 3, None]
        out = np.zeros((count, length), dtype=np.int64)
        y1 = np.zeros(count, dtype=np.int64); y2 = y1.copy()
        for t in range(length):
            raw = (ff[:, t] + c[:, 4]*y1 + c[:, 5]*y2) >> sh
            sats[:, k] += (raw < -32768) | (raw > 32767)
            peaks[:, k] = np.maximum(peaks[:, k], np.abs(raw))
            y0 = np.clip(raw, -32768, 32767)
            out[:, t] = y0
            y2, y1 = y1, y0
        if length:
            states[:, k, 0] = data[:, -1]
            states[:, k, 1] = data[:, -2] if length > 1 else 0
            states[:, k, 2] = out[:, -1]
            states[:, k, 3] = out[:, -2] if length > 1 else 0
        if tail_samples:
            tails[:, k] = np.max(np.abs(out[:, -tail_samples:]), axis=1)
        if capture_stages:
            traces[:, k] = out
        data = out
    return BatchSimulation(data.astype(np.int16), states, sats, peaks, tails, traces)
