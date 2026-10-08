"""Finite deterministic section-order/power-of-two candidate generation."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import numpy as np
from scipy import signal
from .spec import BuildSpec, BiquadForgeError


@dataclass(frozen=True)
class Descriptor:
    order: tuple[int, ...]
    exponents: tuple[int, ...]
    post_shift: int


@dataclass(frozen=True)
class CandidateSet:
    descriptors: tuple[Descriptor, ...]
    digest: str
    center_observations: tuple[str, ...]
    source_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CandidateReject:
    descriptor: Descriptor
    reason: str


@dataclass(frozen=True)
class PreparedCandidate:
    descriptor: Descriptor
    coefficients: tuple[tuple[int, ...], ...]
    quantized_sos: tuple[tuple[float, ...], ...]
    jury_margins: tuple[tuple[int, ...], ...]
    grid_error: float


def frequencies(spec):
    # Work in radians; sample rate labels the mathematically same grid.
    return np.linspace(0.0, np.pi, spec.grid_points)


def reference_response(spec):
    with np.errstate(all="ignore"):
        h = signal.sosfreqz(spec.normalized_sos, worN=frequencies(spec))[1]
    if not np.all(np.isfinite(h)):
        raise BiquadForgeError("Original response is nonfinite on the qualification grid", "sos")
    return h


def jury(a1, a2, post_shift):
    q = 2 ** (15-post_shift)
    margins = (q-int(a1)-int(a2), q+int(a1)-int(a2), q+int(a2))
    return all(v > 0 for v in margins), margins


def generate_candidates(spec: BuildSpec) -> CandidateSet:
    sos = np.array(spec.normalized_sos)
    n = len(sos)
    if any(np.max(np.abs(np.roots([1, row[4], row[5]]))) >= 1 for row in sos):
        return CandidateSet((), hashlib.sha256(b"[]").hexdigest(), (), ("source_linear_poles_not_strictly_stable",))
    reference_response(spec)
    w = frequencies(spec)
    centers, observations = [], []
    for order in itertools.permutations(range(n)):
        centers.append((order, (0,)*n))
        if n == 1:
            continue
        prefix = np.ones(len(w), dtype=np.complex128)
        cumulative = [0]
        for index in order[:-1]:
            with np.errstate(all="ignore"):
                prefix *= signal.freqz(sos[index,:3], sos[index,3:], worN=w)[1]
            peak = float(np.max(np.abs(prefix)))
            if not math.isfinite(peak) or peak <= 0:
                observations.append(f"prefix_center_unavailable_for_order_{','.join(map(str,order))}")
                break
            cumulative.append(-math.ceil(math.log2(peak)))
        else:
            cumulative.append(0)
            centers.append((order, tuple(b-a for a,b in zip(cumulative,cumulative[1:]))))
    descriptors, seen = [], set()
    def add(order, exponents):
        for p in spec.search.post_shifts:
            d = Descriptor(tuple(order), tuple(exponents), p)
            if d not in seen:
                seen.add(d); descriptors.append(d)
    for order, center in centers:
        add(order, center)
    r = spec.search.gain_radius
    for order, center in centers:
        for prefix_delta in itertools.product(range(-r,r+1), repeat=n-1):
            delta = (*prefix_delta, -sum(prefix_delta))
            add(order, tuple(a+b for a,b in zip(center,delta)))
    payload = json.dumps([asdict(d) for d in descriptors], sort_keys=True, separators=(",",":"))
    return CandidateSet(tuple(descriptors), hashlib.sha256(payload.encode()).hexdigest(), tuple(observations), ())


def quantize_candidate(spec: BuildSpec, descriptor: Descriptor, reference=None):
    d = descriptor
    if any(abs(v) > 64 for v in d.exponents) or sum(d.exponents) != 0:
        return CandidateReject(d, "transformation_range")
    sos = np.array(spec.normalized_sos)[list(d.order)].copy()
    before = sos[:,:3].copy()
    with np.errstate(all="ignore"):
        sos[:,:3] = np.ldexp(before, np.array(d.exponents)[:,None])
    if not np.all(np.isfinite(sos)) or np.any((before != 0) & (sos[:,:3] == 0)):
        return CandidateReject(d, "transformation_range")
    q = 2**(15-d.post_shift)
    row = np.column_stack((sos[:,0], np.zeros(len(sos)), sos[:,1:3], -sos[:,4:]))
    with np.errstate(all="ignore"):
        rounded = np.rint(row*q)
    if not np.all(np.isfinite(rounded)) or np.any(rounded < -32768) or np.any(rounded > 32767):
        return CandidateReject(d, "coefficient_range")
    co = rounded.astype(np.int64)
    checks = [jury(c[4],c[5],d.post_shift) for c in co]
    if not all(ok for ok,_ in checks):
        return CandidateReject(d, "quantized_linear_poles_not_strictly_stable")
    effective = np.column_stack((co[:,[0,2,3]]/q, np.ones(len(co)), -co[:,4:]/q))
    with np.errstate(all="ignore"):
        h = signal.sosfreqz(effective,worN=frequencies(spec))[1]
    if not np.all(np.isfinite(h)):
        return CandidateReject(d, "nonfinite_quantized_response")
    if reference is None:
        reference = reference_response(spec)
    err = float(np.max(np.abs(h-reference)))
    return PreparedCandidate(d, tuple(tuple(map(int,c)) for c in co), tuple(tuple(map(float,c)) for c in effective), tuple(m for _,m in checks), err)
