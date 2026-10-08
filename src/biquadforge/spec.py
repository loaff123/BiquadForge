"""Strict data-only input loading and immutable, materialized specifications."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np

MAX_BYTES = 16 * 1024 * 1024
MAX_SAMPLES = 2_000_000
KEYWORDS = frozenset("auto break case char const continue default do double else enum extern float for goto if inline int long register restrict return short signed sizeof static struct switch typedef union unsigned void volatile while _Alignas _Alignof _Atomic _Bool _Complex _Generic _Imaginary _Noreturn _Static_assert _Thread_local".split())


class BiquadForgeError(ValueError):
    def __init__(self, message: str, field: str = "input", status: str = "error"):
        super().__init__(message)
        self.status, self.field = status, field

    def as_dict(self):
        return {"status": self.status, "field": self.field, "message": str(self)}


@dataclass(frozen=True)
class Suite:
    amplitude_q15: int = 24576
    excitation_samples: int = 1024
    settling_samples: int = 8192
    tail_samples: int = 1024
    seed: int = 20261008


@dataclass(frozen=True)
class Limits:
    max_grid_error: float
    max_abs_error: float
    max_rmse: float
    max_saturations: int = 0
    max_final_tail_q15: int = 1
    max_internal_tail_q15: int | None = None


@dataclass(frozen=True)
class Search:
    post_shifts: tuple[int, ...] = (0, 1, 2, 3)
    gain_radius: int = 1
    candidate_budget: int = 10000
    batch_size: int = 128


@dataclass(frozen=True)
class Vector:
    name: str
    samples: tuple[int, ...]
    encoding: str
    source_sha256: str
    clipped_input_count: int = 0


@dataclass(frozen=True)
class BuildSpec:
    name: str
    sample_rate_hz: float
    original_sos: tuple[tuple[float, ...], ...]
    normalized_sos: tuple[tuple[float, ...], ...]
    suite: Suite
    limits: Limits
    search: Search
    grid_points: int
    vectors: tuple[Vector, ...]
    source_sha256: str

    def as_dict(self):
        return asdict(self)


def _fail(message, field="input", status="error"):
    raise BiquadForgeError(message, field, status)


def _object(value, allowed, field):
    if not isinstance(value, dict):
        _fail("Expected an object", field)
    extra = value.keys() - set(allowed)
    if extra:
        _fail("Unknown keys: " + ", ".join(sorted(extra)), field)
    return value


def _integer(value, lo, hi, field, unsupported=False):
    if type(value) is not int:
        _fail("Expected an integer (not boolean or decimal)", field)
    if not lo <= value <= hi:
        _fail(f"Expected {lo}..{hi}", field, "unsupported" if unsupported else "error")
    return value


def _number(value, field, lo=None, hi=None):
    if type(value) not in (int, float):
        _fail("Expected a finite number", field)
    try:
        v = float(value)
    except (OverflowError, ValueError):
        _fail("Number cannot be represented in float64", field)
    if not math.isfinite(v) or (lo is not None and v < lo) or (hi is not None and v > hi):
        _fail("Number is nonfinite or outside the admitted range", field)
    return v


def validate_name(value, field="name"):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", value) or value in KEYWORDS:
        _fail("Expected a non-reserved ASCII identifier of 1..48 characters", field)
    return value


def _read_bytes(path):
    try:
        if path.stat().st_size > MAX_BYTES:
            _fail("Input file exceeds 16 MiB", str(path), "unsupported")
        with path.open("rb") as f:
            raw = f.read(MAX_BYTES + 1)
    except OSError as e:
        _fail(f"Cannot read input: {e}", str(path))
    if len(raw) > MAX_BYTES:
        _fail("Input file exceeds 16 MiB", str(path), "unsupported")
    return raw


def _parse_float_token(token):
    """Preserve the fact that a decimal token was nonzero before float64 conversion."""
    try:
        exact = Decimal(token)
        value = float(exact)
    except (InvalidOperation, OverflowError, ValueError) as e:
        _fail(f"Invalid float64 numeric token: {e}")
    if not exact.is_finite() or not math.isfinite(value):
        _fail("Numeric token is nonfinite or overflows float64")
    if exact != 0 and value == 0:
        _fail("Nonzero numeric token underflows float64")
    return value


def parse_json(raw: bytes):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                _fail(f"Duplicate JSON key: {k}")
            result[k] = v
        return result
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_float=_parse_float_token,
                          parse_constant=lambda x: _fail(f"Nonfinite JSON constant: {x}"))
    except (UnicodeError, ValueError, RecursionError) as e:
        if isinstance(e, BiquadForgeError):
            raise
        _fail(f"Invalid UTF-8 JSON: {e}")


def samples_digest(samples):
    return hashlib.sha256(np.asarray(samples, dtype="<i2").tobytes()).hexdigest()


def _read_vector(data, base, silence, names):
    _object(data, ("name", "path", "format", "encoding"), "suite.vectors")
    if set(data) != {"name", "path", "format", "encoding"}:
        _fail("Each vector requires name, path, format, encoding", "suite.vectors")
    name = validate_name(data["name"], "suite.vectors.name")
    if name in names:
        _fail("Vector names must be unique", "suite.vectors.name")
    names.add(name)
    if not isinstance(data["path"], str) or not data["path"] or "\0" in data["path"]:
        _fail("Expected a nonempty path", "suite.vectors.path")
    raw = _read_bytes(base / data["path"])
    fmt, enc = data["format"], data["encoding"]
    if fmt not in ("json", "csv") or enc not in ("q15", "normalized_float"):
        _fail("Admitted formats: json/csv; encodings: q15/normalized_float", "suite.vectors")
    if fmt == "json":
        values = parse_json(raw)
    else:
        values = []
        try:
            for row in csv.reader(io.StringIO(raw.decode("utf-8")), strict=True):
                if len(row) != 1 or not row[0].strip():
                    _fail("CSV requires exactly one number per row, no header", name)
                token = row[0].strip()
                if enc == "q15":
                    if not re.fullmatch(r"[+-]?\d+", token):
                        _fail("Q15 CSV requires integer tokens", name)
                    values.append(int(token))
                else:
                    values.append(_parse_float_token(token))
        except (UnicodeError, csv.Error, ValueError) as e:
            if isinstance(e, BiquadForgeError):
                raise
            _fail(f"Invalid numeric CSV: {e}", name)
    if not isinstance(values, list) or not 1 <= len(values) <= 65536:
        _fail("Vector must be a one-dimensional array of 1..65536 samples", name)
    clipped = 0
    if enc == "q15":
        converted = [_integer(v, -32768, 32767, name) for v in values]
    else:
        floats = [_number(v, name) for v in values]
        # Avoid overflow in x*32768 while preserving explicit rounded clipping counts.
        converted = []
        for v in floats:
            if v > 2.0:
                q = 32768
            elif v < -2.0:
                q = -32769
            else:
                q = int(round(v * 32768))
            clipped += not -32768 <= q <= 32767
            converted.append(min(32767, max(-32768, q)))
    return Vector(name, tuple(converted) + (0,) * silence, enc,
                  hashlib.sha256(raw).hexdigest(), clipped)


def load_spec(path: str | Path) -> BuildSpec:
    path = Path(path)
    raw = _read_bytes(path)
    data = _object(parse_json(raw), ("schema_version", "name", "sample_rate_hz", "sos", "suite", "grid", "limits", "search"), "input")
    for key in ("schema_version", "name", "sample_rate_hz", "sos", "limits"):
        if key not in data:
            _fail("Required field is missing", key)
    _integer(data["schema_version"], 1, 1, "schema_version", True)
    name = validate_name(data["name"])
    fs = _number(data["sample_rate_hz"], "sample_rate_hz", 0, 1e9)
    if fs == 0:
        _fail("Sample rate must be positive", "sample_rate_hz")
    rows = data["sos"]
    if not isinstance(rows, list) or not rows:
        _fail("Expected 1..4 SOS rows", "sos")
    if len(rows) > 4:
        _fail("At most four SOS rows supported", "sos", "unsupported")
    original, normalized = [], []
    for row in rows:
        if not isinstance(row, list) or len(row) != 6:
            _fail("Each SOS row needs six numbers", "sos")
        v = [_number(x, "sos") for x in row]
        if v[3] == 0:
            _fail("a0 must be nonzero", "sos")
        with np.errstate(all="ignore"):
            n = np.array(v, dtype=float) / v[3]
        if not np.all(np.isfinite(n)) or any(a != 0 and b == 0 for a, b in zip(v, n)):
            _fail("SOS normalization overflowed or underflowed", "sos")
        original.append(tuple(v)); normalized.append(tuple(map(float, n)))
    suite_d = _object(data.get("suite", {}), (*Suite.__dataclass_fields__, "vectors"), "suite")
    ranges = {"amplitude_q15": (1, 32767), "excitation_samples": (2, 16384), "settling_samples": (2, 65536), "tail_samples": (2, 4096), "seed": (0, 2**32-1)}
    suite = Suite(**{k: _integer(v, *ranges[k], f"suite.{k}") for k, v in suite_d.items() if k != "vectors"})
    lim = _object(data["limits"], Limits.__dataclass_fields__, "limits")
    for key in ("max_grid_error", "max_abs_error", "max_rmse"):
        if key not in lim:
            _fail("Required error threshold missing", f"limits.{key}")
    values = {k: _number(lim[k], f"limits.{k}", 0) for k in ("max_grid_error", "max_abs_error", "max_rmse")}
    for k in ("max_saturations", "max_final_tail_q15", "max_internal_tail_q15"):
        if k in lim:
            values[k] = None if k == "max_internal_tail_q15" and lim[k] is None else _integer(lim[k], 0, 2**63-1 if k == "max_saturations" else 32768, f"limits.{k}")
    limits = Limits(**values)
    sd = _object(data.get("search", {}), Search.__dataclass_fields__, "search")
    shifts = sd.get("post_shifts", [0, 1, 2, 3])
    if not isinstance(shifts, list) or not shifts:
        _fail("Expected a nonempty list", "search.post_shifts")
    shifts = [_integer(v, 0, 3, "search.post_shifts", True) for v in shifts]
    if len(shifts) != len(set(shifts)):
        _fail("Duplicate postShift", "search.post_shifts")
    search = Search(tuple(sorted(shifts)), _integer(sd.get("gain_radius", 1), 0, 1, "search.gain_radius", True), _integer(sd.get("candidate_budget", 10000), 1, 10000, "search.candidate_budget"), _integer(sd.get("batch_size", 128), 1, 256, "search.batch_size"))
    gd = _object(data.get("grid", {}), ("points",), "grid")
    points = _integer(gd.get("points", 4097), 17, 65537, "grid.points")
    custom = suite_d.get("vectors", [])
    if not isinstance(custom, list) or len(custom) > 16:
        _fail("At most 16 custom vectors", "suite.vectors", "unsupported")
    silence = suite.settling_samples + suite.tail_samples
    a, n = suite.amplitude_q15, suite.excitation_samples
    rng = np.random.Generator(np.random.PCG64(suite.seed))
    noises = tuple(map(int, rng.integers(-a, a + 1, size=n)))
    generated = [("positive_impulse", (a,) + (0,) * (n - 1)), ("negative_impulse", (-a,) + (0,) * (n - 1)), ("positive_step", (a,) * n), ("negative_step", (-a,) * n), ("noise", noises)]
    vectors = [Vector(k, v + (0,) * silence, "q15", samples_digest(v)) for k, v in generated]
    names = {v.name for v in vectors}
    total = sum(len(v.samples) for v in vectors)
    for d in custom:
        v = _read_vector(d, path.parent, silence, names)
        total += len(v.samples)
        if total > MAX_SAMPLES:
            _fail("Realized suite exceeds 2,000,000 samples", "suite", "unsupported")
        vectors.append(v)
    return BuildSpec(name, fs, tuple(original), tuple(normalized), suite, limits, search, points, tuple(vectors), hashlib.sha256(raw).hexdigest())
