# BiquadForge

A small, headless SciPy-SOS → CMSIS standard scalar Q15 qualification workflow.
It emits a reproducible C integration/replay pack **or a precise rejection**.
Normal use needs Python, NumPy and SciPy. A compiler and CMSIS sources are optional.

This is a pre-release source checkpoint; it has not been published to PyPI.

## Quick start

From this source directory, with Python 3.10 or newer:

```sh
python -m pip install .
biquadforge build examples/quiet_fir.json --out quiet-pack
biquadforge build examples/reject_issue127.json --out rejected-pack
```

The second command deliberately exits 2. A rejection is a useful result: no
candidate in the declared bounded search passed the strict finite-suite rules.
It does not prove that no fixed-point realization exists.

A minimal specification:

```json
{
  "schema_version": 1,
  "name": "half_gain",
  "sample_rate_hz": 48000,
  "sos": [[0.5, 0, 0, 1, 0, 0]],
  "limits": {
    "max_grid_error": 0.001,
    "max_abs_error": 0.001,
    "max_rmse": 0.001
  }
}
```

The three error limits are required caller choices. Defaults materialized in the
manifest include five signed/noise excitations, amplitude 24576 Q15, 1024 excitation
samples, 8192 settling zeros, 1024 measured tail zeros, a 4097-point response grid,
zero permitted saturation events, and at most **1 Q15 unit final-output tail**.
See [the full contract](docs/contracts.md) for every field, bound and convention.

## What it checks

- SciPy normalization, CMSIS feedback sign, padded six-value coefficient layout
- Explicit ties-even coefficient rounding, representability without clipping
- Exact integer Jury conditions for the quantized **linear denominator**
- Complex-response error on a finite grid, including DC and Nyquist
- Actual integer-vector error, exact per-stage preclamp saturation counts
- Positive/negative impulse, positive/negative step, seeded noise to silence
- Every stage's tail, exact repeated full states, and final-state continuity

Candidate preparation uses original-order conversion, conventional cumulative-
prefix gain scaling, section permutations, and bounded powers-of-two numerator
redistribution whose exponents sum to zero. It never changes input gain, redesigns
the filter, loosens thresholds, or silently switches arithmetic. Search is local
and finite; no optimizer superiority or global optimality is claimed.

## Internal residuals matter

`examples/bandpass_observations.json` is a four-section Butterworth bandpass. In the
recorded local run, all 5184 descriptors were evaluated and 735 passed. The selected
final-output tails were zero, while internal stages retained up to **2 Q15 units**
and exact repeated nonzero state was observed. Its status is therefore
`accepted_for_suite_with_observations`. These figures describe that exact suite
and dependency versions, not a general filter-quality benchmark.

`examples/internal_residual.json` is a smaller regression fixture with quiet final
output and an upstream -1 fixed point. `examples/reject_issue127.json` demonstrates
strict rejection of real public support-thread coefficients. No rejection example
is silently rewritten into a passing design.

## Outputs and statuses

An accepted pack includes `filter.h`, `filter.c`, `manifest.json`, `report.md`,
`vectors.json`, `vectors.h`, `replay.c` and `README.md`. Coefficients are const;
caller-owned contexts have independent state. Init/reset is explicit and blocks
carry state. Every stored vector has exact output and final-state expectations.
A rejected/exhausted pack contains only diagnostic manifest, report and README.

| Build status | Exit | Meaning |
|---|---:|---|
| `accepted_for_suite` | 0 | Finite gates pass; no nonzero tail observed in this suite |
| `accepted_for_suite_with_observations` | 0 | Passes with prominent residual/clipping/relaxed-limit/truncated-search observations |
| `rejected` | 2 | Declared set fully searched with no passing candidate, or unstable original source poles |
| `search_budget_exhausted` | 3 | No pass yet; declared descriptors remain unevaluated |
| `unsupported` | 4 | Understood input outside admitted v1 domain/workload |
| `error` | 1 | Invalid input, I/O or analysis failure |

Output directories must be new. Existing files, directories and symlinks are
preserved. An exclusive reservation and `INCOMPLETE` marker make handled partial
writes detectable. Never use a marked directory; the verifier refuses it. This is
best-effort interruption handling, not universal atomicity or crash durability.

## Optional official-C replay

The source distribution includes an optional minimal, unmodified official fixture:

```sh
biquadforge verify quiet-pack --cmsis-root tests/reference/cmsisdsp-1.10.3
```

The fixture is from **cmsisdsp Python distribution 1.10.3**, not an inferred CMSIS
C release number. Apache-2.0 LICENSE and upstream notices are preserved. The wheel
contains only the provenance/hash allowlist, not the optional C fixture.

Verification never downloads or installs software. It repeats the declared bounded
qualification (including ranking/audit facts), regenerates and compares manifest,
report, README and C artifacts, checks all pinned transitive local
source/header bytes, copies only those checked bytes to an isolated directory, then
invokes the explicitly selected compiler. This adds another search workload before
compilation. It checks outputs and final states for
chunks 1, 3, 7, 16, 127, whole-vector, and cyclic [1, 7, 4, 31, 64].

`verification.json` separates `verified`, `compiler_unavailable`, `source_mismatch`,
`invalid_pack`, `compile_failed`, and `replay_failed`. Unavailable compiler exits 5;
other verification failure exits 6. No attempted or failed compile becomes a pass.
See [verification and provenance](docs/verification.md).

## Python API

```python
from biquadforge import load_spec, qualify, write_pack, verify_pack, Q15Cascade
result = qualify(load_spec("filter.json"))
write_pack(result, "new-pack")
# Explicit optional host-C verification:
# evidence = verify_pack("new-pack", "tests/reference/cmsisdsp-1.10.3")

cascade = Q15Cascade([[16384, 0, 0, 0, 0, 0]], 0)
y = cascade.process([-1, 1])  # [-1, 0]: runtime negative shift floors
snapshot = cascade.state   # copy; changing it does not inject state
cascade.reset()
```

## Scope and development

Supported: 1–4 SOS rows, signed Q15, zero initial state, standard scalar CMSIS DF1,
one common postShift in 0–3. No fast, DSP-intrinsic, Neon, Helium/MVE, Q31, arbitrary
external state, hardware/firmware, real-time, ABI-portability, universal stability,
limit-cycle-freedom, whole-band, arbitrary-input, or safety-critical claims.

```sh
python -m pip install . pytest build "setuptools>=77" wheel
python -m pytest -q
python -m build
```

Optional C tests skip explicitly when the compiler is unavailable; ordinary unit
and build use never requires it. The local qualification figures above are
historical evidence from 2026-10-08. The [initial public source snapshot](https://github.com/loaff123/BiquadForge/commit/535273457b2b6a483e4145bade069e69b1c7cee2)
passed all 145 tests, package builds and standard scalar C replay on Python
3.10/3.12/3.13 in [its exact-commit CI run](https://github.com/loaff123/BiquadForge/actions/runs/37752250973).
See [Actions](https://github.com/loaff123/BiquadForge/actions) for later commits;
these host checks do not establish target-hardware suitability.

Original code: MIT. Optional upstream fixture: Apache-2.0 with notices retained.
See [third-party notices](THIRD_PARTY_NOTICES.md).
