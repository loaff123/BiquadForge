# Verification, source provenance and honest limits

The optional source fixture is from the official cmsisdsp Python distribution1.10.3:
https://pypi.org/project/cmsisdsp/1.10.3/

Archive SHA-256:307d32298faa7c58db912851027fce35f886fec7eb1a27c4b309ce1a97f47519.
This is not a C library release identifier. The kernel/init retain upstream V1.9.0
revision headers. `reference_hashes.json` covers the12 minimal source/header/license
files. They are unmodified, with full upstream Apache LICENSE and file notices.

The verifier compiles only snapshots of those checked bytes. It does not compile
arbitrary locally supplied variants or additional shadow headers. It regenerates
C from checked numeric manifest/vector content, byte-compares the pack's generated
C, rechecks the selected finite gates/output/state/tails, and compiles the regenerated
bytes in an isolated temporary directory. It reruns the complete declared bounded qualification, including candidate ranking and
audit counts, and compares canonical manifest, report and README claims. Verification
therefore adds approximately another build/search workload before compilation; no
latency guarantee is made.
Hashes record provenance/integrity, not authorship or cryptographic authentication.
Original custom input files are not reopened during verification; exact stored
Q15 inputs, including realized noise, are authoritative. Numeric-backend version
strings are retained as declared provenance, not independently authenticated
history. An explicitly chosen compiler is trusted software.

Exact flags: `-std=c11 -O2 -D__GNUC_PYTHON__ -DARM_MATH_AUTOVECTORIZE`.
The compiler receives controlled include paths and a sanitized environment, with
no shell. Kernel/init plus generated filter/replay are compiled. Compile/replay each
have a120-second subprocess timeout; timeout is a failed attempt, not qualification
failure or proof of impossibility.

Seven schedules start from zero state independently: chunks1,3,7,16,127,whole-vector,
and cyclic[1,7,4,31,64]. Every output and final-state entry is checked. The public
integration supports continuous state and independent caller-owned contexts.

Verification result distinctions:
- `not_requested` in the deterministic build manifest: no host-C verification yet
- `verified`: actual optional host scalar C checks passed
- `compiler_unavailable`: source/pack checked but selected executable unavailable
- `source_mismatch`: missing or changed pinned official source/header/license
- `invalid_pack`: incomplete, modified or inconsistent numeric/generated content
- `compile_failed`: attempted compiler failed/timed out
- `replay_failed`: compiled replay failed/timed out or comparison summary invalid

Actual attempt evidence is in separate `verification.json`: compiler/version, flags,
source hashes, source identity, validated-manifest digest, schedule and comparison
counts. An invalid recheck replaces previous success when the pack is writable;
if recording is impossible, the returned invalid result discloses that failure.
Arm hardware remains
unverified in every case. Build artifacts exclude times, paths and timings; for the
same specification/data/tool/dependency versions, output bytes are reproducible
across directories. Cross-version/platform floating-transcendental identity is not
promised; exact selected-metric revalidation can reject a pack under different
numeric versions. Use the recorded versions to reproduce it.

## Test coverage

The independent tests/oracle.py recurrence uses Python integer floor division and
has no production arithmetic imports. Optional ctypes testing compares128 seeded
random/extreme coefficient/input/initial-state cases (1–4 stages, p0–3),9488 distinct
samples, seven schedules, against the original official C, then production output
and final state. These are tests within scope, not exhaustive safety evidence.
PostShift14/15 are rejected: exploratory high-shift counterexamples make any broader
exactness claim inappropriate.

Further tests cover schema errors, ties, saturation vs legitimate boundary output,
Jury boundaries/random roots, noncontiguous/empty data, mutation isolation, strict
positive/rejection examples, hidden internal residuals, search accounting, deterministic
packs, no-clobber destinations, interruption markers, forged hashes/code, source
header changes, optional-compiler statuses and package contents.

The local validation described above is a historical 2026-10-08 checkpoint.
The [initial public source snapshot](https://github.com/loaff123/BiquadForge/commit/535273457b2b6a483e4145bade069e69b1c7cee2)
passed all 145 tests, package builds and standard scalar C replay on Python
3.10/3.12/3.13 in [its exact-commit CI run](https://github.com/loaff123/BiquadForge/actions/runs/37752250973).
Local fresh wheel/sdist installations are separate evidence. Hosted Linux CI does
not establish target-hardware, ABI-portability or real-time behavior.

## CI preparation

The workflow triggers on push and pull requests with read-only contents permission
and checkout credential persistence disabled. Official actions are fixed to
[checkout v4.2.2](https://github.com/actions/checkout/commit/11bd71901bbe5b1630ceea73d27597364c9af683)
and [setup-python v5.6.0](https://github.com/actions/setup-python/commit/a26af69be951a213d495a4c3e4e4022e16d87065).
These are deliberate immutable official release pins, not a claim to be the latest
action versions. Local tests check workflow guardrails. The initial public run
linked above verified the pinned workflow; later commits require their own
exact-commit result.
