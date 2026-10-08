# Input and arithmetic contracts

All JSON is data only, UTF-8, with duplicate/unknown keys and nonfinite numbers
rejected. Nonzero decimal tokens that would underflow to zero in float64 are rejected
before source normalization or vector conversion. Booleans do not count as numbers. Each input file is at most 16 MiB.

## Required root fields

- `schema_version`: integer 1
- `name`: non-reserved ASCII C identifier beginning with a letter, 1–48 characters
- `sample_rate_hz`: finite positive number at most 1e9; never implicit resampling
- `sos`: 1–4 finite real rows `[b0,b1,b2,a0,a1,a2]`, nonzero a0. Normalize each
  complete row by a0; reject overflow and nonzero-to-zero underflow. Original
  linear poles must be strictly inside the unit circle.
- `limits`: required `max_grid_error`, `max_abs_error`, `max_rmse`, each finite >=0.
  Error and RMSE use actual quantized input/32768 and original float64 SciPy SOS.
  RMSE is gated separately on every vector, not averaged across vectors.

## Optional fields

`limits` also allows:
- `max_saturations`: integer 0..2^63-1, default0; sum of all stage/vector events
- `max_final_tail_q15`: integer0..32768, default1; every vector separately
- `max_internal_tail_q15`: null (default) or integer0..32768; every stage/vector

A requested final-tail limit above1 is an explicit prominent relaxation. No internal
limit means observations still appear; it never means internal stages are ignored.

`suite` defaults and bounds:
- `amplitude_q15`: 24576, integer1..32767
- `excitation_samples`: 1024, integer2..16384
- `settling_samples`: 8192, integer2..65536 zero samples
- `tail_samples`: 1024, integer2..4096 further zeros, exactly the measured interval
- `seed`: 20261008, integer0..2^32-1; NumPy PCG64 integer noise, inclusive ±amplitude
- `vectors`: up to16 `{name,path,format,encoding}` objects. Names unique across the
  entire suite. `json` is a 1-D numeric array; `csv` exactly one column, no header,
  comments, empty rows. `q15` encoding requires integer tokens -32768..32767;
  `normalized_float` performs nearest-even(x*32768), then saturation. Rounded
  values requiring clipping are counted. Relative paths resolve beside the spec.
  Each custom excitation contains1..65536 samples; configured silence is appended.

Mandatory vectors are +impulse, -impulse, +step, -step, noise. Impulses occupy sample0
of the excitation interval. Each complete vector starts from zero state. Realized
input samples, source-byte hashes, little-endian-int16 vector hashes and backend
versions are stored. Verification replays exact stored noise, not a future PRNG.
The total admitted realized suite is at most2,000,000 samples. This is not a total
process-memory or time guarantee.

`grid.points`: 4097 by default, integer17..65537. Uniform float64 grid including DC
and Nyquist. The metric is max |H_quantized-H_original| on this finite grid.

`search`:
- `post_shifts`: unique list from0,1,2,3; defaultall, canonically sorted
- `gain_radius`: integer0 or1, default1
- `candidate_budget`: integer1..10000, default10000
- `batch_size`: integer1..256, default128; results independent of batch size

## Search

All section permutations are lexicographic, original first. Each has a zero-gain
center and conventional cumulative-prefix-peak center. For each proper prefix j,
`k_j=-ceil(log2(max_grid|H_prefix|))`; prefix exponent differences define stage
exponents, ending with the negative sum so overall gain is preserved before
quantization. Zero/nonfinite prefixes have no heuristic center. Center descriptors
are considered first; then first N-1 local exponent offsets in[-r,r], with last
offset cancelling their sum. Descriptors are deduplicated, preserving first order.
At most5184 exist before deduplication. Exponent magnitudes above64, overflow and
underflow are explicitly rejected. Coefficient rejects still consume budget.

Exact descriptor list/digest, planned/evaluated/remaining counts, limits and failure
counts are retained. A passing truncated search can emit a pack with an observation;
a nonpassing truncated search is budget exhaustion. A complete nonpassing search
is rejection of that declared set only. Selection minimizes maximum per-vector RMSE,
then max absolute error, grid error, total saturations, then candidate ordinal.
The best three fully simulated rejected candidates retain diagnostic tails. Early
coefficient/pole/grid rejects retain counts and example descriptors.

## Arithmetic

For common p, Q=2^(15-p). Coefficients are nearest-even(float64
`[b0,0,b1,b2,-a1,-a2]*Q`); rounded values outside int16 reject the candidate.
Runtime forms five signed16 products in int64, shifts arithmetically by15-p
(floor for negative values), then saturates to int16. Exactly out-of-range preclamp
values are saturation events; legitimate boundary outputs are not.

State per stage: `[x[n-1],x[n-2],y[n-1],y[n-2]]`. Coefficients have6N entries, state4N.
Exact Jury inequalities: Q-A1-A2>0, Q+A1-A2>0, Q+A2>0. Their conclusion is linear
rational denominator stability, not nonlinear limit-cycle freedom.

Full states are compared at every measured zero-tail sample. A nonzero repeat
proves periodic continuation of this deterministic integer recurrence. No repeat
is `inconclusive`; a zero fixed point is ordinary finite-suite quiescence. Nonzero
stage tails are disclosed even if decaying or masked by zero final output.
