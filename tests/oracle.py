"""Independent Python-integer recurrence: no production imports."""
def oracle(coefficients, shift, samples, state=None):
    stream = [int(x) for x in samples]
    all_states, counts, peaks, traces = [], [], [], []
    for stage, coefficients_row in enumerate(coefficients):
        b0, pad, b1, b2, a1, a2 = [int(v) for v in coefficients_row]
        x1, x2, y1, y2 = [0, 0, 0, 0] if state is None else [int(v) for v in state[stage]]
        result, count, peak = [], 0, 0
        for current in stream:
            total = b0*current + b1*x1 + b2*x2 + a1*y1 + a2*y2
            value = total // (2 ** (15-shift))
            count += int(value < -32768 or value > 32767)
            peak = max(peak, abs(value))
            clipped = max(-32768, min(32767, value))
            result.append(clipped)
            x2, x1 = x1, current
            y2, y1 = y1, clipped
        all_states.append([x1, x2, y1, y2]); counts.append(count); peaks.append(peak)
        traces.append(result)
        stream = result
    return stream, all_states, counts, peaks, traces
