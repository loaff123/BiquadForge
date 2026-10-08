import json
import pytest

@pytest.fixture
def minimal():
    return {"schema_version": 1, "name": "demo", "sample_rate_hz": 48000,
            "sos": [[0.5, 0, 0, 1, 0, 0]],
            "limits": {"max_grid_error": 0.001, "max_abs_error": 0.001, "max_rmse": 0.001}}

@pytest.fixture
def specfile(tmp_path, minimal):
    def write(value=None):
        p = tmp_path / "filter.json"
        p.write_text(json.dumps(minimal if value is None else value))
        return p
    return write
