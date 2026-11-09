import numpy as np
import pytest

pytest.importorskip("fastapi")


@pytest.fixture
def artifacts(tmp_path, monkeypatch, params):
    import json
    params.save(str(tmp_path / "params.npz"))
    (tmp_path / "results.json").write_text(json.dumps({"verdict": "X", "reliable": True, "roas": [], "fit": {}}))
    from mmm.config import CFG
    monkeypatch.setattr(CFG, "artifacts_dir", str(tmp_path))
    return tmp_path


def test_api_optimize_and_errors(artifacts):
    from fastapi.testclient import TestClient
    import api
    c = TestClient(api.app)
    assert c.get("/health").json()["artifacts"] is True
    ok = c.post("/optimize", json={})
    assert ok.status_code == 200 and len(ok.json()["allocation"]) == 3
    assert c.post("/optimize", json={"total_budget": 1e12}).status_code == 422
    assert c.get("/summary").json()["verdict"] == "X"


def test_api_without_artifacts(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from mmm.config import CFG
    import api
    monkeypatch.setattr(CFG, "artifacts_dir", str(tmp_path))
    assert TestClient(api.app).post("/optimize", json={}).status_code == 503
