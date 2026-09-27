"""API tests: the status endpoint reports requested and placed particle counts."""

import warnings

import pytest
from fastapi.testclient import TestClient

from api.main import app
from simulator.generator import PlacementSaturationWarning


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.parametrize("n_requested, expect_placed", [(5, 5), (100, 32)])
def test_status_reports_requested_and_placed(client, n_requested, expect_placed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PlacementSaturationWarning)
        r = client.post("/simulate", json={"n_particles": n_requested, "seed": 42})
    assert r.status_code in (200, 202)
    job_id = r.json()["job_id"]
    status = client.get(f"/status/{job_id}").json()
    assert status["status"] == "done"
    assert status["n_particles_requested"] == n_requested
    assert status["n_particles_placed"] == expect_placed
    # the existing field is still present
    assert status["sim_params"]["n_particles"] == n_requested
