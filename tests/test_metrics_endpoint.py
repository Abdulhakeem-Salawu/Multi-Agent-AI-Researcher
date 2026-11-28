from fastapi.testclient import TestClient

from research_agent.api import create_app
from research_observability.metrics import PROMETHEUS_AVAILABLE


def test_metrics_endpoint_is_available_when_prometheus_extra_is_installed():
    response = TestClient(create_app()).get("/metrics")
    if PROMETHEUS_AVAILABLE:
        assert response.status_code == 200
        assert "research_runs" in response.text
    else:
        assert response.status_code == 503
