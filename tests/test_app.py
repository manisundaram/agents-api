from fastapi.testclient import TestClient

from app.main import create_app


app = create_app()
client = TestClient(app)


def test_ping_endpoint() -> None:
    response = client.get("/ping")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service_version"] == app.state.settings.app_version
    assert payload["service_name"] == app.state.settings.app_name


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == app.state.settings.app_version
    assert payload["configuration"]["provider"] == app.state.settings.provider_type
    assert payload["configuration"]["vectorstore"] == app.state.settings.vectorstore_backend
    assert payload["checks"][0]["name"] == "configuration"
    assert payload["providers"][0]["name"] == app.state.settings.provider_type


def test_diagnostics_endpoint() -> None:
    response = client.get("/diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["providers"][0]["provider"] == app.state.settings.provider_type
    assert payload["vectorstores"][0]["backend"] == app.state.settings.vectorstore_backend
    assert payload["summary"]["total_checks"] >= 3
    assert "bootstrap" in payload["performance_benchmarks"]


def test_metrics_endpoint() -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service_name"] == app.state.settings.app_name
    assert payload["service_version"] == app.state.settings.app_version
    assert payload["provider"] == app.state.settings.provider_type


def test_agent_tools_endpoint() -> None:
    response = client.get("/agent/tools")

    assert response.status_code == 200
    payload = response.json()
    assert "tools" in payload
    assert isinstance(payload["tools"], list)


def test_agent_query_endpoint_validation() -> None:
    # Test missing query
    response = client.post("/agent/query", json={})
    assert response.status_code == 422

    # Test valid request structure
    response = client.post("/agent/query", json={
        "query": "What is 2 + 2?",
        "tools": ["calculator"],
        "max_steps": 3
    })
    assert response.status_code == 200
    payload = response.json()
    assert "answer" in payload
    assert "trace" in payload


def test_agent_multi_endpoint_validation() -> None:
    # Test missing query
    response = client.post("/agent/multi", json={})
    assert response.status_code == 422

    # Test valid request structure
    response = client.post("/agent/multi", json={
        "query": "Analyze the weather and provide recommendations",
        "tools": ["search", "calculator"]
    })
    assert response.status_code == 200
    payload = response.json()
    assert "plan" in payload
    assert "final_answer" in payload
    assert "steps" in payload