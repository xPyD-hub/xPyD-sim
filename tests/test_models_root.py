"""Test /v1/models root field for vLLM bench compatibility."""
from fastapi.testclient import TestClient
from xpyd_sim.server import ServerConfig, create_app


class TestModelsRoot:
    def test_root_field_present(self):
        """vLLM bench reads data[0]['root'], must be present."""
        c = TestClient(create_app(ServerConfig(model_name="my-model")))
        d = c.get("/v1/models").json()
        model = d["data"][0]
        assert "root" in model, "root field missing from /v1/models response"
        assert model["root"] == "my-model"

    def test_root_equals_model_id(self):
        """root should match the model id."""
        c = TestClient(create_app(ServerConfig(model_name="test-7b")))
        d = c.get("/v1/models").json()
        assert d["data"][0]["root"] == d["data"][0]["id"]

    def test_root_default_model(self):
        """Default model name should also have root."""
        c = TestClient(create_app(ServerConfig()))
        d = c.get("/v1/models").json()
        assert d["data"][0]["root"] is not None
