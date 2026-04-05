"""Test API key authentication for vLLM bench / proxy compatibility."""
import pytest
from fastapi.testclient import TestClient
from xpyd_sim.server import ServerConfig, create_app


class TestApiKeyAuth:
    def test_no_auth_configured_allows_all(self):
        """Without require_api_key, all requests pass."""
        c = TestClient(create_app(ServerConfig()))
        assert c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1
        }).status_code == 200

    def test_missing_header_returns_401(self):
        """With auth configured, missing Authorization returns 401."""
        c = TestClient(create_app(ServerConfig(require_api_key="secret")))
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1
        })
        assert r.status_code == 401
        assert "error" in r.json()

    def test_wrong_key_returns_401(self):
        """Wrong API key returns 401."""
        c = TestClient(create_app(ServerConfig(require_api_key="secret")))
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1
        }, headers={"Authorization": "Bearer wrong-key"})
        assert r.status_code == 401

    def test_correct_key_returns_200(self):
        """Correct API key allows request through."""
        c = TestClient(create_app(ServerConfig(require_api_key="secret")))
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1
        }, headers={"Authorization": "Bearer secret"})
        assert r.status_code == 200

    def test_completions_endpoint_also_checked(self):
        """/v1/completions also requires auth."""
        c = TestClient(create_app(ServerConfig(require_api_key="key123")))
        assert c.post("/v1/completions", json={
            "prompt": "hi", "max_tokens": 1
        }).status_code == 401
        assert c.post("/v1/completions", json={
            "prompt": "hi", "max_tokens": 1
        }, headers={"Authorization": "Bearer key123"}).status_code == 200

    def test_models_endpoint_checked(self):
        """/v1/models also requires auth."""
        c = TestClient(create_app(ServerConfig(require_api_key="key123")))
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer key123"}).status_code == 200

    def test_non_v1_paths_no_auth(self):
        """Health, ping, metrics don't need auth."""
        c = TestClient(create_app(ServerConfig(require_api_key="secret")))
        assert c.get("/health").status_code == 200
        assert c.get("/ping").status_code == 200
        assert c.get("/metrics").status_code == 200

    def test_auth_error_format(self):
        """Error response matches OpenAI error format."""
        c = TestClient(create_app(ServerConfig(require_api_key="secret")))
        r = c.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1
        })
        d = r.json()
        assert d["error"]["type"] == "auth_error"
        assert "api key" in d["error"]["message"].lower() or "API key" in d["error"]["message"]
