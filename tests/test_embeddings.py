"""Test /v1/embeddings endpoint for vLLM bench compatibility."""
import pytest
from fastapi.testclient import TestClient
from xpyd_sim.server import ServerConfig, create_app


class TestEmbeddings:
    def test_string_input(self):
        """Single string input returns one embedding."""
        c = TestClient(create_app(ServerConfig(embedding_dim=128)))
        r = c.post("/v1/embeddings", json={"model": "dummy", "input": "hello world"})
        assert r.status_code == 200
        d = r.json()
        assert d["object"] == "list"
        assert len(d["data"]) == 1
        assert d["data"][0]["object"] == "embedding"
        assert d["data"][0]["index"] == 0
        assert len(d["data"][0]["embedding"]) == 128
        assert "usage" in d

    def test_list_input(self):
        """List of strings returns multiple embeddings."""
        c = TestClient(create_app(ServerConfig(embedding_dim=64)))
        r = c.post("/v1/embeddings", json={"model": "dummy", "input": ["hello", "world", "test"]})
        assert r.status_code == 200
        d = r.json()
        assert len(d["data"]) == 3
        for i, emb in enumerate(d["data"]):
            assert emb["index"] == i
            assert len(emb["embedding"]) == 64

    def test_default_dim(self):
        """Default embedding dimension is 1536."""
        c = TestClient(create_app(ServerConfig()))
        r = c.post("/v1/embeddings", json={"model": "dummy", "input": "test"})
        assert len(r.json()["data"][0]["embedding"]) == 1536

    def test_usage_tokens(self):
        """Usage should report prompt_tokens and total_tokens."""
        c = TestClient(create_app(ServerConfig()))
        r = c.post("/v1/embeddings", json={"model": "dummy", "input": "hello world"})
        u = r.json()["usage"]
        assert u["prompt_tokens"] > 0
        assert u["total_tokens"] == u["prompt_tokens"]

    def test_model_name_echoed(self):
        """Response model field matches config."""
        c = TestClient(create_app(ServerConfig(model_name="emb-model")))
        r = c.post("/v1/embeddings", json={"model": "emb-model", "input": "test"})
        assert r.json()["model"] == "emb-model"

    def test_extra_fields_accepted(self):
        """vLLM sends truncate_prompt_tokens, should not error."""
        c = TestClient(create_app(ServerConfig()))
        r = c.post("/v1/embeddings", json={
            "model": "dummy", "input": "test",
            "truncate_prompt_tokens": -1,
            "encoding_format": "float",
        })
        assert r.status_code == 200

    def test_empty_input(self):
        """Empty string input should still work."""
        c = TestClient(create_app(ServerConfig(embedding_dim=32)))
        r = c.post("/v1/embeddings", json={"model": "dummy", "input": ""})
        assert r.status_code == 200
        assert len(r.json()["data"]) == 1
