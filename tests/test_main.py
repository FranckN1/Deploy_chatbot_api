import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_get_homepage():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

def test_chatbot_response():
    response = client.post("/chatbot/", json={"message": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert data["response"] is not None
