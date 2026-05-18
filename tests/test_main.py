from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_predict():
    payload = {"complaint": "Street light not working for 3 days"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "department" in data
    assert "priority_label" in data
