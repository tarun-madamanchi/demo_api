def test_sample_api_success(client):
    payload = {
        "user_id": "123",
        "name": "John Doe",
        "description": "Test description",
    }

    response = client.post("/sample/sample-api", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert data["data"]["user_id"] == "123"
    assert "processed_by" in data["data"]
