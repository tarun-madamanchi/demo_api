def test_health_endpoint(client):
    """
    Test the /health endpoint using pytest fixtures.
    """
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
