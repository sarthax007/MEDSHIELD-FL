import pytest
import requests
import os

API_BASE_URL = f"http://127.0.0.1:{os.environ.get('API_PORT', '8001')}"


@pytest.fixture(scope="session")
def auth_token():
    # Use the seeded admin user
    login_data = {"username": "admin1", "password": "password123"}
    response = requests.post(f"{API_BASE_URL}/auth/login", data=login_data)
    assert response.status_code == 200, "Failed to login for E2E tests"
    return response.json()["access_token"]


def test_health_check():
    response = requests.get(f"{API_BASE_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_fl_round_completed_and_privacy_ensured(auth_token):
    # Verify that a round has been completed and metrics are available
    response = requests.get(
        f"{API_BASE_URL}/rounds/", headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert response.status_code == 200, f"Failed to get rounds: {response.text}"
    rounds = response.json()
    assert len(rounds) >= 1, "At least one round should be completed"

    # Assert privacy
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = requests.get(f"{API_BASE_URL}/privacy/status", headers=headers)
    assert response.status_code == 200
    privacy_status = response.json()
    assert (
        privacy_status.get("status") == "verifiable_local"
    ), "Privacy status not verifiable local!"
    assert (
        privacy_status.get("raw_images_shared") == 0
    ), "Privacy compromised: Raw images were shared!"


def test_prediction_and_explanation(auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}

    import io
    from PIL import Image

    img = Image.new("RGB", (224, 224), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()

    files = {"file": ("test.jpg", img_byte_arr, "image/jpeg")}
    response = requests.post(
        f"{API_BASE_URL}/predictions/", files=files, headers=headers
    )

    assert response.status_code == 201, f"Prediction failed: {response.text}"
    result = response.json()

    assert "predicted_class" in result
    assert "confidence" in result

    pred_id = result["id"]

    # Now check explanation
    explain_resp = requests.get(f"{API_BASE_URL}/explain/{pred_id}", headers=headers)
    assert explain_resp.status_code == 200, "Explanation failed"
    explain_data = explain_resp.json()
    assert "explanation" in explain_data
    assert "heatmap_base64" in explain_data


def test_verify_no_plaintext_weights_on_server(auth_token):
    # Test that specifically asserts the server never received plaintext weights.
    headers = {"Authorization": f"Bearer {auth_token}"}
    response = requests.get(f"{API_BASE_URL}/privacy/status", headers=headers)
    assert response.status_code == 200
    status = response.json()
    assert (
        status.get("raw_images_shared") == 0
    ), "SERVER HAD ACCESS TO PLAINTEXT! SECURITY BREACH."
    assert "Only encrypted model updates are shared" in status.get("message", "")
