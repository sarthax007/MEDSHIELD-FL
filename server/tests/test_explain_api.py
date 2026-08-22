import io
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def test_explain_endpoint_with_valid_image():
    # Create a dummy image (e.g. 224x224 grayscale)
    image = Image.new("L", (224, 224), color=128)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    # Post it to the endpoint
    response = client.post(
        "/explain", files={"file": ("test_scan.jpg", img_byte_arr, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "explanation" in data
    assert "heatmap_base64" in data
    assert data["prediction"] in ["Healthy", "Tumor"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["explanation"], str)
    assert (
        data["heatmap_base64"].startswith("/9j/") or data["heatmap_base64"] != ""
    )  # basic check for base64 string


def test_explain_endpoint_with_invalid_extension():
    # Try uploading a text file
    response = client.post(
        "/explain", files={"file": ("test.txt", b"dummy content", "text/plain")}
    )
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file type" in data["detail"]


def test_explain_endpoint_no_file():
    # Call without file
    response = client.post("/explain")
    # FastAPI's default for missing file is 422 Unprocessable Entity
    assert response.status_code == 422
