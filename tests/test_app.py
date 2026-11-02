import io
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app  # noqa: E402


def client():
    app.config["TESTING"] = True
    return app.test_client()


def test_health_endpoint():
    response = client().get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_dashboard_page_loads():
    response = client().get("/")
    assert response.status_code == 200
    assert b"<html" in response.data.lower()


def test_upload_rejects_non_csv():
    data = {"file": (io.BytesIO(b"not a csv"), "notes.txt")}
    response = client().post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["error"] == "CSV only"


def test_sample_dataset_has_required_columns():
    import pandas as pd
    from app import DEFAULT_DATA_PATH, FEATURES, TARGET

    df = pd.read_csv(DEFAULT_DATA_PATH, nrows=5)
    for column in FEATURES + [TARGET]:
        assert column in df.columns
