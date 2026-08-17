import json
import os

BRUNO_COLLECTION = os.path.join(os.path.dirname(__file__), "..", "bruno", "collection.json")
ALLOWED_TYPES = {
    "http-request",
    "graphql-request",
    "folder",
    "js",
    "app",
    "grpc-request",
    "ws-request",
}


def _load_collection():
    with open(BRUNO_COLLECTION) as f:
        return json.load(f)


def test_collection_version_is_1():
    assert _load_collection()["version"] == "1"


def test_collection_has_name_and_items():
    data = _load_collection()
    assert data["name"]
    assert isinstance(data["items"], list)
    assert len(data["items"]) >= 4


def test_item_types_are_valid_bruno_types():
    for item in _load_collection()["items"]:
        assert item["type"] in ALLOWED_TYPES


def test_http_requests_have_headers_params_and_body():
    for item in _load_collection()["items"]:
        assert "request" in item
        request = item["request"]
        assert isinstance(request["headers"], list)
        assert isinstance(request["params"], list)
        assert request["body"]["mode"] in {"none", "json", "text", "xml", "formUrlEncoded", "multipartForm"}


def test_predict_and_batch_use_json_body():
    requests = {item["name"]: item["request"] for item in _load_collection()["items"]}
    assert requests["predict"]["body"]["mode"] == "json"
    assert requests["predict_batch"]["body"]["mode"] == "json"