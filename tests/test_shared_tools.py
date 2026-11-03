"""Offline contracts for the shared visual search tools."""

import io
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from vissearch_bench.errors import AgentError
from vissearch_bench.react.agent import ToolCall
from vissearch_bench.react.tools import ReActToolbox, TOOL_SCHEMAS, _PinnedHttpSession


SECRET = "SERPAPI-SECRET-for-tests"
TEMP_IMAGE_ID = "temporary-upload-id"


class Response:
    def __init__(self, payload=None, *, content=b"", status=200):
        self.payload, self.content = payload, content
        self.status_code, self.headers = status, {}
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}: {SECRET}")

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    def iter_content(self, **kwargs):
        yield self.content

    def close(self):
        self.closed = True


def png_bytes(size=(80, 40), noisy=False):
    if noisy:
        image = Image.frombytes("RGB", size, random.Random(42).randbytes(size[0] * size[1] * 3))
    else:
        image = Image.new("RGB", size, (40, 110, 170))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def lens_payload(count=1):
    return {
        "search_metadata": {"status": "Success"},
        "search_parameters": {"image_id": TEMP_IMAGE_ID, "api_key": SECRET},
        "visual_matches": [
            {"title": f"Match {i}", "link": f"https://source.example/{i}",
             "thumbnail": f"https://images.example/{i}.png", "source": "Example"}
            for i in range(count)
        ],
    }


class Service:
    def __init__(self, *, upload=None, search=None, image=None):
        self.upload = upload or Response({"image_id": TEMP_IMAGE_ID})
        self.search = search or Response(lens_payload())
        self.image = image or Response(content=png_bytes())
        self.posts, self.searches, self.downloads = [], [], []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return self.upload

    def get_json(self, url, **kwargs):
        self.searches.append((url, kwargs))
        return self.search

    def get(self, url, **kwargs):
        self.downloads.append((url, kwargs))
        return self.image


def toolbox(tmp_path, service=None, **kwargs):
    source = tmp_path / "input.png"
    if not source.exists():
        source.write_bytes(png_bytes())
    options = dict(
        image_base_dir=tmp_path, image_references=["input.png"],
        serper_api_key="serper-key", serper_base_url="https://serper.example",
        serpapi_api_key=SECRET, serpapi_base_url="https://serpapi.example",
        jina_api_key="", jina_base_url="https://jina.example",
        http_session=service or Service(), host_resolver=lambda host: ["93.184.216.34"],
    )
    options.update(kwargs)
    return ReActToolbox(**options)


def reverse(box, image_id="input:1", **kwargs):
    return box.execute(ToolCall("r", "reverse_image_search", {"image_id": image_id, **kwargs}))


def crop(box, image_id="input:1"):
    return box.execute(ToolCall("c", "crop_image", {
        "image_id": image_id, "x1": 0, "y1": 0, "x2": 500, "y2": 500,
    }))


def test_schema_exposes_five_shared_tools_and_registered_image_ids():
    schemas = {schema["name"]: schema for schema in TOOL_SCHEMAS}
    assert set(schemas) == {
        "web_search", "open_page", "image_search", "crop_image", "reverse_image_search", "submit_answer",
    }
    reverse_schema = schemas["reverse_image_search"]["input_schema"]
    assert reverse_schema["required"] == ["image_id"]
    assert reverse_schema["properties"]["num_results"]["maximum"] == 5
    assert schemas["image_search"]["input_schema"]["properties"]["num_results"]["maximum"] == 5
    assert reverse_schema["additionalProperties"] is False
    for image_id in ("input:N", "search:N", "crop:N"):
        assert image_id in schemas["reverse_image_search"]["description"]


def test_reverse_upload_lens_contract_and_normalized_observation(tmp_path):
    service = Service()
    box = toolbox(tmp_path, service)
    result = reverse(box)
    assert len(service.posts) == len(service.searches) == 1
    url, upload = service.posts[0]
    assert url == "https://serpapi.example/image"
    assert upload["data"] == {"api_key": SECRET}
    filename, content, mime = upload["files"]["image"]
    assert (filename, mime) == ("image.jpg", "image/jpeg")
    assert content.startswith(b"\xff\xd8") and len(content) <= 500_000
    assert upload["allow_redirects"] is False
    assert service.searches[0][0] == "https://serpapi.example/search.json"
    assert service.searches[0][1]["params"] == {
        "engine": "google_lens", "image_id": TEMP_IMAGE_ID,
        "api_key": SECRET, "hl": "en", "type": "all",
    }
    assert service.searches[0][1]["allow_redirects"] is False
    assert result.metadata["sources"] == ["https://source.example/0"]
    assert result.metadata["result_count"] == result.metadata["images_loaded"] == 1
    assert result.metadata["loaded_image_results"][0]["image_id"] == "search:1"
    assert "search:1" in box.images and len(result.images) == 1
    assert service.upload.closed and service.search.closed


def test_large_reverse_upload_preserves_aspect_ratio_and_size_limit(tmp_path):
    (tmp_path / "input.png").write_bytes(png_bytes((1800, 1200), noisy=True))
    service = Service()
    box = toolbox(tmp_path, service)
    assert len(box.images["input:1"].source_jpeg_bytes) > 500_000
    reverse(box)
    uploaded = service.posts[0][1]["files"]["image"][1]
    assert len(uploaded) <= 500_000
    with Image.open(io.BytesIO(uploaded)) as image:
        assert abs(image.width / image.height - 1.5) < 0.01
        assert max(image.size) <= box.image_max_side


def test_crop_registers_chain_and_reverse_uses_crop(tmp_path):
    service = Service()
    box = toolbox(tmp_path, service)
    first, second = crop(box), None
    assert first.metadata["image_id"] == "input:1"  # Preserve the legacy field.
    assert first.metadata["output_image_id"] == "crop:1"
    assert first.metadata["parent_image_id"] == "input:1"
    assert first.metadata["crop_size"] == [40, 20]
    assert first.metadata["source"] == "input.png"
    assert "crop:1" in first.text and "crop:1" in box.images
    second = crop(box, "crop:1")
    assert second.metadata["output_image_id"] == "crop:2"
    assert second.metadata["crop_size"] == [20, 10]
    reverse(box, "crop:2")
    with Image.open(io.BytesIO(service.posts[0][1]["files"]["image"][1])) as uploaded:
        assert uploaded.size == (20, 10)


def test_content_cache_reuses_paid_calls_across_ids_and_instances(tmp_path):
    cache_dir = tmp_path / "cache"
    first_service = Service()
    box = toolbox(tmp_path, first_service, tool_cache_dir=cache_dir)
    first = reverse(box)
    box.images["input:2"] = box.images["input:1"]
    second = reverse(box, "input:2")
    third_service = Service()
    third_box = toolbox(tmp_path, third_service, tool_cache_dir=cache_dir)
    third = reverse(third_box)
    assert not first.metadata["cache_hit"]
    assert second.metadata["cache_hit"] and third.metadata["cache_hit"]
    assert len(first_service.posts) == len(first_service.searches) == 1
    assert not third_service.posts and not third_service.searches
    cache_text = "".join(path.read_text() for path in cache_dir.rglob("*.json"))
    assert SECRET not in cache_text and TEMP_IMAGE_ID not in cache_text
    assert "image_sha256" in cache_text and "google_lens" in cache_text


def test_in_memory_cache_reuses_paid_calls_without_disk_cache(tmp_path):
    service = Service()
    box = toolbox(tmp_path, service)
    reverse(box)
    assert reverse(box).metadata["cache_hit"] is True
    assert len(service.posts) == len(service.searches) == 1


@pytest.mark.parametrize("stage,payload,status", [
    ("upload", {"error": SECRET}, 200),
    ("upload", {}, 200),
    ("upload", {"image_id": 123}, 200),
    ("upload", {"image_id": TEMP_IMAGE_ID}, 302),
    ("upload", {"image_id": TEMP_IMAGE_ID}, 429),
    ("search", {"error": SECRET}, 200),
    ("search", {}, 200),
    ("search", {"search_metadata": {"status": "Error"}, "visual_matches": []}, 200),
    ("search", {"search_metadata": {"status": "Success"}}, 200),
    ("search", {"search_metadata": {"status": "Success"}, "visual_matches": "bad"}, 200),
    ("search", {"search_metadata": {"status": "Success"}, "visual_matches": ["bad"]}, 200),
    ("search", {"search_metadata": {"status": "Success"}, "visual_matches": [{}]}, 200),
    ("search", ValueError(SECRET), 200),
])
def test_service_errors_are_infrastructure_and_not_cached_or_leaked(tmp_path, stage, payload, status):
    service = Service(**{stage: Response(payload, status=status)})
    box = toolbox(tmp_path, service, tool_cache_dir=tmp_path / "cache")
    with pytest.raises(AgentError) as caught:
        reverse(box)
    assert caught.value.infrastructure
    assert SECRET not in str(caught.value) and SECRET not in caught.value.code
    if status >= 300:
        assert f"HTTP{status}" in caught.value.code
    assert not list((tmp_path / "cache").rglob("*.json"))
    assert len(service.posts) == 1
    assert len(service.searches) == (1 if stage == "search" else 0)


def test_valid_empty_results_are_distinct_and_cached(tmp_path):
    service = Service(search=Response(lens_payload(0)))
    box = toolbox(tmp_path, service)
    first = reverse(box)
    assert first.metadata["status"] == "no_results"
    assert first.metadata["result_count"] == 0 and not first.images
    assert "No reverse image search results" in first.text
    assert reverse(box).metadata["cache_hit"]
    assert not service.downloads


def test_service_echoes_never_leak_credentials_or_temporary_image_id(tmp_path):
    payload = lens_payload()
    payload["visual_matches"][0]["title"] = f"Title {SECRET} {TEMP_IMAGE_ID}"
    service = Service(search=Response(payload))
    result = reverse(toolbox(tmp_path, service, tool_cache_dir=tmp_path / "cache"))
    exposed = result.text + json.dumps(result.metadata)
    exposed += "".join(path.read_text() for path in (tmp_path / "cache").rglob("*.json"))
    assert SECRET not in exposed and TEMP_IMAGE_ID not in exposed


def test_reverse_count_and_download_limits_are_shared(tmp_path):
    service = Service(search=Response(lens_payload(9)))
    box = toolbox(tmp_path, service, max_image_results=10, max_total_images=3)
    result = reverse(box, num_results=99)
    assert result.metadata["result_count"] == 5
    assert result.metadata["images_loaded"] == 2
    assert len(service.downloads) == 2
    assert box.total_images_used == 3


def test_text_image_search_also_caps_results_at_five(tmp_path):
    rows = [{"title": str(i), "link": f"https://source.example/{i}",
             "imageUrl": f"https://images.example/{i}.png"} for i in range(9)]
    service = Service(upload=Response({"images": rows}))
    box = toolbox(tmp_path, service, max_image_results=10)
    result = box.execute(ToolCall("s", "image_search", {"query": "public", "num_results": 99}))
    assert service.posts[0][1]["json"]["num"] == 5
    assert result.metadata["result_count"] == result.metadata["images_loaded"] == 5
    # Search results use the same registry as input images and cropped images.
    service.upload = Response({"image_id": TEMP_IMAGE_ID})
    reverse(box, "search:1")
    assert len(service.searches) == 1


def test_reverse_rejects_malformed_match_urls(tmp_path):
    payload = lens_payload()
    payload["visual_matches"][0]["link"] = "not a URL"
    with pytest.raises(AgentError) as caught:
        reverse(toolbox(tmp_path, Service(search=Response(payload))))
    assert caught.value.infrastructure


def test_reverse_requires_credentials_before_network(tmp_path):
    service = Service()
    with pytest.raises(AgentError) as caught:
        reverse(toolbox(tmp_path, service, serpapi_api_key=""))
    assert caught.value.infrastructure
    assert not service.posts and not service.searches


def test_cache_is_scoped_to_provider_endpoint(tmp_path):
    cache_dir = tmp_path / "cache"
    reverse(toolbox(tmp_path, tool_cache_dir=cache_dir))
    other_service = Service()
    other_box = toolbox(tmp_path, other_service, tool_cache_dir=cache_dir,
                        serpapi_base_url="https://other-serpapi.example")
    assert reverse(other_box).metadata["cache_hit"] is False
    assert len(other_service.posts) == len(other_service.searches) == 1


def test_reverse_rejects_oversized_image_downloads(tmp_path):
    service = Service(image=Response(content=b"x" * 2048))
    box = toolbox(tmp_path, service, max_download_bytes=1024)
    result = reverse(box)
    assert result.metadata["images_loaded"] == 0
    assert set(box.images) == {"input:1"}


def test_reverse_rejects_image_pixel_limit(tmp_path):
    service = Service(image=Response(content=png_bytes((100, 100))))
    box = toolbox(tmp_path, service, max_image_pixels=5000)
    assert reverse(box).metadata["images_loaded"] == 0
    assert set(box.images) == {"input:1"}


def test_reverse_download_infrastructure_failure_is_not_swallowed(tmp_path):
    service = Service()
    def fail(*args, **kwargs):
        raise TimeoutError(SECRET)
    service.get = fail
    with pytest.raises(AgentError) as caught:
        reverse(toolbox(tmp_path, service))
    assert caught.value.infrastructure and SECRET not in str(caught.value)
    assert len(service.posts) == len(service.searches) == 1


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "2"])
def test_reverse_rejects_bad_arguments_before_paid_calls(tmp_path, count):
    service = Service()
    box = toolbox(tmp_path, service)
    with pytest.raises(ValueError):
        reverse(box, num_results=count)
    with pytest.raises(ValueError):
        reverse(box, "https://example.org/unregistered.png")
    assert not service.posts and not service.searches


def test_crop_image_budget_applies_to_registered_outputs(tmp_path):
    box = toolbox(tmp_path, max_total_images=2)
    crop(box)
    with pytest.raises(ValueError, match="image budget"):
        crop(box, "crop:1")
    assert set(box.images) == {"input:1", "crop:1"}


def test_trusted_json_get_does_not_use_pinned_asset_get():
    session = _PinnedHttpSession(host_resolver=lambda host: ["93.184.216.34"])
    calls = []
    expected = Response(lens_payload())
    session._post_session = SimpleNamespace(get=lambda url, **kw: calls.append((url, kw)) or expected)
    result = session.get_json("https://serpapi.example/search.json", params={"api_key": SECRET}, timeout=30, allow_redirects=False)
    assert result is expected and len(calls) == 1


@pytest.mark.parametrize("tool_name,result_field", [("web_search", "organic"), ("image_search", "images")])
@pytest.mark.parametrize("case", ["error", "error_with_results", "missing", "null", "non_object", "bad_rows"])
def test_serper_service_or_malformed_payload_is_infrastructure_not_empty_or_cached(tmp_path, tool_name, result_field, case):
    payloads = {
        "error": {"error": SECRET},
        "error_with_results": {"error": SECRET, result_field: []},
        "missing": {},
        "null": {result_field: None},
        "non_object": [SECRET],
        "bad_rows": {result_field: [SECRET]},
    }
    service = Service(upload=Response(payloads[case]))
    cache_dir = tmp_path / "cache"
    box = toolbox(tmp_path, service, tool_cache_dir=cache_dir)
    with pytest.raises(AgentError) as caught:
        box.execute(ToolCall("s", tool_name, {"query": "public"}))
    assert caught.value.infrastructure
    assert SECRET not in str(caught.value) and SECRET not in caught.value.code
    assert len(service.posts) == 1
    assert not list(cache_dir.rglob("*.json"))


@pytest.mark.parametrize("tool_name,result_field", [("web_search", "organic"), ("image_search", "images")])
def test_serper_explicit_empty_results_remain_valid_and_cacheable(tmp_path, tool_name, result_field):
    service = Service(upload=Response({result_field: []}))
    box = toolbox(tmp_path, service, tool_cache_dir=tmp_path / "cache")
    call = ToolCall("s", tool_name, {"query": "public"})
    first, second = box.execute(call), box.execute(call)
    assert first.metadata["result_count"] == second.metadata["result_count"] == 0
    assert second.metadata["cache_hit"] is True
    assert len(service.posts) == 1


@pytest.mark.parametrize("tool_name,result_field", [("web_search", "organic"), ("image_search", "images")])
def test_serper_json_decode_error_is_safe_infrastructure_and_closes_response(tmp_path, tool_name, result_field):
    from requests.exceptions import JSONDecodeError

    response = Response(JSONDecodeError(SECRET, f"<html>{SECRET}</html>", 1))
    service = Service(upload=response)
    cache_dir = tmp_path / "cache"
    box = toolbox(tmp_path, service, tool_cache_dir=cache_dir)
    with pytest.raises(AgentError) as caught:
        box.execute(ToolCall("s", tool_name, {"query": "public"}))
    assert caught.value.infrastructure
    assert SECRET not in str(caught.value) and SECRET not in caught.value.code
    assert response.closed and len(service.posts) == 1
    assert not list(cache_dir.rglob("*.json"))


@pytest.mark.parametrize("tool_name,result_field", [("web_search", "organic"), ("image_search", "images")])
@pytest.mark.parametrize("status", [302, 401, 429, 503])
def test_serper_unsuccessful_http_status_is_safe_infrastructure_and_closes_response(tmp_path, tool_name, result_field, status):
    response = Response({result_field: []}, status=status)
    service = Service(upload=response)
    cache_dir = tmp_path / "cache"
    box = toolbox(tmp_path, service, tool_cache_dir=cache_dir)
    with pytest.raises(AgentError) as caught:
        box.execute(ToolCall("s", tool_name, {"query": "public"}))
    assert caught.value.infrastructure and f"HTTP{status}" in caught.value.code
    assert SECRET not in str(caught.value)
    assert response.closed and len(service.posts) == 1
    assert not list(cache_dir.rglob("*.json"))


@pytest.mark.parametrize("tool_name,result_field", [("web_search", "organic"), ("image_search", "images")])
def test_serper_closes_successful_response(tmp_path, tool_name, result_field):
    response = Response({result_field: []})
    box = toolbox(tmp_path, Service(upload=response))
    box.execute(ToolCall("s", tool_name, {"query": "public"}))
    assert response.closed
