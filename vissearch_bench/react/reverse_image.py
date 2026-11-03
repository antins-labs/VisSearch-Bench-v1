"""Single-upload, single-search SerpApi Google Lens integration."""

from __future__ import annotations

import io
from typing import Any, Callable
from urllib.parse import urlparse

from PIL import Image

from ..errors import AgentError


MAX_UPLOAD_BYTES = 500_000
MAX_MATCHES = 5
LENS_PARAMETERS = {"engine": "google_lens", "hl": "en", "type": "all"}


def prepare_upload_jpeg(source_jpeg: bytes, *, max_side: int) -> bytes:
    """Bound an already decoded/validated image without changing its aspect ratio."""
    with Image.open(io.BytesIO(source_jpeg)) as source:
        image = source.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        while True:
            for quality in (85, 70, 55, 40):
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
                encoded = buffer.getvalue()
                if len(encoded) <= MAX_UPLOAD_BYTES:
                    return encoded
            image.thumbnail(
                (max(1, image.width // 2), max(1, image.height // 2)),
                Image.Resampling.LANCZOS,
            )


def service_base_url(value: str) -> str:
    """The configured service endpoint must not contain URL credentials."""
    parsed = urlparse(value)
    if (
        parsed.scheme not in {"https", "http"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise AgentError("SerpApiInvalidEndpoint", infrastructure=True)
    return value.rstrip("/")


def _json_response(request: Callable[[], Any], *, stage: str) -> dict:
    response = None
    try:
        response = request()
        status = response.status_code
        if not isinstance(status, int):
            raise AgentError("SerpApi{}MalformedStatus".format(stage), infrastructure=True)
        if not 200 <= status < 300:
            raise AgentError("SerpApi{}HTTP{}".format(stage, status), infrastructure=True)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise AgentError("SerpApi{}MalformedResponse".format(stage), infrastructure=True)
        if "error" in payload:
            raise AgentError("SerpApi{}ServiceError".format(stage), infrastructure=True)
        return payload
    except AgentError:
        raise
    except Exception:
        # Provider exceptions may include the query URL (and API key).
        raise AgentError("SerpApi{}RequestFailure".format(stage), infrastructure=True) from None
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def validate_cached_matches(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != {"matches"}:
        raise ValueError("invalid cached Lens matches")
    matches = payload["matches"]
    if not isinstance(matches, list) or len(matches) > MAX_MATCHES:
        raise ValueError("invalid cached Lens matches")
    fields = {"title", "link", "imageUrl", "thumbnailUrl"}
    if any(
        not isinstance(row, dict)
        or set(row) != fields
        or any(not isinstance(value, str) for value in row.values())
        for row in matches
    ):
        raise ValueError("invalid cached Lens match")


def fetch_visual_matches(
    http: Any,
    *,
    base_url: str,
    api_key: str,
    upload_jpeg: bytes,
    timeout: float,
) -> dict:
    if not api_key:
        raise AgentError("MissingSerpApiCredential", infrastructure=True)
    upload = _json_response(
        lambda: http.post(
            base_url + "/image",
            files={"image": ("image.jpg", upload_jpeg, "image/jpeg")},
            data={"api_key": api_key},
            timeout=timeout,
            allow_redirects=False,
        ),
        stage="Upload",
    )
    image_id = upload.get("image_id")
    if not isinstance(image_id, str) or not image_id.strip():
        raise AgentError("SerpApiUploadMalformedResponse", infrastructure=True)
    # Asset GETs use pinned IPs and streaming; only this configured service gets
    # JSON with query parameters. Ordinary injected requests sessions also work.
    get_json = getattr(http, "get_json", None) or http.get
    payload = _json_response(
        lambda: get_json(
            base_url + "/search.json",
            params={**LENS_PARAMETERS, "image_id": image_id, "api_key": api_key},
            timeout=timeout,
            allow_redirects=False,
        ),
        stage="Search",
    )
    metadata = payload.get("search_metadata")
    matches = payload.get("visual_matches")
    if (
        not isinstance(metadata, dict)
        or metadata.get("status") != "Success"
        or not isinstance(matches, list)
        or any(not isinstance(row, dict) for row in matches)
    ):
        raise AgentError("SerpApiSearchMalformedResponse", infrastructure=True)

    def clean_text(value: Any, *, url: bool = False) -> str:
        if not isinstance(value, str):
            raise AgentError("SerpApiSearchMalformedMatch", infrastructure=True)
        for secret in (api_key, image_id):
            if secret in value:
                if url:
                    return ""
                value = value.replace(secret, "[redacted]")
        return value

    normalized = []
    for row in matches[:MAX_MATCHES]:
        link = row.get("link")
        if not isinstance(link, str) or not link.strip():
            raise AgentError("SerpApiSearchMalformedMatch", infrastructure=True)
        try:
            parsed = urlparse(link)
            valid_link = (
                parsed.scheme in {"http", "https"}
                and parsed.hostname
                and parsed.username is None
                and parsed.password is None
            )
        except ValueError:
            valid_link = False
        if not valid_link:
            raise AgentError("SerpApiSearchMalformedMatch", infrastructure=True)
        normalized.append({
            "title": clean_text(row.get("title", "Untitled")),
            "link": clean_text(link, url=True),
            "imageUrl": clean_text(row.get("image", ""), url=True),
            "thumbnailUrl": clean_text(row.get("thumbnail", ""), url=True),
        })
    return {"matches": normalized}
