"""Bounded web and image tools used by the ReAct evaluator."""

from __future__ import annotations

import base64
import hashlib
import io
import ipaddress
import json
import socket
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.parse import urljoin, urlparse, urlunsplit

from PIL import Image, ImageOps

from .compat import answer_to_text, resolve_local_image
from .agent import ToolCall, ToolObservation
from .cache import JsonToolCache
from .reverse_image import (
    LENS_PARAMETERS,
    MAX_MATCHES,
    fetch_visual_matches,
    prepare_upload_jpeg,
    service_base_url,
    validate_cached_matches,
)
from ..errors import AgentError, safe_exception, http_status


class _SearchResponseError(AgentError, ValueError):
    """Infrastructure failure that also invalidates an existing JSON cache entry."""

    def __init__(self, message: str, *, code: Optional[str] = None):
        super().__init__(message, infrastructure=True, code=code)


class _DownloadBudgetExceeded(ValueError):
    """An ordinary per-record resource limit, not a provider outage."""


class _PageUnavailable(ValueError):
    """The scraper explicitly could not retrieve this particular target."""


def _is_public_address(address: ipaddress._BaseAddress) -> bool:
    return bool(
        address.is_global
        and not address.is_multicast
        and not address.is_unspecified
        and not address.is_reserved
        and not address.is_loopback
        and not address.is_link_local
        and not address.is_private
    )


def _public_addresses(
    hostname: str,
    resolver: Callable[[str], Sequence[Any]],
    *,
    label: str,
) -> List[ipaddress._BaseAddress]:
    try:
        literal = ipaddress.ip_address(hostname.split("%", 1)[0])
        addresses = [literal]
    except ValueError:
        try:
            resolved = resolver(hostname)
        except (OSError, socket.gaierror):
            raise AgentError('DNSResolutionFailure', infrastructure=True) from None
        addresses = []
        for item in resolved:
            raw_address = item if isinstance(item, str) else item[4][0]
            try:
                addresses.append(
                    ipaddress.ip_address(
                        str(raw_address).split("%", 1)[0]
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    "{} resolved to an invalid address".format(label)
                ) from exc
    if not addresses or any(
        not _is_public_address(address) for address in addresses
    ):
        raise ValueError(
            "{} must resolve only to public addresses".format(label)
        )
    return addresses


class _PinnedHttpResponse:
    def __init__(self, response: Any, pool: Any):
        self._response = response
        self._pool = pool
        self.status_code = int(response.status)
        self.headers = response.headers

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError("HTTP {}".format(self.status_code))

    def iter_content(self, chunk_size: int = 64 * 1024):
        yield from self._response.stream(
            chunk_size,
            decode_content=True,
        )

    def close(self) -> None:
        try:
            self._response.release_conn()
        finally:
            close = getattr(self._pool, "close", None)
            if callable(close):
                close()


class _PinnedHttpSession:
    """GET transport that connects to the exact public IP it validated."""

    def __init__(
        self,
        *,
        host_resolver: Callable[[str], Sequence[Any]],
        pool_factory: Optional[Callable[..., Any]] = None,
    ):
        import requests

        self._post_session = requests.Session()
        # The adapter passes any operator proxy through the explicit tool
        # transport, including inside the isolated DeerFlow worker.
        self._post_session.trust_env = False
        self.host_resolver = host_resolver
        self.pool_factory = pool_factory or self._default_pool_factory

    @staticmethod
    def _default_pool_factory(
        *,
        scheme: str,
        ip_address: str,
        port: int,
        tls_hostname: str,
        timeout: float,
    ) -> Any:
        import certifi
        import urllib3

        if scheme == "https":
            return urllib3.HTTPSConnectionPool(
                ip_address,
                port=port,
                timeout=timeout,
                maxsize=1,
                block=True,
                retries=False,
                cert_reqs="CERT_REQUIRED",
                ca_certs=certifi.where(),
                assert_hostname=tls_hostname,
                server_hostname=tls_hostname,
            )
        return urllib3.HTTPConnectionPool(
            ip_address,
            port=port,
            timeout=timeout,
            maxsize=1,
            block=True,
            retries=False,
        )

    def post(self, url: str, **kwargs: Any) -> Any:
        return self._post_session.post(url, **kwargs)

    def get_json(self, url: str, **kwargs: Any) -> Any:
        """JSON GET for a configured API, separate from untrusted asset GETs."""
        return self._post_session.get(url, **kwargs)

    def get(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        timeout: float,
        stream: bool,
        allow_redirects: bool,
    ) -> _PinnedHttpResponse:
        if not stream or allow_redirects:
            raise ValueError("pinned GET requires manual streaming redirects")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("pinned GET requires an HTTP(S) URL")
        hostname = parsed.hostname.rstrip(".").casefold()
        addresses = _public_addresses(
            hostname,
            self.host_resolver,
            label="GET URL",
        )
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        default_port = 443 if parsed.scheme == "https" else 80
        host_label = "[{}]".format(hostname) if ":" in hostname else hostname
        request_headers = dict(headers)
        request_headers["Host"] = (
            host_label
            if port == default_port
            else "{}:{}".format(host_label, port)
        )
        target = urlunsplit(
            ("", "", parsed.path or "/", parsed.query, "")
        )
        last_error: Optional[Exception] = None
        for address in addresses:
            pool = self.pool_factory(
                scheme=parsed.scheme,
                ip_address=str(address),
                port=port,
                tls_hostname=hostname,
                timeout=timeout,
            )
            try:
                response = pool.urlopen(
                    "GET",
                    target,
                    headers=request_headers,
                    redirect=False,
                    preload_content=False,
                    retries=False,
                )
                return _PinnedHttpResponse(response, pool)
            except Exception as exc:
                last_error = exc
                close = getattr(pool, "close", None)
                if callable(close):
                    close()
        # Keep transport failures identifiable without persisting a URL/token
        # from the original exception; unknown implementation errors stay fatal.
        from urllib3.exceptions import HTTPError
        if isinstance(last_error, (HTTPError, OSError)):
            raise AgentError('AssetRequestFailure', infrastructure=True) from None
        raise RuntimeError(
            "GET failed for all validated public addresses: {}".format(
                last_error
            )
        ) from last_error


@dataclass(frozen=True)
class StoredImage:
    image_id: str
    source_jpeg_bytes: bytes
    preview_jpeg_bytes: bytes
    source: str


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "web_search",
        "description": (
            "Search the public web. Use focused queries and inspect promising "
            "results with open_page."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "num_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "open_page",
        "description": (
            "Read a public HTTP or HTTPS webpage as bounded plain text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
            "additionalProperties": False,
        },
    },
    {
        "name": "image_search",
        "description": (
            "Search public images. Returned images are immediately shown in "
            "the observation and assigned IDs such as search:1."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "num_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "crop_image",
        "description": (
            "Crop and enlarge an input, search, or crop image; return a new "
            "crop:N image ID for further crops or reverse search. Coordinates are "
            "normalized from 0 to 1000. Initial image IDs are input:1, input:2, "
            "and so on."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string"},
                "x1": {"type": "number", "minimum": 0, "maximum": 1000},
                "y1": {"type": "number", "minimum": 0, "maximum": 1000},
                "x2": {"type": "number", "minimum": 0, "maximum": 1000},
                "y2": {"type": "number", "minimum": 0, "maximum": 1000},
            },
            "required": ["image_id", "x1", "y1", "x2", "y2"],
            "additionalProperties": False,
        },
    },
    {
        "name": "reverse_image_search",
        "description": (
            "Find visual matches for a registered image using Google Lens. "
            "Use an input:N, search:N, or crop:N image ID. Matching images "
            "are shown in the observation with new search:N IDs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "image_id": {"type": "string"},
                "num_results": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["image_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit_answer",
        "description": (
            "Finish the task. Supply the concise final answer and every "
            "specific concrete intermediate entity used for a requested count "
            "or set. Use an empty list when no entity set is requested."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "final_answer": {},
                "intermediate_answers": {
                    "type": "array",
                    "items": {},
                },
            },
            "required": ["final_answer", "intermediate_answers"],
            "additionalProperties": False,
        },
    },
]


class ReActToolbox:
    def __init__(
        self,
        *,
        image_base_dir: Path,
        image_references: Sequence[str],
        serper_api_key: str,
        serper_base_url: str,
        jina_api_key: str,
        jina_base_url: str,
        serper_scrape_base_url: Optional[str] = None,
        serpapi_api_key: str = "",
        serpapi_base_url: str = "https://serpapi.com",
        http_session: Optional[Any] = None,
        timeout: float = 30.0,
        image_max_side: int = 1536,
        max_search_results: int = 5,
        max_image_results: int = 5,
        max_page_chars: int = 12000,
        max_download_bytes: int = 8 * 1024 * 1024,
        max_local_image_bytes: int = 64 * 1024 * 1024,
        max_total_download_bytes: int = 64 * 1024 * 1024,
        max_image_pixels: int = 25_000_000,
        max_total_images: int = 20,
        max_crop_upscale: float = 4.0,
        max_redirects: int = 3,
        tool_cache_dir: Optional[Path] = None,
        host_resolver: Optional[Callable[[str], Sequence[Any]]] = None,
    ):
        self.image_base_dir = Path(image_base_dir)
        self.serper_api_key = serper_api_key
        self.serper_base_url = serper_base_url.rstrip("/")
        self.jina_api_key = jina_api_key
        self.jina_base_url = jina_base_url.rstrip("/")
        # None preserves the copied legacy toolbox contract. Current adapters
        # always supply the Serper endpoint and never fall back to Jina.
        self.serper_scrape_base_url = serper_scrape_base_url
        self.serpapi_api_key = serpapi_api_key
        self.serpapi_base_url = serpapi_base_url.rstrip("/")
        self.timeout = timeout
        self.image_max_side = image_max_side
        self.max_search_results = max_search_results
        self.max_image_results = max_image_results
        self.max_page_chars = max_page_chars
        self.max_download_bytes = max_download_bytes
        self.max_local_image_bytes = max_local_image_bytes
        self.max_total_download_bytes = max_total_download_bytes
        self.max_image_pixels = max_image_pixels
        self.max_total_images = max_total_images
        self.max_crop_upscale = max_crop_upscale
        self.max_redirects = max_redirects
        self.tool_cache = (
            JsonToolCache(tool_cache_dir)
            if tool_cache_dir is not None
            else None
        )
        self.host_resolver = host_resolver or self._system_resolve
        if http_session is None:
            http_session = _PinnedHttpSession(
                host_resolver=self.host_resolver,
            )
        self.http = http_session
        self.images: Dict[str, StoredImage] = {}
        self.next_search_image = 1
        self.next_crop_image = 1
        self.reverse_image_cache: Dict[str, Any] = {}
        self.total_download_bytes = 0
        self.total_images_used = 0
        for index, reference in enumerate(image_references, start=1):
            self._register_initial_image("input:{}".format(index), reference)

    @property
    def tool_schemas(self) -> List[Dict[str, Any]]:
        return TOOL_SCHEMAS

    @property
    def initial_image_data_urls(self) -> List[str]:
        values = []
        for index in range(1, len(self.images) + 1):
            stored = self.images.get("input:{}".format(index))
            if stored is None:
                break
            values.append(self._data_url(stored.preview_jpeg_bytes))
        return values

    def execute(self, call: ToolCall) -> ToolObservation:
        handlers = {
            "web_search": self._web_search,
            "open_page": self._open_page,
            "image_search": self._image_search,
            "crop_image": self._crop_image,
            "reverse_image_search": self._reverse_image_search,
        }
        if call.name == "submit_answer":
            raise ValueError("submit_answer is handled by the agent loop")
        handler = handlers.get(call.name)
        if handler is None:
            raise ValueError("unknown ReAct tool: {}".format(call.name))
        output = handler(call.arguments)
        return ToolObservation(
            tool_call_id=call.id,
            tool_name=call.name,
            text=output["text"],
            images=output.get("images") or [],
            metadata=output.get("metadata") or {},
        )

    def _register_initial_image(self, image_id: str, reference: str) -> None:
        value = answer_to_text(reference)
        local_path = resolve_local_image(value, self.image_base_dir)
        if local_path is not None:
            raw = self._read_local_bytes(local_path)
        else:
            raw = self._download_bytes(value)
        source_jpeg, preview_jpeg = self._prepare_image(raw)
        self._reserve_image()
        self.images[image_id] = StoredImage(
            image_id=image_id,
            source_jpeg_bytes=source_jpeg,
            preview_jpeg_bytes=preview_jpeg,
            source=value,
        )

    def _require_serper_key(self) -> None:
        if not self.serper_api_key:
            raise ValueError("SERPER_API_KEY is required for search tools")

    def _post_serper_json(self, endpoint: str, payload: Dict[str, Any], *, max_bytes: Optional[int] = None, page_target: bool = False) -> Any:
        """Fetch once, classify provider failures safely, and release the response."""
        self._require_serper_key()
        response = None
        try:
            try:
                response = self.http.post(
                    endpoint,
                    headers={
                        "X-API-KEY": self.serper_api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                    allow_redirects=False,
                    **({"stream": True} if max_bytes is not None else {}),
                )
                status = getattr(response, "status_code", 200)
                if not isinstance(status, int):
                    raise _SearchResponseError("SerperMalformedHTTPStatus")
                if not 200 <= status < 300:
                    # Only the observed target-specific scrape failure is recoverable.
                    # Authentication, capacity and generic service errors still abort.
                    if page_target and status == 500:
                        raw = bytearray()
                        for chunk in response.iter_content(chunk_size=1024):
                            if len(raw) + len(chunk) > 8192:
                                break
                            self._consume_download_bytes(len(chunk))
                            raw.extend(chunk)
                        else:
                            try:
                                failure = json.loads(raw)
                            except (ValueError, TypeError):
                                failure = None
                            if failure == {"message": "Scraping failed.", "statusCode": 500}:
                                raise _PageUnavailable("SerperScrapeTargetUnavailable")
                    raise _SearchResponseError(
                        "Serper HTTP {}".format(status),
                        code="SerperHTTP{}".format(status),
                    )
                response.raise_for_status()
                if max_bytes is not None:
                    declared = response.headers.get("content-length")
                    if declared and int(declared) > max_bytes:
                        raise _SearchResponseError("SerperResponseTooLarge")
                    raw = bytearray()
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        if not chunk:
                            continue
                        if len(raw) + len(chunk) > max_bytes:
                            raise _SearchResponseError("SerperResponseTooLarge")
                        self._consume_download_bytes(len(chunk))
                        raw.extend(chunk)
                    return json.loads(raw)
                return response.json()
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        except (_SearchResponseError, _DownloadBudgetExceeded, _PageUnavailable):
            raise
        except Exception as exc:
            # JSONDecodeError is a ValueError, but invalid provider JSON is an
            # infrastructure failure, not evidence about the benchmark question.
            raise _SearchResponseError("Serper{}".format(type(exc).__name__)) from None

    def _bounded_count(self, value: Any, maximum: int) -> int:
        if value is None:
            return maximum
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("num_results must be an integer")
        if value < 1:
            raise ValueError("num_results must be positive")
        return min(value, maximum)

    def _query(self, arguments: Dict[str, Any]) -> str:
        query = answer_to_text(arguments.get("query", ""))
        if not query:
            raise ValueError("query must be non-empty")
        return query

    @staticmethod
    def _validate_result_payload(
        payload: Any,
        *,
        label: str,
        result_field: str,
    ) -> None:
        if not isinstance(payload, dict):
            raise _SearchResponseError("{} response must be a JSON object".format(label))
        if "error" in payload:
            raise _SearchResponseError("{} service returned an error".format(label))
        if result_field not in payload:
            raise _SearchResponseError("{} response is missing {}".format(label, result_field))
        rows = payload.get(result_field)
        if not isinstance(rows, list):
            raise _SearchResponseError(
                "{} {} must be a list".format(label, result_field)
            )
        if any(not isinstance(item, dict) for item in rows):
            raise _SearchResponseError(
                "{} {} items must be JSON objects".format(
                    label,
                    result_field,
                )
            )

    @staticmethod
    def _validate_jina_payload(payload: Any) -> None:
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("text"), str)
        ):
            raise ValueError("Jina page response must contain text")

    def _web_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        query = self._query(arguments)
        count = self._bounded_count(
            arguments.get("num_results"),
            self.max_search_results,
        )
        endpoint = self.serper_base_url + "/search"
        request = {
            "endpoint": endpoint,
            "q": query,
            "num": count,
        }

        def fetch() -> Any:
            return self._post_serper_json(endpoint, {"q": query, "num": count})

        def validate(payload: Any) -> None:
            self._validate_result_payload(
                payload,
                label="Serper web search",
                result_field="organic",
            )

        if self.tool_cache is None:
            payload, cache_hit = fetch(), False
            validate(payload)
        else:
            payload, cache_hit = self.tool_cache.get_or_fetch(
                "serper_web_search",
                request,
                fetch,
                validate,
            )
        organic = payload.get("organic") or []
        lines = []
        sources = []
        for index, item in enumerate(organic[:count], start=1):
            title = answer_to_text(item.get("title", "Untitled"))
            url = answer_to_text(item.get("link", ""))
            snippet = answer_to_text(item.get("snippet", ""))
            lines.append(
                "[{}] {}\nURL: {}\nSnippet: {}".format(
                    index,
                    title,
                    url,
                    snippet,
                )
            )
            if self._is_http_url(url) and url not in sources:
                sources.append(url)
        text = "\n\n".join(lines) or "No web search results."
        return {
            "text": text,
            "metadata": {
                "query": query,
                "sources": sources,
                "result_count": len(lines),
                "cache_hit": cache_hit,
            },
        }

    def _open_page(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if self.serper_scrape_base_url is not None:
            return self._open_serper_page(arguments)
        return self._open_legacy_jina_page(arguments)

    def _open_serper_page(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        url = answer_to_text(arguments.get("url", ""))
        self._validate_public_http_url(url, label="open_page URL")
        endpoint = self.serper_scrape_base_url.rstrip("/")
        request = {"endpoint": endpoint, "url": url, "includeMarkdown": True,
                   "max_page_chars": self.max_page_chars}

        def validate(payload):
            if not isinstance(payload, dict) or "error" in payload:
                raise _SearchResponseError("SerperScrapeServiceError")
            for field in ("markdown", "text"):
                if payload.get(field) is not None and not isinstance(payload[field], str):
                    raise _SearchResponseError("SerperScrapeMalformedResponse")
            if not any(isinstance(payload.get(k), str) and payload[k].strip()
                       for k in ("markdown", "text")):
                raise _SearchResponseError("SerperScrapeEmptyContent")

        def fetch():
            payload = self._post_serper_json(endpoint, {"url": url, "includeMarkdown": True},
                                             max_bytes=self.max_download_bytes, page_target=True)
            validate(payload)
            # Store only evidence, never arbitrary provider fields or credentials.
            text = payload.get("markdown") or payload.get("text", "")
            if not text.strip():
                text = payload["text"]
            return {"text": text}

        try:
            if self.tool_cache is None:
                payload, cache_hit = fetch(), False
            else:
                payload, cache_hit = self.tool_cache.get_or_fetch(
                    "serper_open_page_v1", request, fetch, validate)
        except _PageUnavailable:
            return {"text": "Page unavailable: Serper could not scrape this URL. No page content was retrieved. Use another source.",
                    "metadata": {"sources": [], "requested_url": url, "characters": 0,
                                 "cache_hit": False, "reader_backend": "serper-scrape",
                                 "error": True, "error_code": "SerperScrapeTargetUnavailable", "http_status": 500}}
        text = payload["text"].strip()
        if len(text) > self.max_page_chars:
            text = text[:self.max_page_chars].rstrip() + "\n[truncated]"
        return {"text": text, "metadata": {"sources": [url], "characters": len(text),
                "cache_hit": cache_hit, "reader_backend": "serper-scrape"}}

    def _open_legacy_jina_page(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        url = answer_to_text(arguments.get("url", ""))
        self._validate_public_http_url(url, label="open_page URL")
        endpoint = self.jina_base_url + "/" + url
        max_bytes = max(self.max_page_chars * 4, self.max_page_chars)
        request = {
            "endpoint": endpoint,
            "url": url,
            "return_format": "markdown",
            "max_page_chars": self.max_page_chars,
        }

        def fetch() -> Any:
            headers = {
                "Accept": "text/plain",
                "X-Return-Format": "markdown",
            }
            if self.jina_api_key:
                headers["Authorization"] = "Bearer " + self.jina_api_key
            raw = self._stream_get(
                endpoint,
                headers=headers,
                max_bytes=max_bytes,
                label="Jina page response",
                validate_target=False,
                truncate=True,
            )
            return {"text": raw.decode("utf-8", errors="replace")}

        if self.tool_cache is None:
            payload, cache_hit = fetch(), False
            self._validate_jina_payload(payload)
        else:
            payload, cache_hit = self.tool_cache.get_or_fetch(
                "jina_open_page",
                request,
                fetch,
                self._validate_jina_payload,
            )
        text = answer_to_text(payload.get("text", ""))
        if len(text) > self.max_page_chars:
            text = text[: self.max_page_chars].rstrip() + "\n[truncated]"
        return {
            "text": text or "[Page reader returned no text.]",
            "metadata": {
                "sources": [url],
                "characters": len(text),
                "cache_hit": cache_hit,
            },
        }

    def _image_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        query = self._query(arguments)
        count = self._bounded_count(
            arguments.get("num_results"),
            min(self.max_image_results, MAX_MATCHES),
        )
        endpoint = self.serper_base_url + "/images"
        request = {
            "endpoint": endpoint,
            "q": query,
            "num": count,
        }

        def fetch() -> Any:
            return self._post_serper_json(endpoint, {"q": query, "num": count})

        def validate(payload: Any) -> None:
            self._validate_result_payload(
                payload,
                label="Serper image search",
                result_field="images",
            )

        if self.tool_cache is None:
            payload, cache_hit = fetch(), False
            validate(payload)
        else:
            payload, cache_hit = self.tool_cache.get_or_fetch(
                "serper_image_search",
                request,
                fetch,
                validate,
            )
        return self._render_image_results(
            (payload.get("images") or [])[:count],
            metadata={"query": query, "cache_hit": cache_hit},
            empty_text="No image search results.",
        )

    def _reverse_image_search(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        image_id = answer_to_text(arguments.get("image_id", ""))
        stored = self.images.get(image_id)
        if stored is None:
            raise ValueError("reverse_image_search requires a registered image_id")
        count = self._bounded_count(
            arguments.get("num_results"),
            min(self.max_image_results, MAX_MATCHES),
        )
        base_url = service_base_url(self.serpapi_base_url)
        upload_jpeg = prepare_upload_jpeg(
            stored.source_jpeg_bytes, max_side=self.image_max_side,
        )
        request = {
            "endpoint": base_url + "/search.json",
            "image_sha256": hashlib.sha256(upload_jpeg).hexdigest(),
            **LENS_PARAMETERS,
        }
        cache_key = JsonToolCache._fingerprint("serpapi_reverse_image", request)

        def fetch() -> Any:
            return fetch_visual_matches(
                self.http, base_url=base_url, api_key=self.serpapi_api_key,
                upload_jpeg=upload_jpeg, timeout=self.timeout,
            )

        if cache_key in self.reverse_image_cache:
            payload, cache_hit = self.reverse_image_cache[cache_key], True
        elif self.tool_cache is None:
            payload, cache_hit = fetch(), False
            validate_cached_matches(payload)
        else:
            payload, cache_hit = self.tool_cache.get_or_fetch(
                "serpapi_reverse_image", request, fetch, validate_cached_matches,
            )
        self.reverse_image_cache[cache_key] = payload
        results = payload["matches"][:count]
        return self._render_image_results(
            results,
            metadata={
                "image_id": image_id, "cache_hit": cache_hit,
                "provider": "serpapi", "engine": "google_lens",
                "status": "success" if results else "no_results",
            },
            empty_text="No reverse image search results.",
        )

    def _render_image_results(
        self,
        results: Sequence[Dict[str, Any]],
        *,
        metadata: Dict[str, Any],
        empty_text: str,
    ) -> Dict[str, Any]:
        lines = []
        image_payloads = []
        loaded_image_results = []
        sources = []
        for index, item in enumerate(results, start=1):
            image_id = "search:{}".format(self.next_search_image)
            self.next_search_image += 1
            title = answer_to_text(item.get("title", "Untitled"))
            source_url = answer_to_text(item.get("link", ""))
            image_url = answer_to_text(item.get("imageUrl", ""))
            thumbnail_url = answer_to_text(item.get("thumbnailUrl", ""))
            stored = None
            errors = []
            for candidate in (image_url, thumbnail_url):
                if self.total_images_used >= self.max_total_images:
                    errors.append("record exceeds total image budget")
                    break
                if not candidate or not self._is_http_url(candidate):
                    continue
                try:
                    source_jpeg, preview_jpeg = self._prepare_image(
                        self._download_bytes(candidate)
                    )
                    self._reserve_image()
                    stored = StoredImage(
                        image_id=image_id,
                        source_jpeg_bytes=source_jpeg,
                        preview_jpeg_bytes=preview_jpeg,
                        source=candidate,
                    )
                    break
                except Exception as exc:
                    error = safe_exception(exc)
                    # Use an independent thumbnail for an unavailable origin;
                    # never retry the search API or silence thumbnail outages.
                    status = http_status(exc)
                    alternate_thumbnail = (candidate == image_url and thumbnail_url != image_url
                                           and self._is_http_url(thumbnail_url))
                    # A public image origin may require payment to view its
                    # asset. This is not a search-provider credential failure.
                    unavailable_asset = (status in {401, 402, 403, 404, 410}
                                         or (status in {500, 502, 503, 504} and alternate_thumbnail))
                    if error.infrastructure and not unavailable_asset:
                        raise error from None
                    errors.append(str(error))
                    if "image budget" in str(exc):
                        break
            if stored is not None:
                self.images[image_id] = stored
                image_payloads.append(
                    self._data_url(stored.preview_jpeg_bytes)
                )
                loaded_image_results.append(
                    {
                        "image_id": image_id,
                        "title": title,
                        "source_page": source_url,
                        "image_source": stored.source,
                        "download_errors": errors,
                    }
                )
                status = "loaded"
            else:
                status = "not loaded ({})".format(
                    "; ".join(errors) or "no usable image URL"
                )
            lines.append(
                "[{}] {} — {}\nSource page: {}\nImage URL: {}".format(
                    image_id,
                    title,
                    status,
                    source_url,
                    image_url,
                )
            )
            if self._is_http_url(source_url) and source_url not in sources:
                sources.append(source_url)
        return {
            "text": "\n\n".join(lines) or empty_text,
            "images": image_payloads,
            "metadata": {
                **metadata,
                "sources": sources,
                "result_count": len(results),
                "images_loaded": len(image_payloads),
                "loaded_image_results": loaded_image_results,
                "image_result_policy": "eager_all",
            },
        }

    def _crop_image(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        image_id = answer_to_text(arguments.get("image_id", ""))
        stored = self.images.get(image_id)
        if stored is None:
            raise ValueError(
                "unknown image_id {}; available: {}".format(
                    image_id,
                    ", ".join(sorted(self.images)) or "none",
                )
            )
        coordinates = []
        for key in ("x1", "y1", "x2", "y2"):
            value = arguments.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("{} must be numeric".format(key))
            coordinates.append(float(value))
        x1, y1, x2, y2 = coordinates
        if not (
            0 <= x1 < x2 <= 1000
            and 0 <= y1 < y2 <= 1000
        ):
            raise ValueError(
                "crop coordinates must satisfy 0 <= x1 < x2 <= 1000 "
                "and 0 <= y1 < y2 <= 1000"
            )
        with Image.open(io.BytesIO(stored.source_jpeg_bytes)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            left, top = int(width * x1 / 1000), int(height * y1 / 1000)
            box = (
                left,
                top,
                max(int(width * x2 / 1000), left + 1),
                max(int(height * y2 / 1000), top + 1),
            )
            cropped = image.crop(box)
            crop_size = list(cropped.size)
            source_jpeg = self._encode_jpeg(cropped)
            scale = min(
                self.image_max_side / cropped.width,
                self.image_max_side / cropped.height,
                self.max_crop_upscale,
            )
            if scale != 1:
                output_size = (
                    max(1, round(cropped.width * scale)),
                    max(1, round(cropped.height * scale)),
                )
                cropped = cropped.resize(
                    output_size,
                    Image.Resampling.LANCZOS,
                )
            jpeg = self._encode_jpeg(cropped)
        self._reserve_image()
        output_image_id = "crop:{}".format(self.next_crop_image)
        self.next_crop_image += 1
        self.images[output_image_id] = StoredImage(
            image_id=output_image_id,
            source_jpeg_bytes=source_jpeg,
            preview_jpeg_bytes=jpeg,
            source=stored.source,
        )
        return {
            "text": (
                "[{}] Crop of {} using normalized box [{}, {}, {}, {}].".format(
                    output_image_id,
                    image_id,
                    x1,
                    y1,
                    x2,
                    y2,
                )
            ),
            "images": [self._data_url(jpeg)],
            "metadata": {
                "image_id": image_id,
                "parent_image_id": image_id,
                "output_image_id": output_image_id,
                "crop_size": crop_size,
                "output_size": list(cropped.size),
                "source": stored.source,
                "sources": (
                    [stored.source]
                    if self._is_http_url(stored.source)
                    else []
                ),
            },
        }

    def _download_bytes(self, url: str) -> bytes:
        return self._stream_get(
            url,
            headers={"User-Agent": "VisSearch-ReAct-Evaluator/1.0"},
            max_bytes=self.max_download_bytes,
            label="image",
            validate_target=True,
        )

    def _read_local_bytes(self, path: Path) -> bytes:
        chunks = bytearray()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(
                    min(64 * 1024, self.max_local_image_bytes + 1 - len(chunks))
                )
                if not chunk:
                    break
                chunks.extend(chunk)
                if len(chunks) > self.max_local_image_bytes:
                    raise ValueError("image exceeds maximum local image size")
        # Trusted benchmark inputs are not network downloads. Pixel/image-count
        # limits still apply in _prepare_image and _reserve_image below.
        return bytes(chunks)

    def _stream_get(
        self,
        url: str,
        *,
        headers: Dict[str, str],
        max_bytes: int,
        label: str,
        validate_target: bool,
        truncate: bool = False,
    ) -> bytes:
        current_url = url
        headers = dict(headers)
        for redirect_index in range(self.max_redirects + 1):
            if validate_target or redirect_index > 0:
                self._validate_public_http_url(
                    current_url,
                    label="{} URL".format(label),
                )
            response = self.http.get(
                current_url,
                headers=headers,
                timeout=self.timeout,
                stream=True,
                allow_redirects=False,
            )
            try:
                if response.status_code in {
                    301,
                    302,
                    303,
                    307,
                    308,
                }:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError(
                            "{} redirect is missing Location".format(label)
                        )
                    if redirect_index >= self.max_redirects:
                        raise ValueError(
                            "{} exceeds maximum redirects".format(label)
                        )
                    next_url = urljoin(current_url, location)
                    previous, following = urlparse(current_url), urlparse(next_url)
                    def origin(parsed):
                        return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == 'https' else 80)
                    if origin(previous) != origin(following):
                        headers = {k: v for k, v in headers.items()
                                   if k.lower() not in {'authorization', 'cookie', 'proxy-authorization', 'x-api-key'}}
                    current_url = next_url
                    continue

                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_size = int(content_length)
                    except ValueError as exc:
                        raise ValueError(
                            "{} has invalid Content-Length".format(label)
                        ) from exc
                    if declared_size > max_bytes:
                        if not truncate:
                            raise ValueError(
                                "{} exceeds maximum download size".format(
                                    label
                                )
                            )

                chunks = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    remaining = max_bytes - len(chunks)
                    if len(chunk) > remaining:
                        if truncate:
                            accepted = chunk[:remaining]
                            self._consume_download_bytes(len(accepted))
                            chunks.extend(accepted)
                            return bytes(chunks)
                        raise ValueError(
                            "{} exceeds maximum download size".format(label)
                        )
                    self._consume_download_bytes(len(chunk))
                    chunks.extend(chunk)
                return bytes(chunks)
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        raise ValueError("{} exceeds maximum redirects".format(label))

    def _prepare_image(self, raw: bytes) -> tuple[bytes, bytes]:
        if not raw:
            raise ValueError("image is empty")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "error",
                    Image.DecompressionBombWarning,
                )
                with Image.open(io.BytesIO(raw)) as source:
                    width, height = source.size
                    if width * height > self.max_image_pixels:
                        raise ValueError(
                            "image exceeds maximum pixel count"
                        )
                    image = ImageOps.exif_transpose(source).convert("RGB")
                    source_jpeg = self._encode_jpeg(image)
                    preview = image.copy()
                    if max(preview.size) > self.image_max_side:
                        preview.thumbnail(
                            (self.image_max_side, self.image_max_side),
                            Image.Resampling.LANCZOS,
                        )
                    return source_jpeg, self._encode_jpeg(preview)
        except Exception as exc:
            if isinstance(exc, ValueError) and "pixel count" in str(exc):
                raise
            raise ValueError("unsupported or invalid image") from exc

    def _consume_download_bytes(self, count: int) -> None:
        if self.total_download_bytes + count > self.max_total_download_bytes:
            raise _DownloadBudgetExceeded("record exceeds total download byte budget")
        self.total_download_bytes += count

    def _reserve_image(self) -> None:
        if self.total_images_used >= self.max_total_images:
            raise ValueError("record exceeds total image budget")
        self.total_images_used += 1

    @staticmethod
    def _encode_jpeg(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90, optimize=True)
        return buffer.getvalue()

    @staticmethod
    def _data_url(jpeg: bytes) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(
            "ascii"
        )

    @staticmethod
    def _is_http_url(value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _system_resolve(hostname: str) -> Sequence[str]:
        addresses = []
        for item in socket.getaddrinfo(
            hostname,
            None,
            type=socket.SOCK_STREAM,
        ):
            address = item[4][0]
            if address not in addresses:
                addresses.append(address)
        return addresses

    def _validate_public_http_url(self, value: str, *, label: str) -> None:
        try:
            parsed = urlparse(value)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("{} is not a valid URL".format(label)) from exc
        del port
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(
                "{} must be a public HTTP(S) URL".format(label)
            )
        hostname = parsed.hostname.rstrip(".").casefold()
        if hostname == "localhost" or hostname.endswith(".local"):
            raise ValueError(
                "{} must resolve only to public addresses".format(label)
            )
        _public_addresses(hostname, self.host_resolver, label=label)
