"""Direct text/image API calls with lazy SDKs and no tool or search capabilities.

``client_factory``, when supplied, accepts the SDK constructor's keyword
arguments. It supports offline tests without installing either SDK. ``retries``
is the total number of attempts, including the first call.
"""

from __future__ import annotations

import base64
import importlib
import re
import threading
import time
from collections.abc import Mapping
from typing import Any, Callable


class CompletionError(RuntimeError):
    """A safe-to-record failure containing no original exception or request."""

    def __init__(self, message, attempt_history=None):
        super().__init__(message)
        self.attempt_history = attempt_history or []


class EmptyResponse(ValueError):
    """The provider returned no visible answer text."""


class UnexpectedToolCall(ValueError):
    """A Direct response unexpectedly requested a tool."""


class InvalidMessage(ValueError):
    """A message cannot be represented by the selected native API."""


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _usage_dict(usage: Any) -> dict:
    if usage is None:
        return {}
    if isinstance(usage, Mapping):
        return dict(usage)
    if callable(getattr(usage, "model_dump", None)):
        return usage.model_dump(mode="json")
    return dict(vars(usage))


class APIClient:
    """Thread-safe wrapper; each thread owns and reuses its own SDK client."""

    def __init__(self, profile: dict, environ: Mapping[str, str],
                 client_factory: Callable[..., Any] | None = None):
        self.profile = dict(profile)
        self.environ = environ
        self.provider = profile.get("provider")
        self.model = profile.get("model")
        self.attempt_limit = profile.get("retries", 1)
        if self.provider not in {"openai", "anthropic"}:
            raise ValueError("Unsupported API provider")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("API model must be a nonempty string")
        if (isinstance(self.attempt_limit, bool)
                or not isinstance(self.attempt_limit, int) or self.attempt_limit < 1):
            raise ValueError("API retries must be a positive total attempt count")
        self._client_factory = client_factory
        self._local = threading.local()

    def _credential(self) -> str | None:
        names = [self.profile.get("api_key_env")]
        names.extend(self.profile.get("api_key_fallback_envs") or [])
        for name in names:
            value = self.environ.get(name) if isinstance(name, str) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def check_credentials(self) -> None:
        """Validate credential availability without importing SDKs or calling APIs."""
        if self._credential() is None:
            raise CompletionError(self._error_message("MissingCredentials", None, 0))

    def _error_message(self, error_class: str, status: int | None, attempts: int) -> str:
        # Only metadata is retained; exception messages may contain keys or prompts.
        model = re.sub(r"[^A-Za-z0-9_.:/-]", "_", self.model)[:160]
        name = re.sub(r"[^A-Za-z0-9_]", "_", error_class)[:80]
        return (f"provider={self.provider} model={model} error={name} "
                f"status={status if status is not None else 'unknown'} attempts={attempts}")

    def _client(self) -> Any:
        if not hasattr(self._local, "client"):
            factory = self._client_factory
            if factory is None:
                sdk = importlib.import_module(self.provider)
                from .network import isolated_model_client
                constructor = sdk.OpenAI if self.provider == "openai" else sdk.Anthropic
                factory = lambda **options: isolated_model_client(constructor, **options)
            kwargs = {"api_key": self._credential(),
                      "timeout": self.profile.get("timeout", 120), "max_retries": 0}
            if self.profile.get("base_url"):
                kwargs["base_url"] = self.profile["base_url"]
            self._local.client = factory(**kwargs)
        return self._local.client

    def _request(self, messages: list[dict]) -> dict:
        args = {"model": self.model}
        if self.profile.get("temperature") is not None:
            args["temperature"] = self.profile["temperature"]
        if self.provider == "openai":
            args["messages"] = messages
            token_key = ("max_completion_tokens" if self.model.lower().startswith("gpt-5.5")
                         else "max_tokens")
            args[token_key] = self.profile["max_tokens"]
            if self.profile.get("reasoning_effort") is not None:
                args["reasoning_effort"] = self.profile["reasoning_effort"]
            qwen_model = self.model.rsplit("/", 1)[-1].lower().startswith("qwen")
            if self.profile.get("disable_thinking") is True and qwen_model:
                args["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        else:
            args["max_tokens"] = self.profile["max_tokens"]
            native_messages, system = self._anthropic_messages(messages)
            args["messages"] = native_messages
            if system:
                args["system"] = system
        return args

    @staticmethod
    def _anthropic_messages(messages: list[dict]) -> tuple[list[dict], str]:
        native = []
        system_parts = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            if role == "system":
                if not isinstance(content, str):
                    raise InvalidMessage()
                system_parts.append(content)
                continue
            if role not in {"user", "assistant"}:
                raise InvalidMessage()
            if isinstance(content, str):
                native.append({"role": role, "content": content})
                continue
            if not isinstance(content, list):
                raise InvalidMessage()
            blocks = []
            for block in content:
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    blocks.append({"type": "text", "text": block["text"]})
                elif block.get("type") == "image_url":
                    image_url = block["image_url"]
                    url = image_url.get("url") if isinstance(image_url, Mapping) else image_url
                    match = re.fullmatch(r"data:(image/(?:png|jpeg|gif|webp));base64,([A-Za-z0-9+/=]+)",
                                         url) if isinstance(url, str) else None
                    if match is None:
                        raise InvalidMessage()
                    media_type, data = match.groups()
                    if not base64.b64decode(data, validate=True):
                        raise InvalidMessage()
                    blocks.append({"type": "image", "source": {
                        "type": "base64", "media_type": media_type, "data": data,
                    }})
                else:
                    raise InvalidMessage()
            native.append({"role": role, "content": blocks})
        return native, "\n\n".join(system_parts)

    def _response(self, response: Any) -> tuple[str, dict, str]:
        if self.provider == "openai":
            choices = _field(response, "choices", [])
            if not choices:
                raise EmptyResponse()
            choice = choices[0]
            message = _field(choice, "message")
            finish_reason = _field(choice, "finish_reason") or "unknown"
            if (_field(message, "tool_calls") or _field(message, "function_call")
                    or finish_reason in {"tool_calls", "function_call"}):
                raise UnexpectedToolCall()
            text = _field(message, "content")
        else:
            blocks = _field(response, "content", []) or []
            finish_reason = _field(response, "stop_reason") or "unknown"
            if (finish_reason == "tool_use" or any(
                    _field(block, "type") in {"tool_use", "server_tool_use"} for block in blocks)):
                raise UnexpectedToolCall()
            text = "".join(_field(block, "text", "") for block in blocks
                           if _field(block, "type") == "text")
        if not isinstance(text, str) or not text.strip():
            raise EmptyResponse()
        return text, _usage_dict(_field(response, "usage")), finish_reason

    def complete(self, messages: list[dict]) -> dict:
        """Return visible free-form text and metadata, retaining truncation reasons.

        SDK errors are reduced to class/status metadata outside the exception
        handler, so no raw exception is chained into a stored or printed error.
        """
        self.check_credentials()
        started = time.perf_counter()
        preparation_error = None
        try:
            request = self._request(messages)
        except Exception as exc:
            preparation_error = self._error_message(type(exc).__name__, None, 0)
        if preparation_error is not None:
            raise CompletionError(preparation_error)

        attempt_history = []
        for attempt in range(1, self.attempt_limit + 1):
            attempt_started = time.perf_counter()
            try:
                client = self._client()
                if self.provider == "openai":
                    response = client.chat.completions.create(**request)
                else:
                    response = client.messages.create(**request)
                text, usage, finish_reason = self._response(response)
                metadata = {name: value for name, field in (('response_id', 'id'), ('response_model', 'model'))
                            if isinstance(value := _field(response, field), str)}
                if self.provider == 'openai':
                    reasoning = _field(_field(response, 'choices')[0], 'message')
                    reasoning = _field(reasoning, 'reasoning_content')
                else:
                    reasoning = '\n'.join(_field(block, 'thinking', '') for block in _field(response, 'content', [])
                                          if _field(block, 'type') == 'thinking')
                if isinstance(reasoning, str) and reasoning:
                    metadata['reasoning_text'] = reasoning
                attempt_history.append({'attempt': attempt, 'status': 'ok',
                                        'latency_seconds': time.perf_counter() - attempt_started,
                                        'finish_reason': finish_reason,
                                        **{k: v for k, v in metadata.items() if k != 'reasoning_text'}})
                return {"text": text, "usage": usage,
                        "latency_seconds": float(time.perf_counter() - started),
                        "finish_reason": finish_reason, "attempts": attempt,
                        'api_attempts': attempt_history, **metadata}
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                status = status if type(status) is int and 100 <= status <= 599 else None
                failure = self._error_message(type(exc).__name__, status, attempt)
                attempt_history.append({'attempt': attempt, 'status': 'error', 'error_class': type(exc).__name__,
                                        'http_status': status, 'latency_seconds': time.perf_counter() - attempt_started})
            if attempt < self.attempt_limit:
                time.sleep(min(2 ** (attempt - 1), 8))
        raise CompletionError(failure, attempt_history)
