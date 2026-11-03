"""Offline provider contract tests; SDK-shaped fakes never make network calls."""

import importlib
import sys
import threading
import traceback
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch



try:
    providers = importlib.import_module("vissearch_bench.providers")
except ModuleNotFoundError as exc:
    if exc.name != "vissearch_bench.providers":
        raise
    providers = None


def profile(**overrides):
    result = {
        "provider": "openai",
        "model": "gpt-5.5",
        "base_url": "https://example.invalid/v1",
        "api_key_env": "TEST_PROVIDER_KEY",
        "max_tokens": 2048,
        "temperature": None,
        "timeout": 90,
        "retries": 1,
    }
    result.update(overrides)
    return result


def openai_response(text="The answer is 42.", finish_reason="stop", **message):
    return SimpleNamespace(
        id="chatcmpl-offline",
        object="chat.completion",
        created=1,
        model="offline-model",
        choices=[SimpleNamespace(
            index=0,
            message=SimpleNamespace(role="assistant", content=text,
                                    tool_calls=None, **message),
            finish_reason=finish_reason,
            logprobs=None,
        )],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=7, total_tokens=19),
    )


def anthropic_response(text="A plain text answer.", stop_reason="end_turn", blocks=None):
    return SimpleNamespace(
        id="msg_offline", type="message", role="assistant", model="offline-model",
        content=blocks if blocks is not None else [SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason, stop_sequence=None,
        usage=SimpleNamespace(input_tokens=12, output_tokens=7),
    )


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.messages = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeFactory:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []
        self.clients = []
        self.lock = threading.Lock()

    def __call__(self, **kwargs):
        client = FakeClient(self.outcomes)
        with self.lock:
            self.calls.append(kwargs)
            self.clients.append(client)
        return client


class FakeAPIError(Exception):
    status_code = 429


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(providers, "the provider implementation does not exist yet")
        self.environ = {"TEST_PROVIDER_KEY": "offline-secret-not-a-real-key"}
        self.messages = [{"role": "user", "content": "Explain the answer."}]

    def make_client(self, outcomes, **overrides):
        factory = FakeFactory(outcomes)
        client = providers.APIClient(profile(**overrides), self.environ, factory)
        return client, factory

    def test_openai_uses_only_direct_chat_parameters(self):
        client, factory = self.make_client([openai_response()])
        result = client.complete(self.messages)
        self.assertEqual(factory.calls, [{
            "api_key": self.environ["TEST_PROVIDER_KEY"],
            "base_url": "https://example.invalid/v1", "timeout": 90, "max_retries": 0,
        }])
        self.assertEqual(factory.clients[0].requests, [{
            "model": "gpt-5.5", "messages": self.messages, "max_completion_tokens": 2048,
        }])
        self.assertEqual(result["text"], "The answer is 42.")
        self.assertEqual(result["usage"], {"prompt_tokens": 12, "completion_tokens": 7,
                                          "total_tokens": 19})
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["attempts"], 1)
        self.assertIsInstance(result["latency_seconds"], float)
        self.assertGreaterEqual(result["latency_seconds"], 0)

    def test_gpt55_version_uses_completion_tokens(self):
        client, factory = self.make_client([openai_response()], model="gpt-5.5-2026-10-01")
        client.complete(self.messages)
        self.assertIn("max_completion_tokens", factory.clients[0].requests[0])
        self.assertNotIn("max_tokens", factory.clients[0].requests[0])

    def test_actual_response_metadata_and_reasoning_are_preserved_separately(self):
        client, _ = self.make_client([openai_response(text="Final answer", reasoning_content="Returned reasoning")])
        result = client.complete(self.messages)
        self.assertEqual(result['text'], 'Final answer')
        self.assertEqual(result['reasoning_text'], 'Returned reasoning')
        self.assertEqual(result['response_id'], 'chatcmpl-offline')
        self.assertEqual(result['response_model'], 'offline-model')
        self.assertEqual(result['api_attempts'][0]['status'], 'ok')

    def test_retry_trace_records_safe_metadata_for_each_attempt(self):
        client, _ = self.make_client([FakeAPIError('offline-secret-not-a-real-key'), openai_response()], retries=2)
        with patch('vissearch_bench.providers.time.sleep'):
            result = client.complete(self.messages)
        self.assertEqual([x['status'] for x in result['api_attempts']], ['error', 'ok'])
        self.assertEqual(result['api_attempts'][0]['http_status'], 429)
        self.assertNotIn('offline-secret-not-a-real-key', str(result))

    def test_other_models_receive_max_tokens_and_only_configured_options(self):
        client, factory = self.make_client(
            [openai_response()], model="qwen3.5-397b-a17b", temperature=0.2,
            reasoning_effort="high", disable_thinking=True)
        client.complete(self.messages)
        self.assertEqual(factory.clients[0].requests[0], {
            "model": "qwen3.5-397b-a17b", "messages": self.messages,
            "max_tokens": 2048, "temperature": 0.2, "reasoning_effort": "high",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        })

    def test_false_thinking_and_null_reasoning_add_no_request_parameters(self):
        client, factory = self.make_client([openai_response()], disable_thinking=False,
                                          reasoning_effort=None)
        client.complete(self.messages)
        self.assertEqual(set(factory.clients[0].requests[0]),
                         {"model", "messages", "max_completion_tokens"})

    def test_disable_thinking_is_not_sent_to_gpt_or_gemini(self):
        for model in ("gpt-5.5", "gemini-3.1-pro-preview"):
            with self.subTest(model=model):
                client, factory = self.make_client([openai_response()], model=model,
                                                  disable_thinking=True)
                client.complete(self.messages)
                self.assertNotIn("extra_body", factory.clients[0].requests[0])

    def test_disable_thinking_is_supported_for_namespaced_qwen_models(self):
        client, factory = self.make_client([openai_response()],
                                          model="Qwen/Qwen3.5-397B-A17B",
                                          disable_thinking=True)
        client.complete(self.messages)
        self.assertEqual(factory.clients[0].requests[0]["extra_body"],
                         {"chat_template_kwargs": {"enable_thinking": False}})

    def test_openai_images_are_preserved(self):
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "Read the figure."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
        ]}]
        client, factory = self.make_client([openai_response()])
        client.complete(messages)
        self.assertEqual(factory.clients[0].requests[0]["messages"], messages)

    def test_credentials_checked_without_importing_sdk_or_constructing_client(self):
        factory = FakeFactory([])
        client = providers.APIClient(profile(), self.environ, factory)
        with patch("importlib.import_module", side_effect=AssertionError("SDK import forbidden")):
            self.assertIsNone(client.check_credentials())
        self.assertEqual(factory.calls, [])

    def test_missing_credentials_fail_before_factory_call_and_do_not_leak_environment(self):
        factory = FakeFactory([])
        client = providers.APIClient(profile(), {"UNRELATED": "secret-unrelated"}, factory)
        with self.assertRaises(providers.CompletionError) as caught:
            client.complete(self.messages)
        self.assertEqual(factory.calls, [])
        self.assertNotIn("secret-unrelated", str(caught.exception))
        self.assertIn("Credentials", str(caught.exception))

    def test_env_fallback_uses_first_nonempty_value(self):
        factory = FakeFactory([openai_response()])
        client = providers.APIClient(
            profile(api_key_fallback_envs=["SECOND_KEY", "THIRD_KEY"]),
            {"TEST_PROVIDER_KEY": " ", "SECOND_KEY": "", "THIRD_KEY": "fallback-secret"},
            factory)
        client.complete(self.messages)
        self.assertEqual(factory.calls[0]["api_key"], "fallback-secret")

    def test_primary_credential_wins_over_fallback(self):
        factory = FakeFactory([openai_response()])
        client = providers.APIClient(profile(api_key_fallback_envs=["SECOND_KEY"]),
                                     {**self.environ, "SECOND_KEY": "fallback-secret"}, factory)
        client.complete(self.messages)
        self.assertEqual(factory.calls[0]["api_key"], self.environ["TEST_PROVIDER_KEY"])

    def test_transient_failures_retry_with_total_attempt_limit(self):
        client, factory = self.make_client(
            [FakeAPIError("request had offline-secret-not-a-real-key"),
             FakeAPIError("request had private prompt"), openai_response()], retries=3)
        with patch("time.sleep"):
            result = client.complete(self.messages)
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(len(factory.clients[0].requests), 3)
        self.assertEqual(len(factory.calls), 1)

    def test_retry_exhaustion_is_sanitized_including_traceback(self):
        client, factory = self.make_client(
            [FakeAPIError("offline-secret-not-a-real-key PRIVATE_PROMPT") for _ in range(3)],
            retries=3)
        with patch("time.sleep"):
            try:
                client.complete(self.messages)
            except providers.CompletionError as exc:
                rendered = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                self.assertIsNone(exc.__cause__)
                self.assertIsNone(exc.__context__)
                self.assertIn("openai", str(exc))
                self.assertIn("gpt-5.5", str(exc))
                self.assertIn("FakeAPIError", str(exc))
                self.assertIn("429", str(exc))
                self.assertIn("3", str(exc))
            else:
                self.fail("exhausted retries must raise CompletionError")
        self.assertNotIn("offline-secret-not-a-real-key", rendered)
        self.assertNotIn("PRIVATE_PROMPT", rendered)
        self.assertEqual(len(factory.clients[0].requests), 3)

    def test_constructor_failures_are_sanitized(self):
        def failing_factory(**kwargs):
            raise FakeAPIError("offline-secret-not-a-real-key private constructor data")
        client = providers.APIClient(profile(), self.environ, failing_factory)
        with self.assertRaises(providers.CompletionError) as caught:
            client.complete(self.messages)
        self.assertNotIn("offline-secret", str(caught.exception))
        self.assertIsNone(caught.exception.__context__)

    def test_openai_reasoning_only_is_not_an_answer(self):
        client, _ = self.make_client([openai_response(None, reasoning_content="hidden answer")])
        with self.assertRaises(providers.CompletionError) as caught:
            client.complete(self.messages)
        self.assertIn("EmptyResponse", str(caught.exception))
        self.assertNotIn("hidden answer", str(caught.exception))

    def test_openai_rejects_unexpected_tool_calls(self):
        response = openai_response()
        response.choices[0].message.tool_calls = [{"type": "function", "function": {"name": "web"}}]
        client, _ = self.make_client([response])
        with self.assertRaises(providers.CompletionError) as caught:
            client.complete(self.messages)
        self.assertIn("UnexpectedToolCall", str(caught.exception))

    def test_openai_length_response_is_returned_for_runner_to_mark_truncated(self):
        client, _ = self.make_client([openai_response("Partial answer", finish_reason="length")])
        result = client.complete(self.messages)
        self.assertEqual(result["text"], "Partial answer")
        self.assertEqual(result["finish_reason"], "length")

    def test_whitespace_response_is_rejected(self):
        client, _ = self.make_client([openai_response(" \n\t ")])
        with self.assertRaises(providers.CompletionError):
            client.complete(self.messages)

    def test_full_free_form_text_is_preserved(self):
        text = "  First, calculate.\n\nThe answer is 42.\n"
        client, _ = self.make_client([openai_response(text)])
        self.assertEqual(client.complete(self.messages)["text"], text)

    def test_clients_are_lazy_reused_per_thread_and_not_shared_between_threads(self):
        client, factory = self.make_client([openai_response(), openai_response()])
        self.assertEqual(factory.calls, [])
        client.complete(self.messages)
        client.complete(self.messages)
        self.assertEqual(len(factory.calls), 1)
        barrier = threading.Barrier(2)

        def complete_twice():
            barrier.wait(timeout=5)
            client.complete(self.messages)
            client.complete(self.messages)

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda _: complete_twice(), range(2)))
        self.assertEqual(len(factory.calls), 3)
        self.assertEqual([len(item.requests) for item in factory.clients], [2, 2, 2])

    def test_anthropic_uses_native_messages_and_base64_images(self):
        messages = [
            {"role": "system", "content": "Be precise."},
            {"role": "system", "content": "Explain reasoning."},
            {"role": "user", "content": [
                {"type": "text", "text": "Read the figure."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,d29ybGQ="}},
            ]},
        ]
        client, factory = self.make_client([anthropic_response()], provider="anthropic",
                                          model="claude-opus-4-8", reasoning_effort="high",
                                          disable_thinking=True)
        result = client.complete(messages)
        self.assertEqual(factory.clients[0].requests[0], {
            "model": "claude-opus-4-8", "max_tokens": 2048,
            "system": "Be precise.\n\nExplain reasoning.",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Read the figure."},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "d29ybGQ="}},
            ]}],
        })
        self.assertEqual(result["text"], "A plain text answer.")
        self.assertEqual(result["finish_reason"], "end_turn")
        self.assertEqual(result["usage"], {"input_tokens": 12, "output_tokens": 7})

    def test_anthropic_no_system_and_configured_temperature(self):
        client, factory = self.make_client([anthropic_response()], provider="anthropic", temperature=0)
        client.complete(self.messages)
        self.assertEqual(factory.clients[0].requests[0], {
            "model": "gpt-5.5", "max_tokens": 2048, "temperature": 0, "messages": self.messages,
        })

    def test_anthropic_rejects_tool_blocks(self):
        response = anthropic_response(blocks=[SimpleNamespace(type="tool_use", id="tool1", name="web", input={})])
        client, _ = self.make_client([response], provider="anthropic")
        with self.assertRaises(providers.CompletionError) as caught:
            client.complete(self.messages)
        self.assertIn("UnexpectedToolCall", str(caught.exception))

    def test_anthropic_ignores_thinking_blocks_when_reading_answer(self):
        response = anthropic_response(blocks=[
            SimpleNamespace(type="thinking", thinking="private reasoning", signature="abc"),
            SimpleNamespace(type="text", text="The answer "),
            SimpleNamespace(type="text", text="is 42."),
        ])
        client, _ = self.make_client([response], provider="anthropic")
        self.assertEqual(client.complete(self.messages)["text"], "The answer is 42.")

    def test_anthropic_reasoning_only_is_rejected(self):
        client, _ = self.make_client([anthropic_response(blocks=[
            SimpleNamespace(type="thinking", thinking="private reasoning", signature="abc"),
        ])], provider="anthropic")
        with self.assertRaises(providers.CompletionError) as caught:
            client.complete(self.messages)
        self.assertIn("EmptyResponse", str(caught.exception))

    def test_anthropic_max_tokens_response_preserves_truncation(self):
        client, _ = self.make_client([anthropic_response("Partial", stop_reason="max_tokens")],
                                     provider="anthropic")
        self.assertEqual(client.complete(self.messages)["finish_reason"], "max_tokens")

    def test_anthropic_invalid_image_url_is_rejected_before_remote_request(self):
        client, factory = self.make_client([anthropic_response()], provider="anthropic")
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "https://example.invalid/secret.png"}},
        ]}]
        with self.assertRaises(providers.CompletionError) as caught:
            client.complete(messages)
        self.assertNotIn("secret.png", str(caught.exception))
        self.assertFalse(factory.clients and factory.clients[0].requests)

    def test_invalid_provider_and_attempt_count_fail_locally(self):
        for overrides in ({"provider": "unknown"}, {"retries": 0}, {"retries": -1}):
            with self.subTest(overrides=overrides):
                factory = FakeFactory([])
                with self.assertRaises((ValueError, providers.CompletionError)):
                    client = providers.APIClient(profile(**overrides), self.environ, factory)
                    client.complete(self.messages)
                self.assertEqual(factory.calls, [])


if __name__ == "__main__":
    unittest.main()
