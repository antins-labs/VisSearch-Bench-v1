"""Shared webpage reading uses Serper once, caches success, and bounds content."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vissearch_bench.react import make_toolbox
from vissearch_bench.react.agent import ToolCall
from vissearch_bench.errors import AgentError, safe_exception


class Response:
    def __init__(self, payload, status=200, raw=None):
        self.raw = raw if raw is not None else json.dumps(payload).encode()
        self.status_code = status
        self.headers = {}
        self.closed = False

    def raise_for_status(self):
        assert 200 <= self.status_code < 300

    def iter_content(self, chunk_size):
        yield self.raw

    def close(self):
        self.closed = True


def toolbox(tmp_path, response, **settings):
    calls = []
    def post(url, **kw):
        calls.append((url, kw))
        return response
    with patch('vissearch_bench.react.tools._PinnedHttpSession', return_value=SimpleNamespace(post=post)):
        box = make_toolbox([], {'SERPER_API_KEY': 'test-serper-secret',
            'SF_JINA_API_KEY': 'unused-jina-secret',
            'VISBENCH_TOOLS_HTTP_PROXY': '', 'VISBENCH_TOOLS_HTTPS_PROXY': ''},
            {'tool_cache_dir': tmp_path, **settings})
    box.host_resolver = lambda host: ['93.184.216.34']
    return box, calls


def read(box, url='https://example.com/article'):
    return box.execute(ToolCall('page', 'open_page', {'url': url}))


def test_serper_contract_and_cache_across_framework_toolboxes(tmp_path):
    response = Response({'text': 'Plain body', 'markdown': '# Article\nEvidence',
                         'metadata': {'title': 'Article'}, 'credits': 2})
    box, calls = toolbox(tmp_path, response)
    result = read(box)
    assert result.text == '# Article\nEvidence'
    assert result.metadata['reader_backend'] == 'serper-scrape'
    assert result.metadata['sources'] == ['https://example.com/article']
    assert result.metadata['cache_hit'] is False
    endpoint, kwargs = calls[0]
    assert endpoint == 'https://scrape.serper.dev'
    assert kwargs['json'] == {'url': 'https://example.com/article', 'includeMarkdown': True}
    assert kwargs['headers']['X-API-KEY'] == 'test-serper-secret'
    assert kwargs['stream'] is True and kwargs['allow_redirects'] is False
    assert response.closed and len(calls) == 1
    other, second_calls = toolbox(tmp_path, Response({}))
    assert read(other).metadata['cache_hit'] is True
    assert second_calls == []
    saved = ''.join(p.read_text() for p in tmp_path.rglob('*.json'))
    assert 'test-serper-secret' not in saved and 'unused-jina-secret' not in saved


def test_plain_text_fallback_and_page_character_limit(tmp_path):
    box, calls = toolbox(tmp_path, Response({'text': 'A' * 50}), max_page_chars=10)
    result = read(box)
    assert result.text == 'A' * 10 + '\n[truncated]'
    assert len(calls) == 1


@pytest.mark.parametrize('payload,status,raw', [
    ({}, 402, None), ({}, 429, None), ({}, 503, None),
    ({'error': 'bad account'}, 200, None), ({}, 200, None),
    ({'text': []}, 200, None), ({'text': ''}, 200, None),
    ({}, 200, b'not json'),
])
def test_failures_abort_without_retry_or_cache(tmp_path, payload, status, raw):
    response = Response(payload, status, raw)
    box, calls = toolbox(tmp_path, response)
    with pytest.raises(AgentError) as error:
        read(box)
    assert error.value.infrastructure is True
    assert len(calls) == 1 and response.closed
    assert not list(tmp_path.rglob('*.json'))


def test_response_size_limit_is_enforced_and_not_cached(tmp_path):
    box, calls = toolbox(tmp_path, Response({'text': 'A' * 100}), max_download_bytes=50)
    with pytest.raises(AgentError) as error:
        read(box)
    assert error.value.infrastructure and len(calls) == 1
    assert not list(tmp_path.rglob('*.json'))


def test_private_target_is_rejected_before_serper_call(tmp_path):
    box, calls = toolbox(tmp_path, Response({'text': 'body'}))
    with pytest.raises(ValueError):
        read(box, 'http://127.0.0.1/private')
    assert calls == []


def test_record_download_budget_is_not_a_service_outage(tmp_path):
    response = Response({'text': 'a public body'})
    box, calls = toolbox(tmp_path, response, max_total_download_bytes=5)
    with pytest.raises(ValueError) as error:
        read(box)
    assert safe_exception(error.value).infrastructure is False
    assert response.closed and len(calls) == 1
    assert not list(tmp_path.rglob('*.json'))


def test_explicit_target_scraping_failure_is_unavailable_observation_not_cached(tmp_path):
    box, calls = toolbox(tmp_path, Response({'message':'Scraping failed.','statusCode':500},500))
    result = read(box)
    assert 'unavailable' in result.text.lower()
    assert result.metadata['error_code'] == 'SerperScrapeTargetUnavailable'
    assert result.metadata['http_status'] == 500
    assert result.metadata['sources'] == []
    assert len(calls) == 1 and not list(tmp_path.rglob('*.json'))


@pytest.mark.parametrize('payload,status', [
    ({'message':'Internal server error','statusCode':500},500),
    ({'message':'Scraping failed.','statusCode':503},503),
    ({'message':'Scraping failed.','statusCode':429},429),
])
def test_other_service_errors_remain_infrastructure_failures(tmp_path,payload,status):
    box,calls=toolbox(tmp_path,Response(payload,status))
    with pytest.raises(AgentError) as error:read(box)
    assert error.value.infrastructure and len(calls)==1
