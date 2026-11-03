from types import SimpleNamespace
import pytest

from vissearch_bench.errors import AgentError
from vissearch_bench.react import ObservedToolbox
from vissearch_bench.react.agent import ToolCall


@pytest.mark.parametrize('label', [
    'SerperReadTimeout', 'SerperConnectionError', 'SerperHTTP503', 'SerperHTTP429',
    'SerperHTTP404', 'SerperHTTP400', 'SerperHTTP410', 'SerperHTTP422',
    'SerperScrapeEmptyContent', 'SerperJSONDecodeError', 'SerperScrapeMalformedResponse',
    'SerperScrapeServiceError', 'SerperResponseTooLarge',
    'Serper web search response is missing organic', 'Serper image search images must be a list',
    'SerpApiSearchHTTP404', 'SerpApiSearchMalformedResponse', 'SerpApiUploadServiceError',
])
def test_transient_tool_failure_is_recorded_as_observation_without_retry(label):
    calls, events = [], []
    def fail(call):
        calls.append(call)
        raise AgentError(label, infrastructure=True)
    box = ObservedToolbox(SimpleNamespace(execute=fail), events.append)
    result = box.execute(ToolCall('one', 'open_page', {'url': 'https://example.org'}))
    assert len(calls) == 1
    assert result.metadata['error'] and result.metadata['recoverable']
    assert 'No evidence' in result.text
    assert [e['event'] for e in events] == ['tool_request', 'tool_error', 'tool_response']
    assert events[1]['infrastructure'] is True


@pytest.mark.parametrize('label', ['SerperHTTP401', 'SerperHTTP402', 'SerperHTTP403',
                                  'SerpApiSearchHTTP402', 'MissingCredentials',
                                  'SerperAttributeError', 'SerpApiInvalidEndpoint', 'unknown'])
def test_auth_and_unknown_failure_are_not_silently_retried(label):
    def fail(call): raise AgentError(label, infrastructure=True)
    box = ObservedToolbox(SimpleNamespace(execute=fail), lambda e: None)
    with pytest.raises(AgentError): box.execute(ToolCall('one', 'web_search', {'query': 'q'}))


def test_direct_image_download_timeout_remains_identifiable_and_secret_free():
    from urllib3.exceptions import ReadTimeoutError
    from vissearch_bench.react.tools import _PinnedHttpSession
    from vissearch_bench.errors import recoverable_tool_error
    calls = []
    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise ReadTimeoutError(None, '/private?secret=test', 'upstream timeout')
    pool = SimpleNamespace(urlopen=fail, close=lambda: None)
    session = _PinnedHttpSession(host_resolver=lambda host: ['8.8.8.8'], pool_factory=lambda **kw: pool)
    with pytest.raises(AgentError) as caught:
        session.get('https://example.org/picture', headers={}, timeout=1, stream=True, allow_redirects=False)
    assert recoverable_tool_error(caught.value)
    assert str(caught.value) == 'AssetRequestFailure'
    assert len(calls) == 1 and calls[0]['retries'] is False


def test_real_serper_404_wrapper_returns_failed_observation_and_closes_once():
    from vissearch_bench.react.tools import ReActToolbox
    posts, closed, events = [], [], []
    def post(*args, **kwargs):
        posts.append((args, kwargs))
        return SimpleNamespace(status_code=404, close=lambda: closed.append(True))
    transport = SimpleNamespace(http=SimpleNamespace(post=post), serper_api_key='offline-test',
                                timeout=1, _require_serper_key=lambda: None)
    def execute(call):
        return ReActToolbox._post_serper_json(transport, 'https://scrape.serper.dev',
                                            {'url': call.arguments['url']}, page_target=True)
    observation = ObservedToolbox(SimpleNamespace(execute=execute), events.append).execute(
        ToolCall('a', 'open_page', {'url': 'https://example.org/missing'}))
    assert len(posts) == len(closed) == 1
    assert observation.metadata['error_code'] == 'SerperHTTP404'
    assert observation.metadata['recoverable'] and not observation.metadata['sources']
    assert events[1]['error'] == 'Serper HTTP 404'
