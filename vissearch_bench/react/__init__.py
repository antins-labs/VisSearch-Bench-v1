"""Current-data bridge around the copied native ReAct implementation."""
from dataclasses import asdict
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

from ..errors import AgentError, safe_exception, scrub, recoverable_tool_error, failed_tool_observation
from ..tool_config import TOOL_CACHE, TOOL_LIMITS, tool_environment, tool_proxies
from ..network import isolated_model_client
from .agent import (AnthropicReActClient, OpenAIReActClient, run_react_agent,
                    _trace_tool_call, _trace_observation)
from .tools import ReActToolbox
from .compat import answer_to_text


class CheckedSession:
    def __init__(self, session, emit):
        self.session, self.emit = session, emit

    def __getattr__(self, name): return getattr(self.session, name)

    def complete(self, **kwargs):
        self.emit({'event': 'model_request', 'model': kwargs.get('model'),
                   'forced_final': bool(kwargs.get('force_tool_name'))})
        try:
            turn = self.session.complete(**kwargs)
        except Exception as exc:
            error = safe_exception(exc)
            self.emit({'event': 'model_error', 'error': str(error), 'infrastructure': error.infrastructure})
            raise error from None
        self.emit({'event': 'model_response', 'text': turn.text, 'stop_reason': turn.stop_reason,
                   'usage': turn.usage, 'tool_calls': [_trace_tool_call(c) for c in turn.tool_calls]})
        if turn.stop_reason in {'length', 'max_tokens', 'incomplete'}:
            raise AgentError('TruncatedResponse')
        if not turn.text.strip() and not turn.tool_calls:
            raise AgentError('EmptyResponse')
        return turn


class ObservedToolbox:
    def __init__(self, toolbox, emit): self.toolbox, self.emit = toolbox, emit
    def __getattr__(self, name): return getattr(self.toolbox, name)

    def execute(self, call):
        self.emit({'event': 'tool_request', **_trace_tool_call(call)})
        try:
            observation = self.toolbox.execute(call)
        except Exception as exc:
            error = safe_exception(exc)
            recoverable = recoverable_tool_error(exc)
            self.emit({'event': 'tool_error', 'tool': call.name, 'error': str(error),
                       'infrastructure': error.infrastructure, 'recoverable': recoverable})
            if not recoverable:
                raise error from None
            observation = failed_tool_observation(call, exc)
        self.emit({'event': 'tool_response', **_trace_observation(observation)})
        return observation


def credential(profile, environ):
    for name in [profile['api_key_env'], *profile.get('api_key_fallback_envs', [])]:
        if environ.get(name, '').strip(): return environ[name].strip()
    raise AgentError('MissingCredentials', infrastructure=True)


def make_toolbox(image_paths, environ, settings):
    environ = tool_environment(environ)
    paths = [Path(p).resolve() for p in image_paths]
    base = Path(os.path.commonpath([str(p.parent) for p in paths])) if paths else Path.cwd()
    toolbox = ReActToolbox(
        image_base_dir=base, image_references=[str(p.relative_to(base)) for p in paths],
        serper_api_key=environ.get('SERPER_API_KEY', ''),
        serper_base_url=environ.get('SERPER_BASE_URL', 'https://google.serper.dev'),
        jina_api_key='', jina_base_url='https://r.jina.ai',
        serper_scrape_base_url=environ.get('SERPER_SCRAPE_BASE_URL', 'https://scrape.serper.dev'),
        serpapi_api_key=environ.get('SERPAPI_KEY', ''),
        serpapi_base_url=environ.get('SERPAPI_BASE_URL', 'https://serpapi.com'),
        image_max_side=settings.get('image_max_side', 1536),
        **{key: settings.get(key, default) for key, default in TOOL_LIMITS.items()},
        tool_cache_dir=settings.get('tool_cache_dir', TOOL_CACHE),
    )
    proxies = tool_proxies(environ)
    if proxies:
        from ..network import ProxyToolSession
        toolbox.http = ProxyToolSession(proxies, toolbox._validate_public_http_url, fallback=toolbox.http)
    return toolbox


def _sdk_factory(profile):
    """Keep the original native wire format; add the existing Qwen profile flag."""
    sdk_module = importlib.import_module(profile['provider'])
    constructor = sdk_module.OpenAI if profile['provider'] == 'openai' else sdk_module.Anthropic
    def factory(**kwargs):
        client = isolated_model_client(constructor, **kwargs)
        if profile['provider'] == 'openai' and profile.get('disable_thinking') is True and profile['model'].rsplit('/', 1)[-1].lower().startswith('qwen'):
            original = client.chat.completions.create
            def create(**request):
                request['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
                return original(**request)
            client.chat.completions.create = create
        return client
    return factory


def generate(*, question, image_paths, profile, environ, settings, event_sink):
    if type(profile.get('retries')) is not int or profile['retries'] != 1:
        raise AgentError('ReAct adapter requires exactly one API attempt per model call')
    events = []
    def emit(event):
        clean = scrub(event, environ)
        events.append(clean)
        event_sink(clean)
    args = dict(api_key=credential(profile, environ), base_url=profile['base_url'],
                timeout=profile['timeout'], retries=profile['retries'], sdk_factory=_sdk_factory(profile))
    if profile['provider'] == 'anthropic':
        client = AnthropicReActClient(**args)
    else:
        client = OpenAIReActClient(**args, reasoning_effort=profile.get('reasoning_effort'),
                                 omit_temperature=profile.get('temperature') is None)
    checked = SimpleNamespace(new_session=lambda **kw: CheckedSession(client.new_session(**kw), emit))
    toolbox = ObservedToolbox(make_toolbox(image_paths, environ, settings), emit)
    result = run_react_agent(
        client=checked, toolbox=toolbox, question=question,
        initial_image_data_urls=toolbox.initial_image_data_urls,
        model=profile['model'], max_tokens=profile['max_tokens'], temperature=profile.get('temperature'),
        max_turns=settings.get('max_turns', 30), max_total_tool_calls=settings.get('max_total_tool_calls', 60),
        max_tool_calls_per_turn=settings.get('max_tool_calls_per_turn', 4))
    payload = asdict(result)
    return scrub({'text': answer_to_text(result.prediction['final_answer']), 'usage': result.usage,
                  'finish_reason': result.termination_reason, 'framework': 'react',
                  'trace': events, 'agent_result': payload}, environ)
