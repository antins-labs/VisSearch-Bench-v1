from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vissearch_bench import react
from vissearch_bench.tool_config import tool_environment


def test_operator_proxy_is_explicitly_carried_to_isolated_worker():
    env = tool_environment({'https_proxy':'http://127.0.0.1:7890',
                            'http_proxy':'http://127.0.0.1:7890'})
    assert env['VISBENCH_TOOLS_HTTPS_PROXY'] == 'http://127.0.0.1:7890'
    assert env['VISBENCH_TOOLS_HTTP_PROXY'] == 'http://127.0.0.1:7890'
    from vissearch_bench.tool_config import TOOL_ENV_KEYS
    assert {'VISBENCH_TOOLS_HTTPS_PROXY','VISBENCH_TOOLS_HTTP_PROXY'} <= TOOL_ENV_KEYS


def test_explicit_blank_disables_proxy_alias():
    env=tool_environment({'VISBENCH_TOOLS_HTTPS_PROXY':'','https_proxy':'http://localhost:7890'})
    assert env['VISBENCH_TOOLS_HTTPS_PROXY'] == ''


def test_proxy_mode_keeps_public_url_guard_and_manual_redirects():
    from vissearch_bench.network import ProxyToolSession
    requests=[]
    session=SimpleNamespace(trust_env=True,proxies={},get=lambda url,**kw: requests.append((url,kw)))
    def validate(url,**kw):
        if '127.0.0.1' in url: raise ValueError('private address')
    proxy=ProxyToolSession({'https':'http://localhost:7890'},validate,session=session)
    assert session.trust_env is False
    with pytest.raises(ValueError):
        proxy.get('https://127.0.0.1/',headers={},stream=True,allow_redirects=False,timeout=3)
    assert requests == []
    proxy.get('https://example.org/image',headers={},stream=True,allow_redirects=False,timeout=3)
    assert requests[0][1]['allow_redirects'] is False
    with pytest.raises(ValueError):
        proxy.get('https://example.org/image',headers={},stream=True,allow_redirects=True,timeout=3)


def test_proxy_credentials_never_enter_protocol():
    from vissearch_bench.tool_config import network_profile
    value=network_profile(tool_environment({'https_proxy':'http://user:password@localhost:7890'}))
    assert 'password' not in str(value) and 'user:' not in str(value)
    assert value['mode'] == 'operator-proxy'
    assert value != network_profile(tool_environment({'https_proxy':'http://user:different@localhost:7890'}))


def test_scheme_without_proxy_retains_pinned_transport():
    from vissearch_bench.network import ProxyToolSession
    calls=[]
    fallback=SimpleNamespace(get=lambda url,**kw: calls.append(('pinned',url)),
                             post=lambda url,**kw: calls.append(('pinned-post',url)))
    session=SimpleNamespace(trust_env=True,proxies={},get=lambda *a,**k: pytest.fail('unproxied requests GET'))
    proxy=ProxyToolSession({'https':'http://localhost:7890'},lambda *a,**k: None,session=session,fallback=fallback)
    proxy.get('http://example.org/',stream=True,allow_redirects=False,timeout=3,headers={})
    proxy.post('http://example.org/api',allow_redirects=False,json={})
    assert calls == [('pinned','http://example.org/'),('pinned-post','http://example.org/api')]


def test_real_actor_sdk_ignores_ambient_socks_proxy_without_network():
    import os
    profile = {'provider':'openai','model':'Qwen3.6-35B-A3B','disable_thinking':True}
    with patch.dict(os.environ, {'ALL_PROXY':'socks5://127.0.0.1:1',
                                'all_proxy':'socks5://127.0.0.1:1'}):
        client = react._sdk_factory(profile)(api_key='offline-test',base_url='https://example.com/v1',max_retries=0)
        try:
            assert client._client.trust_env is False
            assert client.max_retries == 0
        finally:
            client.close()


def test_real_judge_sdk_ignores_ambient_socks_proxy_without_network():
    import os
    import openai
    from vissearch_bench.network import isolated_model_client
    from vissearch_bench.providers import APIClient
    with patch.dict(os.environ, {'ALL_PROXY':'socks5://127.0.0.1:1',
                                'all_proxy':'socks5://127.0.0.1:1'}):
        judge = APIClient({'provider':'openai','model':'judge','api_key_env':'KEY','retries':1},
                          {'KEY':'offline-test'},
                          client_factory=lambda **kw: isolated_model_client(openai.OpenAI,**kw))
        client = judge._client()
        try:
            assert client._client.trust_env is False and client.max_retries == 0
        finally:
            client.close()
