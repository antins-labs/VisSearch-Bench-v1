"""Shared evidence-tool identity, budgets and credential aliases."""
from pathlib import Path
import hashlib
from urllib.parse import urlsplit

TOOL_LIMITS = {
    'max_search_results': 5, 'max_image_results': 5, 'max_page_chars': 12000,
    'max_total_images': 20, 'max_crop_upscale': 4,
    'max_download_bytes': 8 * 1024 * 1024,
    'max_total_download_bytes': 64 * 1024 * 1024,
    'max_image_pixels': 25_000_000,
}
TOOL_SWITCHES = {}
TOOL_ENDPOINTS = {
    'SERPER_BASE_URL': 'https://google.serper.dev',
    'SERPER_SCRAPE_BASE_URL': 'https://scrape.serper.dev',
    'SERPAPI_BASE_URL': 'https://serpapi.com',
}
PROXY_KEYS = {'http':'VISBENCH_TOOLS_HTTP_PROXY', 'https':'VISBENCH_TOOLS_HTTPS_PROXY'}
TOOL_ENV_KEYS = {*TOOL_ENDPOINTS, *PROXY_KEYS.values(), 'SERPER_API_KEY', 'SERPAPI_KEY'}
TOOL_CACHE = Path('.cache/vissearch_bench/tools')
TOOL_PROFILE = {
    'name': 'shared-five-tools-v2',
    'tools': ['web_search', 'open_page', 'image_search', 'crop_image', 'reverse_image_search'],
    'python_execution': False,
    'page_reader_backend': 'serper-scrape',
    'reverse_image_backend': 'serpapi-google-lens-upload',
    'cache': 'successful-responses-by-content-and-settings-v1',
}


def tool_environment(environ):
    """Honor explicit values, including blank ones, before legacy aliases."""
    result = dict(environ)
    for key, alias in [('SERPAPI_KEY', 'SERPAPI_API_KEY')]:
        if key not in result and alias in result:
            result[key] = result[alias]
    for scheme, key in PROXY_KEYS.items():
        if key not in result:
            result[key] = result.get(scheme + '_proxy', result.get(scheme.upper() + '_PROXY', ''))
    return result


def tool_proxies(environ):
    result = {}
    for scheme, key in PROXY_KEYS.items():
        value = environ.get(key, '').strip()
        if not value:
            continue
        try:
            parsed = urlsplit(value)
            valid = (parsed.scheme in {'http','https'} and parsed.hostname and parsed.port != 0
                     and not parsed.query and not parsed.fragment and parsed.path in {'','/'})
        except ValueError:
            valid = False
        if not valid:
            raise ValueError('Tool proxy requires an explicit HTTP(S) proxy URL')
        result[scheme] = value
    return result


def network_profile(environ):
    proxies = tool_proxies(environ)
    return {'mode': 'operator-proxy' if proxies else 'direct-pinned',
            'proxy_config_sha256': {scheme:hashlib.sha256(value.encode()).hexdigest()
                                    for scheme,value in proxies.items()}}
