"""Explicit operator proxy transport for public evidence tools.

Direct requests retain the original pinned-IP transport. In proxy mode the
operator's proxy resolves hostnames remotely; local public-address checks and
manual redirect checks still apply, but are not a remote-DNS pinning guarantee.
"""
from urllib.parse import urlsplit


def isolated_model_client(constructor, **kwargs):
    """Model and judge SDKs must not inherit the evidence tools' OS proxy."""
    import httpx
    transport = httpx.Client(trust_env=False)
    try:
        return constructor(http_client=transport, **kwargs)
    except Exception:
        transport.close()
        raise


class ProxyToolSession:
    def __init__(self, proxies, validate_public, *, session=None, fallback=None):
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.session.trust_env = False
        self.session.proxies = dict(proxies)
        self.validate_public = validate_public
        self.fallback = fallback

    def _transport(self, url):
        if urlsplit(url).scheme in self.session.proxies:
            return self.session
        if self.fallback is None:
            raise ValueError('No transport configured for target URL scheme')
        return self.fallback

    @staticmethod
    def _no_redirects(kwargs):
        if kwargs.get('allow_redirects') is not False:
            raise ValueError('Tool HTTP requests require manual redirects')

    def get(self, url, **kwargs):
        self._no_redirects(kwargs)
        if kwargs.get('stream') is not True:
            raise ValueError('Asset GET requires bounded streaming')
        self.validate_public(url, label='tool GET URL')
        return self._transport(url).get(url, **kwargs)

    def post(self, url, **kwargs):
        self._no_redirects(kwargs)
        return self._transport(url).post(url, **kwargs)

    def get_json(self, url, **kwargs):
        self._no_redirects(kwargs)
        transport = self._transport(url)
        method = getattr(transport, 'get_json', None) or transport.get
        return method(url, **kwargs)
