"""Safe failure metadata and secret-free event serialization."""
import re


class AgentError(RuntimeError):
    def __init__(self, message, *, infrastructure=False, code=None):
        super().__init__(message)
        self.infrastructure = infrastructure
        self.code = code or message


def http_status(exc):
    status = getattr(exc, 'status_code', None)
    if status is None:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
    # Old bounded HTTP transport records only "HTTP <status>".
    match = re.fullmatch(r'HTTP (\d{3})', str(exc))
    if status is None and match:
        status = int(match[1])
    return status


def safe_exception(exc):
    if isinstance(exc, AgentError):
        return exc
    from .providers import CompletionError
    if isinstance(exc, CompletionError):
        label = str(exc)
        pattern = r'provider=(openai|anthropic) model=[A-Za-z0-9_.:/-]+ error=[A-Za-z0-9_]+ status=(unknown|\d{3}) attempts=\d+'
        if re.fullmatch(pattern, label):
            return AgentError(label, infrastructure=True, code=label)
    name = type(exc).__name__
    status = http_status(exc)
    infrastructure = (status in (401, 402, 403, 408, 409, 429) or
                      (isinstance(status, int) and status >= 500) or
                      any(token in name.lower() for token in ('timeout', 'connection', 'transport', 'credential', 'network')))
    # Unknown SDK/runtime errors are not evidence of a benchmark failure.
    if not isinstance(exc, (ValueError, TypeError)) and status is None:
        infrastructure = True
    label = f'{name}' + (f' HTTP {status}' if isinstance(status, int) else '')
    return AgentError(label, infrastructure=infrastructure, code=label)


def transient_model_failure(part):
    """Recognize saved SDK transport/service failures, never tool or auth errors."""
    if part.get('status') != 'error' or not part.get('infrastructure'):
        return False
    attempts = part.get('api_attempts', [])
    if attempts:
        kind = attempts[-1].get('error_class', '')
        status = attempts[-1].get('http_status')
    else:
        match = re.fullmatch(r'([A-Za-z]+)(?: HTTP (\d{3}))?', part.get('error', ''))
        if not match: return False
        kind = match[1]
        status = int(match[2]) if match[2] else None
    if status in (401, 402, 403): return False
    if kind in {'APIConnectionError', 'APITimeoutError'} and status is None:
        return True
    return (kind in {'APIStatusError', 'APIError', 'RateLimitError', 'InternalServerError',
                     'ConflictError', 'RequestTimeoutError'}
            and isinstance(status, int) and (status in (408, 409, 429) or 500 <= status <= 599))


def recoverable_tool_error(exc):
    """Tool-only policy: return failed service results to the agent, without retry."""
    error = safe_exception(exc)
    label = str(error.code)
    if 'credential' in label.lower() or str(exc) == 'SERPER_API_KEY is required for search tools':
        return False
    status = http_status(exc)
    if status is None and (match := re.search(r'HTTP\s*(\d{3})', label)):
        status = int(match[1])
    if status in (401, 402, 403):
        return False
    response_failure = (
        label.startswith(('Serper', 'SerpApi')) and
        (any(token in label for token in ('Malformed', 'ServiceError', 'EmptyContent',
                                         'ResponseTooLarge', 'JSONDecodeError', 'UnicodeDecodeError'))
         or label.startswith(('Serper web search ', 'Serper image search '))))
    transport_failure = any(label == prefix + kind
                         for prefix in ('', 'Serper', 'SerpApi')
                         for kind in ('ChunkedEncodingError', 'ContentDecodingError', 'ProtocolError',
                                      'DecodeError', 'IncompleteRead', 'InvalidChunkLength',
                                      'ResponseNotChunked', 'SSLError', 'ProxyError'))
    return (not error.infrastructure or response_failure or transport_failure or
            isinstance(status, int) and 300 <= status < 600 or
            any(word in label.lower() for word in ('timeout', 'connection', 'transport', 'network', 'dns', 'requestfailure')))


def failed_tool_observation(call, exc):
    from .react.agent import ToolObservation
    error = safe_exception(exc)
    return ToolObservation(
        tool_call_id=call.id, tool_name=call.name,
        text=f'Tool failed: {error.code}. No evidence was retrieved. Continue using available evidence or choose another source or tool.',
        metadata={'error': True, 'error_code': error.code, 'recoverable': True,
                  'infrastructure': error.infrastructure, 'sources': []})


def scrub(value, environ=None):
    secrets = [v for k, v in (environ or {}).items()
               if any(word in k.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'PROXY'))
               and isinstance(v, str) and (len(v) >= 6 or k == 'VISSEARCH_CONFIGURED_API_KEY' and v)]
    def clean(item):
        if isinstance(item, str):
            item = re.sub(r'data:[^;,\s]+;base64,[A-Za-z0-9+/=\s]+', '[image data omitted]', item)
            for secret in secrets:
                item = item.replace(secret, '[REDACTED]')
            return item
        if isinstance(item, dict):
            return {str(k): ('[REDACTED]' if str(k).lower() in {'api_key', 'authorization', 'x-api-key'} else clean(v))
                    for k, v in item.items()}
        if isinstance(item, (list, tuple)): return [clean(v) for v in item]
        if item is None or isinstance(item, (int, float, bool)): return item
        return type(item).__name__
    return clean(value)
