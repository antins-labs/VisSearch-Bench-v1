"""Offline distinction between an image-source paywall and provider payment failure."""
import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image
from requests import HTTPError

from vissearch_bench.errors import AgentError, recoverable_tool_error
from vissearch_bench.react.reverse_image import fetch_visual_matches
from vissearch_bench.react.tools import ReActToolbox


def toolbox():
    return ReActToolbox(image_base_dir=Path.cwd(), image_references=[], serper_api_key='',
                       serper_base_url='https://google.serper.dev', jina_api_key='',
                       jina_base_url='https://r.jina.ai', http_session=SimpleNamespace())


def test_paywalled_image_source_uses_independent_thumbnail():
    image = io.BytesIO()
    Image.new('RGB', (4, 4), 'blue').save(image, format='PNG')
    box = toolbox()
    error = HTTPError('image paywall', response=SimpleNamespace(status_code=402))
    with patch.object(box, '_download_bytes', side_effect=[error, image.getvalue()]) as download:
        result = box._render_image_results(
            [{'link': 'https://example.org/page', 'imageUrl': 'https://example.org/full.png',
              'thumbnailUrl': 'https://example.net/thumb.png'}], metadata={'cache_hit': True}, empty_text='none')
    assert download.call_count == 2
    assert len(result['images']) == 1
    assert result['metadata']['cache_hit'] is True
    assert result['metadata']['loaded_image_results'][0]['download_errors'] == ['HTTPError HTTP 402']


def test_paywalled_image_and_thumbnail_keep_text_source_without_fabricated_pixels():
    box = toolbox()
    error = HTTPError('image paywall', response=SimpleNamespace(status_code=402))
    with patch.object(box, '_download_bytes', side_effect=error) as download:
        result = box._render_image_results(
            [{'link': 'https://example.org/page', 'imageUrl': 'https://example.org/full.png',
              'thumbnailUrl': 'https://example.net/thumb.png'}], metadata={'cache_hit': True}, empty_text='none')
    assert download.call_count == 2
    assert result['images'] == []
    assert 'not loaded' in result['text'] and 'HTTPError HTTP 402' in result['text']
    assert result['metadata']['sources'] == ['https://example.org/page']


def test_real_provider_http402_is_still_fatal():
    response = SimpleNamespace(status_code=402, close=lambda: None)
    http = SimpleNamespace(post=lambda *a, **kw: response)
    with pytest.raises(AgentError) as caught:
        fetch_visual_matches(http, base_url='https://serpapi.com', api_key='offline-key',
                             upload_jpeg=b'offline-fixture', timeout=30)
    assert caught.value.code == 'SerpApiUploadHTTP402'
    assert caught.value.infrastructure is True
    assert recoverable_tool_error(caught.value) is False
