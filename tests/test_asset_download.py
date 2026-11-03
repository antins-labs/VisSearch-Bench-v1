import io
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image
from requests import HTTPError

from vissearch_bench.errors import AgentError
from vissearch_bench.react.agent import ToolCall
from vissearch_bench.react.tools import ReActToolbox


@pytest.mark.parametrize('status', [401,403,404,410,500,502,503,504])
def test_unavailable_source_image_uses_thumbnail_without_repeat_search(status):
    buffer=io.BytesIO();Image.new('RGB',(4,4),'blue').save(buffer,format='PNG')
    response=SimpleNamespace(status_code=200,raise_for_status=lambda:None,close=lambda:None,
        json=lambda:{'images':[{'title':'Public figure','link':'https://example.org/page',
            'imageUrl':'https://example.org/full.png','thumbnailUrl':'https://example.org/thumb.png'}]})
    box=ReActToolbox(image_base_dir=Path.cwd(),image_references=[],serper_api_key='key',
        serper_base_url='https://example.org',jina_api_key='',jina_base_url='https://example.org',
        http_session=SimpleNamespace(post=lambda *a,**k:response))
    error=HTTPError('unavailable',response=SimpleNamespace(status_code=status))
    with patch.object(box,'_download_bytes',side_effect=[error,buffer.getvalue()]) as fetch:
        result=box.execute(ToolCall('x','image_search',{'query':'public'}))
    assert len(result.images)==1 and fetch.call_count==2
    assert result.metadata['loaded_image_results'][0]['download_errors']


def test_thumbnail_service_failure_is_not_silenced():
    box=ReActToolbox(image_base_dir=Path.cwd(),image_references=[],serper_api_key='',
        serper_base_url='https://example.org',jina_api_key='',jina_base_url='https://example.org',
        http_session=SimpleNamespace())
    error=HTTPError('unavailable',response=SimpleNamespace(status_code=503))
    with patch.object(box,'_download_bytes',side_effect=error) as fetch:
        with pytest.raises(AgentError) as caught:
            box._render_image_results([{'imageUrl':'https://origin.example/full.png',
                'thumbnailUrl':'https://thumb.example/image.png'}],metadata={},empty_text='none')
    assert caught.value.infrastructure and fetch.call_count==2


def test_source_rate_limit_still_stops_without_thumbnail_retry():
    box=ReActToolbox(image_base_dir=Path.cwd(),image_references=[],serper_api_key='',
        serper_base_url='https://example.org',jina_api_key='',jina_base_url='https://example.org',
        http_session=SimpleNamespace())
    error=HTTPError('limited',response=SimpleNamespace(status_code=429))
    with patch.object(box,'_download_bytes',side_effect=error) as fetch:
        with pytest.raises(AgentError) as caught:
            box._render_image_results([{'imageUrl':'https://example.org/full.png','thumbnailUrl':'https://example.org/thumb.png'}],metadata={},empty_text='none')
    assert caught.value.infrastructure and fetch.call_count==1
