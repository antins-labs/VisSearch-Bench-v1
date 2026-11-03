import unittest
import io
from PIL import Image
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vissearch_bench.react import generate, CheckedSession, _sdk_factory
from vissearch_bench.react.agent import ModelTurn, ToolCall, ToolObservation, run_react_agent
from vissearch_bench.errors import AgentError
from vissearch_bench.react.tools import ReActToolbox, StoredImage, _public_addresses


class FakeSession:
    def __init__(self, turns): self.turns, self.observations = iter(turns), []
    def complete(self, **kw): return next(self.turns)
    def append_observations(self, obs): self.observations.extend(obs)
    def append_user_text(self, text): pass


def turn(calls=(), text='', stop='tool_calls'):
    return ModelTurn(text, list(calls), {}, 0.0, stop, {})


class ReActAdaptationTests(unittest.TestCase):
    def test_adapter_rejects_hidden_retries_before_sdk_creation(self):
        with patch('vissearch_bench.react._sdk_factory', side_effect=AssertionError('SDK factory forbidden')):
            with self.assertRaises(AgentError):
                generate(question='public',image_paths=[],profile={'retries':2},environ={},settings={},event_sink=lambda e:None)

    def test_truncated_submit_is_never_accepted(self):
        raw = FakeSession([turn([ToolCall('s', 'submit_answer', {'final_answer':'partial'})], stop='length')])
        events = []
        checked = CheckedSession(raw, events.append)
        with self.assertRaises(AgentError): checked.complete()
        self.assertTrue(events)

    def test_empty_does_not_spend_thirty_turns(self):
        checked = CheckedSession(FakeSession([turn(stop='stop')]), lambda e: None)
        with self.assertRaises(AgentError): checked.complete()

    def test_original_loop_observes_then_submits_multipart_answer(self):
        raw = FakeSession([turn([ToolCall('s','web_search',{'query':'q'})]), turn([ToolCall('f','submit_answer',{'final_answer':'Part A. Part B.','intermediate_answers':[]})])])
        client = SimpleNamespace(new_session=lambda **kw: CheckedSession(raw, lambda e: None))
        box = SimpleNamespace(tool_schemas=[{'name':'submit_answer','input_schema':{'type':'object'}}], execute=lambda call: ToolObservation(call.id, call.name, 'evidence'))
        result = run_react_agent(client=client, toolbox=box, question='public', initial_image_data_urls=[], model='fake', max_turns=3, max_tokens=50, temperature=0)
        self.assertEqual(result.prediction['final_answer'], 'Part A. Part B.')
        self.assertEqual(len(raw.observations), 1)
        self.assertEqual(result.main_turns, 2)

    def test_infrastructure_tool_error_escapes_loop(self):
        raw = FakeSession([turn([ToolCall('s','web_search',{'query':'q'})])])
        def fail(call): raise AgentError('HTTP429', infrastructure=True)
        client = SimpleNamespace(new_session=lambda **kw: raw)
        box = SimpleNamespace(tool_schemas=[], execute=fail)
        with self.assertRaises(AgentError):
            run_react_agent(client=client, toolbox=box, question='public', initial_image_data_urls=[], model='fake', max_turns=3, max_tokens=50, temperature=0)

    def test_qwen_profile_flag_and_no_hidden_retries(self):
        requests, constructors = [], []
        def create(**kw): requests.append(kw); return 'response'
        def constructor(**kw):
            constructors.append(kw)
            return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch('vissearch_bench.react.importlib.import_module', return_value=SimpleNamespace(OpenAI=constructor)):
            client = _sdk_factory({'provider':'openai','model':'Qwen3.6-35B-A3B','disable_thinking':True})(max_retries=0)
        client.chat.completions.create(messages=[])
        self.assertEqual(constructors[0]['max_retries'], 0)
        self.assertFalse(constructors[0]['http_client'].trust_env)
        constructors[0]['http_client'].close()
        self.assertEqual(requests[0]['extra_body'], {'chat_template_kwargs':{'enable_thinking':False}})

    def test_zero_answer_is_serialized_without_dropping_it(self):
        raw = FakeSession([turn([ToolCall('f', 'submit_answer', {'final_answer':0,'intermediate_answers':[]})])])
        box = SimpleNamespace(initial_image_data_urls=[], tool_schemas=[], execute=lambda c: None)
        events = []
        profile = {'provider':'openai','model':'fake','api_key_env':'KEY','base_url':'https://example.invalid',
                   'timeout':3,'retries':1,'max_tokens':50,'temperature':0}
        with patch('vissearch_bench.react._sdk_factory'), patch('vissearch_bench.react.OpenAIReActClient', return_value=SimpleNamespace(new_session=lambda **kw: raw)), patch('vissearch_bench.react.make_toolbox', return_value=box):
            result = generate(question='public', image_paths=[], profile=profile, environ={'KEY':'secret-test'}, settings={}, event_sink=events.append)
        self.assertEqual(result['text'], '0')
        self.assertEqual(len(events), 2)

    def test_image_download_network_error_is_not_an_observation(self):
        response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'images':[{'title':'public','imageUrl':'https://example.org/image.png'}]})
        box = ReActToolbox(image_base_dir=Path.cwd(), image_references=[], serper_api_key='key',
                          serper_base_url='https://example.org', jina_api_key='', jina_base_url='https://example.org',
                          http_session=SimpleNamespace(post=lambda *a, **kw: response))
        with patch.object(box, '_download_bytes', side_effect=TimeoutError('secret-request-body')) as download:
            with self.assertRaises(AgentError) as caught: box.execute(ToolCall('x','image_search',{'query':'public'}))
        self.assertTrue(caught.exception.infrastructure)
        self.assertNotIn('secret-request-body', str(caught.exception))
        self.assertEqual(download.call_count, 1)

    def test_dns_failure_remains_infrastructure(self):
        def fail(host): raise OSError('secret')
        with self.assertRaises(AgentError) as caught:
            _public_addresses('example.org', fail, label='image')
        self.assertTrue(caught.exception.infrastructure)

    def test_subpixel_crop_is_at_least_one_pixel(self):
        buffer=io.BytesIO(); Image.new('RGB',(100,100)).save(buffer,format='JPEG')
        blob=buffer.getvalue()
        box=ReActToolbox(image_base_dir=Path.cwd(),image_references=[],serper_api_key='',serper_base_url='https://example.org',
                        jina_api_key='',jina_base_url='https://example.org',http_session=SimpleNamespace())
        box.images['input:1']=StoredImage('input:1',blob,blob,'public.png')
        observation=box.execute(ToolCall('x','crop_image',{'image_id':'input:1','x1':500,'y1':500,'x2':501,'y2':501}))
        self.assertEqual(observation.metadata['crop_size'],[1,1])

    def test_jina_cross_origin_redirect_drops_credentials(self):
        calls=[]
        responses=iter([
            SimpleNamespace(status_code=302,headers={'location':'https://other.example/page'},close=lambda:None),
            SimpleNamespace(status_code=200,headers={},close=lambda:None,raise_for_status=lambda:None,iter_content=lambda **kw:iter([b'Public page']))])
        def get(url,**kw): calls.append((url,dict(kw['headers']))); return next(responses)
        box=ReActToolbox(image_base_dir=Path.cwd(),image_references=[],serper_api_key='',serper_base_url='https://example.org',
                        jina_api_key='SECRET-JINA',jina_base_url='https://r.jina.ai',http_session=SimpleNamespace(get=get),host_resolver=lambda h:['93.184.216.34'])
        box.execute(ToolCall('x','open_page',{'url':'https://source.example/page'}))
        self.assertEqual(calls[0][1]['Authorization'],'Bearer SECRET-JINA')
        self.assertNotIn('Authorization',calls[1][1])

    def test_serper_never_automatically_redirects_auth_header(self):
        calls=[]
        def post(url,**kw):
            calls.append(kw)
            return SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=lambda:{'organic':[]})
        box=ReActToolbox(image_base_dir=Path.cwd(),image_references=[],serper_api_key='secret',serper_base_url='https://example.org',
                        jina_api_key='',jina_base_url='https://example.org',http_session=SimpleNamespace(post=post))
        box.execute(ToolCall('x','web_search',{'query':'public'}))
        self.assertIs(calls[0].get('allow_redirects'),False)

if __name__ == '__main__': unittest.main()
