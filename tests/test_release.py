"""Offline contracts for the public commands; never call paid services."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from vissearch_bench.data import load_tasks, normalize_task
from vissearch_bench.evaluation import build_judge_messages, parse_score
from vissearch_bench.common import load_profile
from vissearch_bench.inference import run_inference
from vissearch_bench.evaluate import run_evaluation

PROMPT = 'Question: {{question}}\nReference: {{reference_answer}}\nPrediction: {{model_answer}}'


def task(tmp_path, task_id=1):
    image = tmp_path / 'input.png'
    Image.new('RGB', (12, 8), 'red').save(image)
    return normalize_task({'id': task_id, 'question_en': 'Name the color.',
        'answer': 'PRIVATE_REFERENCE', 'gold_trajectory': 'PRIVATE_SOLUTION',
        'accuracy_prompt': PROMPT, 'Process Score': PROMPT + '\nPartial credit.',
        'category_name': 'Example', 'domain': 'Example', 'images': ['input.png']},
        base_dir=tmp_path, cache_dir=tmp_path / 'cache')


def profile():
    return {'provider':'openai','model':'offline-model','base_url':'https://example.invalid/v1',
            'api_key_env':'MODEL_API_KEY','max_tokens':1024,'temperature':0,'timeout':10,'retries':1}


class Client:
    def __init__(self, responses): self.responses, self.messages = iter(responses), []
    def check_credentials(self): pass
    def complete(self, messages):
        self.messages.append(messages)
        value = next(self.responses)
        if isinstance(value, Exception): raise value
        return {'text': value, 'usage': {}, 'finish_reason': 'stop'}


def test_local_and_embedded_image_loading(tmp_path):
    row = task(tmp_path)
    raw = {k:v for k,v in row.items() if not k.startswith('_')}
    raw['images'] = [{'bytes': (tmp_path/'input.png').read_bytes()}]
    result = normalize_task(raw, base_dir=tmp_path, cache_dir=tmp_path/'embedded')
    assert len(result['_image_paths']) == 1
    assert Image.open(result['_image_paths'][0]).size == (12,8)
    raw['images'] = []
    assert normalize_task(raw, base_dir=tmp_path, cache_dir=tmp_path/'empty')['_image_paths'] == []


def test_loader_preserves_order_and_rejects_duplicates(tmp_path):
    rows = [task(tmp_path, i) for i in [3,1,2]]
    path=tmp_path/'data.jsonl'
    path.write_text('\n'.join(json.dumps({k:v for k,v in r.items() if not k.startswith('_')}) for r in rows))
    assert [r['id'] for r in load_tasks(str(path), cache_dir=tmp_path/'cache', ids=[1,3])] == [3,1]
    path.write_text(path.read_text()+'\n'+path.read_text().splitlines()[0])
    with pytest.raises(ValueError, match='Duplicate'): load_tasks(str(path), cache_dir=tmp_path/'cache')


def test_profiles_reject_literal_keys(tmp_path):
    path=tmp_path/'profile.json'
    path.write_text(json.dumps({'provider':'openai','model':'example','api_key':'do-not-store-secrets'}))
    with pytest.raises(ValueError): load_profile(path, {}, offline=True)


def test_direct_isolates_gold_and_scrubs_outputs(tmp_path):
    row=task(tmp_path); client=Client(['red secret-value-123'])
    run_inference([row],profile(),tmp_path/'run',method='direct',environ={'MODEL_API_KEY':'secret-value-123'},client=client)
    sent=json.dumps(client.messages)
    assert 'PRIVATE_REFERENCE' not in sent and 'PRIVATE_SOLUTION' not in sent
    assert 'image_url' in sent
    saved=(tmp_path/'run/predictions.jsonl').read_text()
    assert 'secret-value-123' not in saved and '[REDACTED]' in saved
    run_inference([row],profile(),tmp_path/'run',method='direct',environ={},client=Client([]),resume=True)
    assert len((tmp_path/'run/predictions.jsonl').read_text().splitlines())==1
    with pytest.raises(ValueError,match='fingerprint'):
        run_inference([dict(row,question_en='Changed')],profile(),tmp_path/'run',method='direct',environ={},client=Client([]),resume=True)


def test_react_receives_only_public_inputs(tmp_path):
    row=task(tmp_path); seen={}
    def generate(**kwargs):
        seen.update(kwargs);kwargs['event_sink']({'event':'model_response','text':'secret-value-123'})
        return {'text':'red','usage':{},'finish_reason':'submit_answer'}
    run_inference([row],profile(),tmp_path/'run',method='react',environ={'MODEL_API_KEY':'secret-value-123'},generate=generate)
    assert seen['question']==row['question_en'] and seen['image_paths']==row['_image_paths']
    assert 'PRIVATE_REFERENCE' not in str(seen) and 'PRIVATE_SOLUTION' not in str(seen)
    assert 'secret-value-123' not in (tmp_path/'run/traces/1.json').read_text()


def test_grading_and_literal_template_preservation(tmp_path):
    row=task(tmp_path)
    messages=build_judge_messages(row,'literal {{reference_answer}}','accuracy_prompt')
    assert 'Prediction: literal {{reference_answer}}' in messages[1]['content']
    assert parse_score('Reason\n{"score": 1}',binary=True)==1
    for text in ['{"score":0.5}','{"score":true}','{"score":1,"score":0}']:
        with pytest.raises(ValueError): parse_score(text,binary=True)
    assert parse_score('{"score":0.25}')==0.25


def test_evaluation_scores_independently_and_supports_resume(tmp_path):
    rows=[task(tmp_path)];pred=[{'id':1,'status':'ok','answer':'red'}]
    client=Client(['{"score":0}','{"score":0.5}'])
    summary=run_evaluation(rows,pred,profile(),tmp_path/'scores',environ={},client=client)
    assert summary['accuracy']==0 and summary['process_score']==0.5
    assert len(client.messages)==2
    again=run_evaluation(rows,pred,profile(),tmp_path/'scores',environ={},client=Client([]),resume=True)
    assert again==summary


def test_missing_prediction_is_explicit_not_dropped(tmp_path):
    rows=[task(tmp_path,1),task(tmp_path,2)]
    pred=[{'id':1,'status':'ok','answer':'red'}]
    with pytest.raises(ValueError,match='Missing'):
        run_evaluation(rows,pred,profile(),tmp_path/'strict',environ={},client=Client([]))
    result=run_evaluation(rows,pred,profile(),tmp_path/'zero',environ={},client=Client(['{"score":1}','{"score":0.5}']),failure_policy='zero')
    assert result['num_tasks']==2 and result['accuracy']==0.5 and result['process_score']==0.25
    assert result['num_zeroed']==1


def test_missing_key_does_not_create_failed_run(tmp_path):
    from vissearch_bench.providers import CompletionError
    with pytest.raises(CompletionError):
        run_inference([task(tmp_path)],profile(),tmp_path/'bad-direct',method='direct',environ={})
    assert not (tmp_path/'bad-direct').exists()
    with pytest.raises(CompletionError):
        run_evaluation([task(tmp_path)],[{'id':1,'answer':'red'}],profile(),tmp_path/'bad-judge',environ={})
    assert not (tmp_path/'bad-judge').exists()


def test_default_provider_ignores_tool_proxy(monkeypatch):
    from vissearch_bench.providers import APIClient
    monkeypatch.setenv('HTTPS_PROXY','http://127.0.0.1:1')
    client=APIClient(profile(),{'MODEL_API_KEY':'offline-value'})._client()
    try: assert client._client.trust_env is False
    finally: client.close()


def test_arbitrary_credential_env_name_is_redacted(tmp_path):
    config=dict(profile(),api_key_env='MY_CREDENTIAL')
    run_inference([task(tmp_path)],config,tmp_path/'custom-key',method='direct',
                  environ={'MY_CREDENTIAL':'  private-value  '},client=Client(['private-value']))
    assert 'private-value' not in (tmp_path/'custom-key/predictions.jsonl').read_text()


def test_provider_failure_keeps_safe_diagnostics(tmp_path):
    from vissearch_bench.providers import CompletionError
    error=CompletionError('provider=openai model=offline-model error=AuthenticationError status=401 attempts=1',
                          [{'attempt':1,'status':'error','error_class':'AuthenticationError','http_status':401}])
    run_inference([task(tmp_path)],profile(),tmp_path/'diagnostic',method='direct',environ={},client=Client([error]))
    result=json.loads((tmp_path/'diagnostic/predictions.jsonl').read_text())
    assert '401' in result['error'] and 'AuthenticationError' in result['error']
