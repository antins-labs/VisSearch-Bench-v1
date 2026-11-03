"""Independent ReAct workers must share successful cache fills, not requests."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('namespace', ['serper_web_search', 'serper_open_page'])
def test_same_key_three_processes_fetch_once(tmp_path, namespace):
    script = f'''
import sys,time
from pathlib import Path
sys.path.insert(0,{str(SCRIPTS)!r})
from vissearch_bench.react.cache import JsonToolCache
root=Path(sys.argv[1]); ident=sys.argv[2]
(root/('ready'+ident)).touch()
deadline=time.monotonic()+10
while len(list(root.glob('ready*')))<3:
    assert time.monotonic()<deadline
    time.sleep(.01)
def fetch():
    with (root/'fetches').open('a') as f: f.write(ident+'\\n')
    time.sleep(.2)
    return {{'content':'shared evidence'}}
value,hit=JsonToolCache(root/'cache').get_or_fetch({namespace!r},{{'url':'same'}},fetch)
assert value=={{'content':'shared evidence'}}
(root/('result'+ident)).write_text(str(hit))
'''
    children = [subprocess.Popen([sys.executable, '-I', '-c', script, str(tmp_path), str(i)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(3)]
    try:
        for child in children:
            out, err = child.communicate(timeout=15)
            assert child.returncode == 0, out + err
        assert len((tmp_path/'fetches').read_text().splitlines()) == 1
        assert sorted(p.read_text() for p in tmp_path.glob('result*')) == ['False', 'True', 'True']
        saved = list((tmp_path/'cache').rglob('*.json'))
        assert len(saved) == 1
        assert json.loads(saved[0].read_text())['response']['content'] == 'shared evidence'
    finally:
        for child in children:
            if child.poll() is None: child.kill()
            child.wait()


def test_failed_fetch_releases_process_lock_without_caching(tmp_path):
    from vissearch_bench.react.cache import JsonToolCache
    cache=JsonToolCache(tmp_path)
    def fail(): raise TimeoutError('offline failure')
    with pytest.raises(TimeoutError): cache.get_or_fetch('web', {'q':'same'}, fail)
    assert not list(tmp_path.rglob('*.json'))
    assert cache.get_or_fetch('web', {'q':'same'}, lambda: {'ok':True}) == ({'ok':True}, False)
