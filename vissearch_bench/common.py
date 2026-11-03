"""Portable profiles, safe artifacts, and resumable run identities."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

from .data import DATASET_ID, DATASET_REVISION
from .errors import scrub


def credential_environment(profile, environ):
    result = dict(environ)
    value = environ.get(profile.get('api_key_env', ''), '').strip()
    if value:
        result['VISSEARCH_CONFIGURED_API_KEY'] = value
    return result


def load_profile(path, environ, *, offline=False):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    allowed = {'provider','model','model_env','base_url','base_url_env','api_key_env',
               'max_tokens','temperature','timeout','reasoning_effort','disable_thinking'}
    if set(config)-allowed: raise ValueError('Unknown profile fields; credentials must use api_key_env')
    provider = config.get('provider', 'openai')
    if provider not in {'openai','anthropic'}: raise ValueError('Unsupported provider')
    model = environ.get(config.get('model_env',''), '') or config.get('model')
    if not model:
        if offline: model = 'offline-validation'
        else: raise ValueError('Set the model environment variable specified by the profile')
    url = environ.get(config.get('base_url_env',''), '') or config.get('base_url')
    url = url or ('https://api.openai.com/v1' if provider=='openai' else 'https://api.anthropic.com')
    parts = urlsplit(url)
    if parts.scheme not in {'http','https'} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError('Base URL must be HTTP(S), without credentials, query, or fragment')
    key_env = config.get('api_key_env', 'MODEL_API_KEY')
    if not isinstance(key_env,str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',key_env):
        raise ValueError('api_key_env must name an environment variable')
    result = {'provider':provider,'model':model,'base_url':url,'api_key_env':key_env,
              'max_tokens':config.get('max_tokens',32768),'temperature':config.get('temperature',0),
              'timeout':config.get('timeout',180),'retries':1}
    for key in ('reasoning_effort','disable_thinking'):
        if key in config: result[key] = config[key]
    if type(result['max_tokens']) is not int or result['max_tokens']<1: raise ValueError('Invalid max_tokens')
    for key in ('temperature','timeout'):
        value = result[key]
        if value is None and key=='temperature': continue
        if type(value) not in (int,float) or not math.isfinite(value) or value<0 or key=='timeout' and value==0:
            raise ValueError(f'Invalid {key}')
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def task_fingerprint(tasks):
    normalized=[]
    for task in tasks:
        value={k:v for k,v in task.items() if k not in {'images','_image_paths'}}
        value['image_sha256']=[hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in task['_image_paths']]
        normalized.append(value)
    return digest(normalized)


def code_fingerprint():
    root=Path(__file__).parent
    return digest({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*.py'))})


def save_json(path, value, environ=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(scrub(value,environ),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def save_jsonl(path, records, environ=None):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(''.join(json.dumps(scrub(r,environ),ensure_ascii=False,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    temporary.replace(path)


def read_predictions(path):
    records=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    ids=[r.get('id') for r in records]
    if any(type(i) is not int or i<1 for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('Invalid or duplicate prediction ids')
    return records


@contextmanager
def run_directory(path, *, fingerprint, resume):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    with (path/'.lock').open('a+b') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('Another process owns this run directory') from None
        try:
            manifest=path/'manifest.json'
            if manifest.exists():
                if not resume: raise ValueError('Output exists; use --resume or a new output directory')
                if json.loads(manifest.read_text())['fingerprint']!=fingerprint:
                    raise ValueError('Run fingerprint changed; choose a new output directory')
            else:
                if any(p.name!='.lock' for p in path.iterdir()): raise ValueError('Output directory is not empty')
                save_json(manifest,{'fingerprint':fingerprint,'format_version':1})
            yield path
        finally: fcntl.flock(lock,fcntl.LOCK_UN)


def add_dataset_arguments(parser):
    parser.add_argument('--dataset',default=DATASET_ID,help='Hub dataset id or local JSONL/Parquet')
    parser.add_argument('--revision',default=DATASET_REVISION,help='Pinned Hub revision; ignored for local files')
    parser.add_argument('--split',default='test')
    parser.add_argument('--cache-dir',type=Path,default=Path('.cache/vissearch_bench'))
    parser.add_argument('--ids',help='Comma-separated task ids; release order is preserved')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--dry-run',action='store_true',help='Validate data without model/search requests')


def dataset_arguments(args, *, include_images=True):
    return dict(source=args.dataset,split=args.split,revision=args.revision,cache_dir=args.cache_dir,
                ids=[int(x) for x in args.ids.split(',')] if args.ids else None,
                limit=args.limit,include_images=include_images)
