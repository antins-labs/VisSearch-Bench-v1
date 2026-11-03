"""Generate answers with Direct or the native five-tool ReAct agent."""
import argparse
import json
import os
from pathlib import Path

from .common import (add_dataset_arguments, code_fingerprint, credential_environment, dataset_arguments, digest,
                     load_profile, run_directory, save_json, save_jsonl, task_fingerprint)
from .data import load_tasks
from .errors import AgentError, safe_exception, scrub
from .evaluation import build_answer_messages
from .providers import APIClient, CompletionError

DEFAULT_SETTINGS = {'max_turns':30,'max_total_tool_calls':60,'max_tool_calls_per_turn':4,'image_max_side':1536}


def run_inference(tasks, profile, output, *, method, environ, settings=None, resume=False,
                  client=None, generate=None):
    if method not in {'direct','react'}: raise ValueError('Unknown inference method')
    environ=credential_environment(profile,environ)
    settings={**DEFAULT_SETTINGS,**(settings or {})}
    if any(type(settings[k]) is not int or settings[k]<1 for k in DEFAULT_SETTINGS):
        raise ValueError('Inference budgets must be positive integers')
    fingerprint=digest({'tasks':task_fingerprint(tasks),'profile':profile,'method':method,
                        'settings':settings,'code':code_fingerprint()})
    if method=='direct':
        client=client or APIClient(profile,environ)
        client.check_credentials()
    elif generate is None:
        from .react import generate
        from .react import credential
        credential(profile,environ)
        if not environ.get('SERPER_API_KEY') or not (environ.get('SERPAPI_KEY') or environ.get('SERPAPI_API_KEY')):
            raise ValueError('ReAct requires SERPER_API_KEY and SERPAPI_KEY')
    records=[]
    with run_directory(output,fingerprint=fingerprint,resume=resume) as directory:
        for task in tasks:
            record_path=directory/'records'/f'{task["id"]}.json'
            if record_path.exists():
                records.append(json.loads(record_path.read_text()))
                continue
            record={'id':task['id'],'method':method,'status':'pending'}
            events=[]
            try:
                if method=='direct':
                    response=client.complete(build_answer_messages(task,settings['image_max_side']))
                else:
                    def emit(event):
                        events.append(scrub(event,environ))
                        save_json(directory/'traces'/f'{task["id"]}.json',events,environ)
                    response=generate(question=task['question_en'],image_paths=task['_image_paths'],
                                      profile=profile,environ=environ,settings=settings,event_sink=emit)
                if response.get('finish_reason') in {'length','max_tokens','incomplete'}:
                    raise ValueError('Truncated response')
                if not isinstance(response.get('text'),str) or not response['text'].strip():
                    raise ValueError('Empty answer')
                record.update(status='ok',answer=response['text'],usage=response.get('usage',{}),
                              finish_reason=response.get('finish_reason'))
                if method=='react': record['trace_file']=f'traces/{task["id"]}.json'
            except Exception as exc:
                error=safe_exception(exc)
                record.update(status='error',error=str(error),infrastructure=error.infrastructure)
            record=scrub(record,environ)
            save_json(record_path,record,environ)
            records.append(record)
            save_jsonl(directory/'predictions.jsonl',records,environ)
            print(f'Task {task["id"]}: {record["status"]}',flush=True)
        save_jsonl(directory/'predictions.jsonl',records,environ)
    return records


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument('--method',choices=['direct','react'],required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs/inference'))
    parser.add_argument('--resume',action='store_true')
    for key,value in DEFAULT_SETTINGS.items():
        parser.add_argument('--'+key.replace('_','-'),type=int,default=value)
    args=parser.parse_args(argv)
    try:
        profile=load_profile(args.config,os.environ,offline=args.dry_run)
        tasks=load_tasks(**dataset_arguments(args))
        settings={key:getattr(args,key) for key in DEFAULT_SETTINGS}
        settings['tool_cache_dir']=str(args.cache_dir/'tools')
        if any(settings[k]<1 for k in DEFAULT_SETTINGS): raise ValueError('Budgets must be positive')
        if args.dry_run:
            for task in tasks: build_answer_messages(task,args.image_max_side)
            print(f'Validated {len(tasks)} tasks, {sum(len(t["_image_paths"]) for t in tasks)} images; no model/search requests.')
            return 0
        result=run_inference(tasks,profile,args.output,method=args.method,environ=os.environ,
                             settings=settings,resume=args.resume)
        return 0 if all(r['status']=='ok' for r in result) else 1
    except (ValueError, OSError, CompletionError, AgentError) as exc:
        print(f'Cannot run inference: {scrub(str(exc),os.environ)}')
        return 2


if __name__=='__main__': raise SystemExit(main())
