"""Evaluate saved answers with per-task Accuracy and Process Score rubrics."""
import argparse
import json
import os
from pathlib import Path

from .common import (add_dataset_arguments, code_fingerprint, credential_environment, dataset_arguments, digest,
                     load_profile, read_predictions, run_directory, save_json, save_jsonl, task_fingerprint)
from .data import load_tasks
from .errors import safe_exception, scrub
from .evaluation import build_judge_messages, parse_score
from .providers import APIClient, CompletionError

METRICS={'accuracy':'accuracy_prompt','process_score':'Process Score'}


def summarize(records, metrics):
    result={'num_tasks':len(records),'num_zeroed':sum(r.get('zeroed',False) for r in records)}
    for name in metrics:
        values=[r[name]['score'] for r in records if r.get(name,{}).get('score') is not None]
        result[name]=sum(values)/len(records) if records and len(values)==len(records) else None
        result[name+'_percent']=100*result[name] if result[name] is not None else None
        result[name+'_coverage']=len(values)/len(records) if records else 0
        result[name+'_judged']=sum(r.get(name,{}).get('status')=='ok' for r in records)
    return result


def run_evaluation(tasks, predictions, profile, output, *, environ, metric='both',
                   failure_policy='error',resume=False,client=None):
    if metric not in {'accuracy','process','both'} or failure_policy not in {'error','zero'}:
        raise ValueError('Invalid evaluation policy')
    environ=credential_environment(profile,environ)
    metrics=list(METRICS) if metric=='both' else ['process_score' if metric=='process' else 'accuracy']
    by_id={p['id']:p for p in predictions}
    if len(by_id)!=len(predictions): raise ValueError('Duplicate prediction id')
    selected={t['id'] for t in tasks}
    if set(by_id)-selected: raise ValueError('Predictions contain ids outside the selected tasks')
    if selected-set(by_id) and failure_policy=='error': raise ValueError('Missing predictions; supply all selected tasks or explicitly use --failure-policy zero')
    fingerprint=digest({'tasks':task_fingerprint(tasks),'predictions':predictions,'profile':profile,
                        'metrics':metrics,'failure_policy':failure_policy,'code':code_fingerprint()})
    client=client or APIClient(profile,environ)
    client.check_credentials()
    records=[]
    with run_directory(output,fingerprint=fingerprint,resume=resume) as directory:
        for task in tasks:
            path=directory/'records'/f'{task["id"]}.json'
            record=json.loads(path.read_text()) if path.exists() else {'id':task['id']}
            prediction=by_id.get(task['id'],{})
            answer=prediction.get('answer')
            valid=prediction.get('status','ok')=='ok' and isinstance(answer,str) and bool(answer.strip())
            for name in metrics:
                if name in record: continue
                if not valid:
                    record[name]={'status':'error','score':None,'error':'Missing or failed answer'}
                else:
                    key=METRICS[name]
                    try:
                        response=client.complete(build_judge_messages(task,answer,key))
                        if response.get('finish_reason') in {'length','max_tokens','incomplete'}:
                            raise ValueError('Truncated judge response')
                        binary=name=='accuracy' or task[key]==task['accuracy_prompt']
                        record[name]={'status':'ok','score':parse_score(response['text'],binary=binary),
                                      'prompt_key':key,'judge_response':response['text'],'usage':response.get('usage',{})}
                    except Exception as exc:
                        record[name]={'status':'error','score':None,'error':str(safe_exception(exc))}
                save_json(path,record,environ)
            # Explicit whole-task zero accounting; retain the original judge result.
            if failure_policy=='zero' and any(record[name]['status']!='ok' for name in metrics):
                record['zeroed']=True
                for name in metrics:
                    record[name].setdefault('raw_score',record[name]['score'])
                    record[name]['score']=0.0
            record=scrub(record,environ)
            save_json(path,record,environ)
            records.append(record)
            save_jsonl(directory/'scores.jsonl',records,environ)
            print(f'Task {task["id"]}: '+', '.join(f'{name}={record[name]["score"]}' for name in metrics),flush=True)
        summary=summarize(records,metrics)
        save_json(directory/'summary.json',summary,environ)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    add_dataset_arguments(parser)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--output',type=Path,default=Path('outputs/evaluation'))
    parser.add_argument('--metric',choices=['accuracy','process','both'],default='both')
    parser.add_argument('--failure-policy',choices=['error','zero'],default='error')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(argv)
    try:
        profile=load_profile(args.config,os.environ,offline=args.dry_run)
        tasks=load_tasks(**dataset_arguments(args,include_images=False))
        if args.dry_run:
            for task in tasks:
                for key in METRICS.values(): build_judge_messages(task,'OFFLINE_VALIDATION',key)
            print(f'Validated {len(tasks)} tasks and {2*len(tasks)} rubrics; no judge requests.')
            return 0
        if args.predictions is None: raise ValueError('--predictions is required for scoring')
        summary=run_evaluation(tasks,read_predictions(args.predictions),profile,args.output,environ=os.environ,
                               metric=args.metric,failure_policy=args.failure_policy,resume=args.resume)
        print(json.dumps(summary,indent=2))
        keys=['accuracy','process_score'] if args.metric=='both' else ['process_score' if args.metric=='process' else 'accuracy']
        return 0 if all(summary[k] is not None for k in keys) else 1
    except (ValueError,OSError,CompletionError) as exc:
        print(f'Cannot evaluate: {scrub(str(exc),os.environ)}')
        return 2


if __name__=='__main__': raise SystemExit(main())
