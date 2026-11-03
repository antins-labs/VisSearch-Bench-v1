"""Load the public Hub release or a local JSONL/Parquet file."""
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

DATASET_ID = 'antins-labs/VisSearch-Bench-v1'
DATASET_REVISION = 'a01927d6a0aa03552b607fd1f8b55558b2a84834'


def normalize_task(row, *, base_dir, cache_dir, include_images=True):
    row = dict(row)
    if type(row.get('id')) is not int or row['id'] < 1:
        raise ValueError('Task id must be a positive integer')
    for key in ('question_en', 'answer', 'accuracy_prompt', 'Process Score'):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f'Task {row["id"]}: missing {key}')
    for key in ('accuracy_prompt', 'Process Score'):
        if any('{{'+name+'}}' not in row[key] for name in ('question','reference_answer','model_answer')):
            raise ValueError(f'Task {row["id"]}: invalid rubric placeholders')
    images = row.get('image_paths', row.get('images', []))
    if not isinstance(images, (list, tuple)):
        raise ValueError('Task images must be a list')
    paths = []
    if include_images:
        for item in images:
            image = None
            if isinstance(item, Image.Image):
                image = item
            elif isinstance(item, dict) and item.get('bytes') is not None:
                image = Image.open(io.BytesIO(item['bytes']))
            else:
                value = item.get('path') if isinstance(item, dict) else item
                if not isinstance(value, str) or not value or '://' in value:
                    raise ValueError('Use embedded images or local image paths')
                path = Path(value)
                path = path if path.is_absolute() else Path(base_dir) / path
                if not path.is_file():
                    raise ValueError(f'Task {row["id"]}: input image is missing')
                with Image.open(path) as probe:
                    probe.verify()
                paths.append(path.resolve())
            if image is not None:
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                payload = buffer.getvalue()
                path = Path(cache_dir) / 'images' / (hashlib.sha256(payload).hexdigest()+'.png')
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists(): path.write_bytes(payload)
                paths.append(path.resolve())
    row['_image_paths'] = paths
    # Embedded PIL objects/bytes are runtime inputs, never JSON output fields.
    row.pop('image_paths', None)
    row['images'] = [str(p) for p in paths]
    return row


def load_tasks(source=DATASET_ID, *, split='test', revision=DATASET_REVISION,
               cache_dir=Path('.cache/vissearch_bench'), ids=None, limit=None, include_images=True):
    source = str(source)
    path = Path(source).expanduser()
    if path.is_file() and path.suffix.lower() == '.jsonl':
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        base_dir = path.resolve().parent
    else:
        from datasets import load_dataset
        if path.is_file():
            rows = load_dataset('parquet', data_files=str(path), split='train')
            base_dir = path.resolve().parent
        else:
            rows = load_dataset(source, split=split, revision=revision)
            base_dir = Path.cwd()
    seen, selected = set(), []
    requested = set(ids) if ids is not None else None
    # Read ids before decoding images, retaining release order.
    all_ids = rows['id'] if not isinstance(rows, list) else [r.get('id') for r in rows]
    for task_id in all_ids:
        if type(task_id) is not int or task_id < 1: raise ValueError('Invalid task id')
        if task_id in seen: raise ValueError('Duplicate task id')
        seen.add(task_id)
    if requested is not None and not requested <= seen:
        raise ValueError('Requested task ids are absent from the dataset')
    if limit is not None and limit < 1: raise ValueError('Limit must be positive')
    indices = [i for i, task_id in enumerate(all_ids) if requested is None or task_id in requested]
    if limit is not None: indices = indices[:limit]
    if not indices: raise ValueError('No tasks selected')
    if not include_images and not isinstance(rows, list):
        rows = rows.remove_columns([key for key in ('images','image_paths') if key in rows.column_names])
    for i in indices:
        selected.append(normalize_task(rows[i], base_dir=base_dir, cache_dir=cache_dir, include_images=include_images))
    return selected
