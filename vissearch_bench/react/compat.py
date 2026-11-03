"""Four unchanged compatibility helpers copied from the original evaluator."""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

def answer_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)

def resolve_local_image(path_or_url: str, base_dir: Path) -> Optional[Path]:
    if re.match(r"^https?://", path_or_url):
        return None
    path = Path(path_or_url)
    if not path.is_absolute():
        path = base_dir / path
    resolved = path.resolve()
    allowed_root = base_dir.resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError(
            "local image path escapes the data directory: {}".format(
                path_or_url
            )
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError("image not found: {}".format(resolved))
    return resolved

def normalize_component_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(decoded, list):
            value = decoded
        else:
            return [stripped]
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, (list, tuple, set)):
        value = [value]

    components = []
    seen = set()
    for item in value:
        text = answer_to_text(item)
        if not text:
            continue
        identity = text.casefold()
        if identity in seen:
            continue
        seen.add(identity)
        components.append(text)
    return components

def openai_token_limit_kwargs(
    model: str,
    max_tokens: int,
) -> Dict[str, int]:
    if model.strip().casefold().startswith("gpt-5.5"):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}
