# Inference and evaluation

The public package contains the Direct provider adapter and the native ReAct loop and evidence tools used in the experiments, with portable command-line wrappers. Direct and ReAct both receive the initial task images. Evaluation uses the released per-task rubrics. The package does not include DeerFlow, visual-access ablations, internal gateways, model-specific batch drivers, or historical retry/accounting overlays.

## Install

Use Python 3.10 or newer on Linux or macOS. Windows users can use WSL; the tool cache and run-directory locks use `fcntl`.

```bash
git clone https://github.com/antins-labs/VisSearch-Bench-v1.git
cd VisSearch-Bench-v1
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

## Configure model access

`configs/openai.json` supports OpenAI-compatible Chat Completions endpoints. ReAct uses the native Responses API for model IDs recognized by the existing adapter as GPT-5.5; other OpenAI-compatible models use Chat Completions with tool calling. `configs/anthropic.json` supports Anthropic Messages. Choose a multimodal model; ReAct also requires compatible tool calling.

```bash
export MODEL_NAME='your-model-id'
export MODEL_BASE_URL='https://your-provider.example/v1'
export MODEL_API_KEY='your-api-key'
```

If `MODEL_BASE_URL` is unset, the profile uses the provider's official endpoint. For Anthropic, use the provider's base URL without an OpenAI `/v1` suffix unless your gateway documents otherwise. Model IDs and endpoints must match your service. `temperature: null` omits that request parameter; optional `reasoning_effort` and Qwen `disable_thinking` settings are supported by the inherited adapters. These generic profiles do not recreate every model-specific setting in the paper.

Store credentials in environment variables. The repository contains an empty `.env.example`; commands do not automatically load `.env`. If you maintain your own shell-compatible `.env`, load it before running:

```bash
set -a
source .env
set +a
```

Local `.env` files, outputs, caches, and virtual environments are ignored by Git. Configuration accepts `api_key_env`, never an inline `api_key`. Saved outputs redact configured secret values and image data URLs; SDK failures are reduced to safe metadata.

## Dataset

Both commands default to the public `test` split of `antins-labs/VisSearch-Bench-v1`, pinned to revision `a01927d6a0aa03552b607fd1f8b55558b2a84834`. The first run downloads the dataset through Hugging Face; no Hugging Face token is required for this public release. Initial embedded images are materialized in `.cache/vissearch_bench/images/`.

You can also pass `--dataset /path/to/tasks.jsonl` or a local Parquet file. JSONL records use the released fields (`id`, `question_en`, `answer`, `accuracy_prompt`, and `Process Score`); `images` can be a list of local image paths, resolved relative to the JSONL file. The local export's `image_paths` field is also supported. Text-initiated tasks use an empty image list. Parquet/Hub records may contain embedded images. A Hub id other than the default may need its own `--revision` and `--split`.

`--ids 168,15` selects tasks while preserving dataset order. `--limit 2` runs only the first two selected tasks; omit it for all 204. Use the same selection for inference and evaluation.

## Direct inference

Validate dataset access and image preparation without any model or search request:

```bash
python -m vissearch_bench.inference \
  --method direct --config configs/openai.json --limit 2 --dry-run
```

Generate answers:

```bash
python -m vissearch_bench.inference \
  --method direct --config configs/openai.json \
  --output outputs/direct
```

Only the question and initial images enter the model request. Reference answers, reference solutions, and grading prompts are excluded. Direct has no search tools. The provider adapter makes one API attempt per call; SDK retries are disabled.

## ReAct inference

Configure the two evidence services in addition to the model:

```bash
export SERPER_API_KEY='your-serper-key'
export SERPAPI_KEY='your-serpapi-key'

python -m vissearch_bench.inference \
  --method react --config configs/openai.json \
  --output outputs/react
```

The five tools are `web_search`, `open_page`, `image_search`, `crop_image`, and `reverse_image_search`. Serper provides web/image search and page scraping; SerpApi Google Lens provides reverse image search, including image upload. Source images and crops used for reverse search are sent to that configured service. Python execution is not available to the agent.

Default budgets match the extracted ReAct runtime: 30 main model turns plus at most one forced final submission, 60 evidence-tool actions, and at most four evidence-tool calls per turn. Search tools return up to five results; pages are limited to 12,000 characters; the image store has at most 20 images; input/display images have maximum side length 1,536 pixels. A forced final submission may be an additional model request after a budget is reached.

Change the public loop limits with `--max-turns`, `--max-total-tool-calls`, `--max-tool-calls-per-turn`, and `--image-max-side`. Changing these settings changes the run fingerprint. Successful evidence responses are cached locally. Tool failures can be returned as observations when recoverable; authentication failures terminate the task. No automatic paid retry is added by these wrappers.

Evidence tools can use explicit `VISBENCH_TOOLS_HTTP_PROXY` / `VISBENCH_TOOLS_HTTPS_PROXY` variables, with conventional proxy variables as fallbacks. Direct, ReAct, and judge model transports are isolated from ambient evidence-tool proxies. See the source for the retained public-address, redirect, image-size, and download-budget checks.

## Accuracy and Process Score

The judge can be a different model from the answering model:

```bash
export JUDGE_MODEL_NAME='your-judge-model-id'
export JUDGE_BASE_URL='https://your-judge-provider.example/v1'
export JUDGE_API_KEY='your-judge-api-key'

python -m vissearch_bench.evaluate \
  --config configs/judge.json \
  --predictions outputs/react/predictions.jsonl \
  --output outputs/react-evaluation
```

For Direct answers, point `--predictions` to `outputs/direct/predictions.jsonl`. To use an Anthropic judge, create a profile with `provider: "anthropic"` and the judge environment-variable names.

- **Accuracy:** reads `accuracy_prompt`; requires a binary 0 or 1.
- **Process Score:** reads `Process Score`; accepts a score between 0 and 1 according to the task's rubric. Tasks with the same rubric as Accuracy also require binary output.
- The two metrics use independent judge calls. The parser requires the final nonempty line to be exactly a JSON object such as `{"score": 0.5}`; malformed, duplicate, nonfinite, or out-of-range results are rejected.
- Judge inputs are the question, reference answer, and model's final answer. The full agent tool trace and private reasoning are not scored step by step. Each task's released rubric defines its partial-credit rules.

Run one metric with `--metric accuracy` or `--metric process`; the default is `--metric both`. Validate all rubrics without judge requests:

```bash
python -m vissearch_bench.evaluate --config configs/judge.json --dry-run
```

External answers can be supplied as one JSON object per line:

```json
{"id":168,"status":"ok","answer":"The model's final answer."}
```

Ids must be unique and match the selected dataset tasks. A failed prediction can use `{"id":168,"status":"error"}`. Missing predictions are rejected by default. Failed answers or judge responses are recorded explicitly; aggregate metrics remain `null` if the selected population is not fully scored. `*_coverage` and `*_judged` distinguish accounted and actually judged records.

To explicitly count missing/failed cases as zero, pass `--failure-policy zero`. This uses whole-task accounting: a failed selected metric makes all selected metrics for that task zero; the original scores remain in `raw_score`. Imputed zeros are counted in `num_zeroed`. This policy does not silently drop failed cases or claim they were successfully judged.

The paper's aggregate tables also depend on specific model versions, model settings, tool availability, and adjudicated historical runs. These commands expose the underlying generation and rubric-scoring implementations; generic example profiles are not a promise of identical scores.

## Outputs and continuation

Inference writes `predictions.jsonl`, per-task `records/`, and ReAct `traces/`. Evaluation writes `scores.jsonl`, per-task `records/`, and `summary.json`. Metrics in `summary.json` are available as fractions and percentages.

Use `--resume` with the same command and output directory. Existing saved stages, including errors, are retained; successful calls are not repeated. Evaluation can continue after a completed Accuracy stage without repeating it. Dataset content/selection, model profile, settings, source code, and evaluation inputs are fingerprinted; changes require a new output directory. Secrets are excluded from these fingerprints. Only one process may own an output directory.

An interrupted call that has not produced a saved result can be repeated when resumed. Inspect provider usage before resuming if the process stopped during a paid request. To retry a saved error deliberately, select those task ids and use a new output directory; combine external predictions explicitly by id before a full evaluation.

Exit codes: `0` for completed/accounted results or a successful dry run, `1` for recorded task failures or incomplete scoring, and `2` for configuration/input errors.

## Offline tests

```bash
python -m pytest -q
```

Tests use synthetic tasks, local images, and fake model/search clients. They exercise provider wire formats, image/tool boundaries, budgets, cache locking, rubric parsing, input isolation, redaction, and public run behavior without paid API calls.
