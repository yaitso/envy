# envy architecture

## overview

envy is an LLM evaluation framework for supervised learning tasks. it runs agents in sandboxed docker containers, measures their ability to build ML models, and validates results against hidden test sets.

## core components

### 1. agent loop (`main.py:run_agent_loop`)

- spins up isolated docker sandbox per rollout
- gives agent two tools: `run_bash_cmd` and `submit_task_completion`
- enforces max_steps and max_tokens limits
- automatically kills sandbox when done or on error

### 2. evaluation pipeline (`main.py:eval_model`)

- uses claude haiku to detect reward hacking in eval.py (hardcoding, sandbox escape, accessing train data)
- if clean, spins up separate eval sandbox with:
  - agent's `/results` mounted read-only
  - `/dataset/val` mounted (hidden during training)
- executes `uv run /results/eval.py` and parses final score from stdout
- kills eval sandbox when done

### 3. dataset structure

```
dataset/
├── meta.json          # expected_score, score_tolerance, description
├── train/
│   └── train.csv      # agent sees this during training
└── val/
    └── val.csv        # hidden during training, used only in eval
```

### 4. sandbox isolation

**training sandbox:**
- `docker run -v dataset/train:/dataset -v results/{rollout_id}:/results envy`
- agent writes train.py, model.pkl, eval.py to /results/

**eval sandbox:**
- `docker run -v results/{rollout_id}:/results:ro -v dataset/val:/dataset envy`
- read-only results mount prevents tampering
- separate container ID prevents cross-contamination

### 5. prompt template (`main.py:PROMPT_TEMPLATE`)

agent must:
1. explore `/dataset/train.csv`
2. create `/results/train.py` with `clean_data()` + `train_model()`
3. create `/results/eval.py` with EXACT same `clean_data()` but loads `/dataset/val.csv`
4. iteratively refine model to maximize accuracy
5. call `submit_task_completion` when done

**critical constraint:** eval.py clean_data() must match train.py exactly (except file path), ensuring same preprocessing on train and val splits.

## current limitations

- **no iterative refinement:** agent currently can't check val score during training (would be reward hacking)
- **blind optimization:** agent tunes on train.csv without feedback loop

## improvement plan

add `eval_model_on_train` tool that:
- runs eval.py against train.csv (not val.csv)
- gives agent feedback on current model quality
- enables multi-step refinement loop: train → eval on train → adjust → repeat

this preserves sandbox isolation while enabling iterative improvement.

## datasets available

### xor (`envy dataset xor`)
- trivial binary classification
- 3 train samples, 1 val sample
- expected: 100% accuracy

### mnist1d (`envy dataset mnist1d`)
- 10-class 1d signal classification
- benchmark: logistic 32%, mlp 68%, cnn 94%
- expected: 65% ± 20%

## file structure

```
envy/
├── envy.py              # CLI interface (click commands)
├── main.py              # core agent loop + eval logic
├── pyproject.toml       # deps: anthropic, click, result, xxhash
├── sandbox.Dockerfile   # uv + sklearn/pandas/numpy/scipy
├── sandbox.toml         # sandbox pyproject.toml
├── dataset/             # gitignored, generated via CLI
└── results/             # gitignored, created during runs
    └── {run_id}/
        └── {rollout_id}/
            ├── train.py
            ├── model.pkl
            └── eval.py
```

## execution flow

```
envy eval --num-runs N --max-steps S --max-tokens T
  ↓
for each run:
  run_id = xxhash(uuid4())[:6]
  rollout_id = xxhash(uuid4())[:6]

  start_sandbox(run_id, rollout_id)
    ↓
  run_agent_loop(PROMPT_TEMPLATE, tools=[run_bash_cmd, submit_task_completion])
    ↓ (agent writes train.py, model.pkl, eval.py)
  kill_sandbox(run_id, rollout_id)
    ↓
  eval_model(run_id, rollout_id)
    ↓ (reward hack check → eval sandbox → parse score)
  compare score to expected_score ± tolerance
    ↓
  ✓ or ✗
```

## design philosophy

- **no cheating:** strict sandbox isolation + reward hack detection
- **realistic constraints:** agent doesn't see val.csv until after submission
- **reproducible:** xxhash-based IDs for deterministic tracking
- **scalable:** concurrent runs via asyncio.as_completed
- **simple:** minimal deps (anthropic SDK + click + result monad)
