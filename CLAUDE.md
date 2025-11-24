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

## working with background tasks

when running long commands in background (dataset generation, eval runs), use conservative sleep timings:

- **dataset generation:** start with `sleep 8`, check output, then use longer sleeps (32-64s) between checks
- **eval runs:** start with `sleep 8`, check progress, then use exponential backoff (32s, 64s, 128s)
- **general rule:** prefer shorter initial sleeps to catch errors early, then ramp up

## running evaluations

CRITICAL: NEVER specify --model flag when running evals. the default model in envy.py is deliberately chosen and should not be overridden unless user explicitly requests it.

correct:
```bash
uv run python envy.py eval --debug --sequential --num-runs 1
```

wrong:
```bash
uv run python envy.py eval --model claude-haiku-4 --debug --sequential --num-runs 1
```

## meta goal: achieving 10-40% pass rate

the framework's target is to achieve a 10-40% pass rate on agent evaluations. this ensures the task is challenging but achievable.

### acceptance criteria

- **pass rate range:** 10-40% (1-4 successful runs out of 10)
- **evaluation model:** default model from envy.py (claude-haiku-4-5 as of 2024-11)
- **success definition:** agent score within ±5% of expert baseline (corrupt_score)

### workflow for establishing new dataset

1. **generate dataset** with minimal params for rapid iteration:
   ```bash
   uv run python envy.py dataset generate \
     --complexity 150 \
     --num-classes 5 \
     --corrupt 0.2 \
     --num-rows 500 \
     --num-cols 3 \
     --used 1.0 \
     --seed 42
   ```
   - 500 rows (350 train, 150 val) for fast execution
   - 3 features only (simpler models)
   - 20% corruption for difficulty
   - used=1.0 means all features participate in dynamics function

2. **establish expert baseline** (pretend to be expert ML engineer):
   - follow instructions in `BASELINE.md`
   - user will prompt: "please do @BASELINE.md on synth-XXXXXX"
   - create `dataset/synth-XXXXXX/corrupt_expert.py` (NO OPTUNA, rapid iteration only)
   - use good defaults: RandomForest n_estimators=100, max_depth=15
   - run expert script to populate `corrupt_score` in `meta.json`
   - expert baseline represents best achievable score with manual tuning
   - target runtime: <30 seconds total

3. **run agent evaluation** (3 runs concurrent, no sequential flag):
   ```bash
   # run eval (no pipes, no tmux, concurrent by default)
   uv run python envy.py eval --num-runs 3 --debug
   ```
   - 3 runs concurrent (faster than sequential)
   - user will pbcopy interesting parts themselves
   - no --sequential flag (slow)

4. **analyze pass rate:**
   - count successful runs (score within ±5% of corrupt_score)
   - calculate pass_rate = successes / 3 (quick feedback)
   - if pass_rate < 10%: dataset too hard → reduce complexity or corruption
   - if pass_rate > 40%: dataset too easy → increase complexity or corruption
   - for final validation, scale to --num-runs 10

5. **adjust dataset params if needed:**
   - **too easy (>40%):** increase `--complexity`, `--corrupt`, or `--num-classes`
   - **too hard (<10%):** decrease `--complexity`, `--corrupt`, or `--num-classes`
   - regenerate dataset and repeat from step 2

### typical param ranges for difficulty tuning

- **easier (40%+ pass rate):**
  ```bash
  --complexity 180 --num-classes 5 --corrupt 0.10
  ```

- **medium (20-30% pass rate):**
  ```bash
  --complexity 250 --num-classes 8 --corrupt 0.15
  ```

- **harder (10-20% pass rate):**
  ```bash
  --complexity 300 --num-classes 10 --corrupt 0.20
  ```
