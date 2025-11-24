# envy

LLM evaluation framework for supervised learning tasks. runs agents in sandboxed docker containers and measures their ability to build ML models.

## setup

1. install dependencies:
   ```bash
   uv venv && uv sync
   ```

2. set anthropic api key:
   ```bash
   export ANTHROPIC_API_KEY=your_key_here
   ```

3. build sandbox image:
   ```bash
   envy sandbox
   ```

4. generate dataset (choose one):
   ```bash
   envy dataset xor      # trivial binary classification (100% expected)
   envy dataset mnist1d  # 10-class benchmark (65% ± 20% expected)
   ```

## usage

run evaluation with custom parameters:
```bash
envy eval --num-runs 10 --max-steps 20 --max-tokens 1000
```

run sequentially (easier to debug):
```bash
envy eval --num-runs 1 --sequential --debug
```

## how it works

1. **agent training phase:**
   - spins up docker sandbox with `/dataset/train` mounted
   - agent explores data, creates `train.py` (cleaning + model training)
   - agent creates `eval.py` (same cleaning logic, loads model, computes accuracy)
   - agent submits completion

2. **evaluation phase:**
   - checks eval.py for reward hacking (hardcoding, sandbox escape, etc)
   - spins up separate sandbox with agent's `/results` mounted read-only
   - mounts hidden `/dataset/val` (agent never saw this during training)
   - runs `uv run /results/eval.py` and parses accuracy from stdout
   - compares to expected score ± tolerance

## architecture

- **sandbox isolation:** separate docker containers for train/eval prevent cheating
- **reward hack detection:** claude haiku validates eval.py before running
- **deterministic tracking:** xxhash-based run/rollout IDs
- **concurrent execution:** asyncio for parallel runs (or `--sequential` for debugging)

see `CLAUDE.md` for detailed architecture docs.
