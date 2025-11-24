# envy

RL environment for teaching models data cleaning + ML engineering via corrupted tabular data.

## quick start

```bash
# clone repo
git clone https://github.com/yaitso/envy.git
cd envy

# create venv
uv venv

# install dependencies
uv sync

# build sandbox
envy build

# generate dataset (recommended params for ~20% haiku pass rate)
envy dataset generate \
  --complexity 300 \
  --num-classes 5 \
  --corrupt 0.5 \
  --num-rows 500 \
  --num-cols 3 \
  --used 1.0

# establish baseline performance on uncorrupted golden.csv
envy eval --expert \
  --model claude-haiku-4-5
  --dataset synth-XXXXXX #from above cmd's output
  --num-runs 1 # only 1 rollout for expert baseline
  # rest of params identical to agent eval below
  --max-steps 100 \
  --max-tokens 3000 \
  --max-eval-uses 20

# run agent eval (recommended params for ~20% haiku pass rate)
envy eval \
  --model claude-haiku-4-5 \
  --dataset synth-XXXXXX #from above cmd's output
  # same params as expert eval above
  --num-runs 10 \
  --max-steps 100 \
  --max-tokens 3000 \
  --max-eval-uses 20

envy eval --slow # this flag is optional it allows agents to use optuna hyperparameter tuning which is slower but does lead to better accuracy
```


## motivation

take-home asked:
> The task should resemble the kinds of things an ML engineer or ML researcher might do

pre-LLM ML was simple: wrangle tables, stack sklearn ensembles, ship. this captures that workflow.

### the platonic data problem

in some platonic world your dataset has clean dynamics: `f(x₁, ..., xₙ) → y` perfectly determines labels. your job is approximating that function.

reality: corruption everywhere.
- measurement errors (sensor drift, quantization)
- manual entry typos (fat-fingered "1000" → "100")
- subtle bugs (timezone conversions, encoding mishaps)
- cosmic rays flipping bits ([yes, actually](https://en.wikipedia.org/wiki/Single-event_upset))

**skill tested:** can agent build preprocessing pipeline that ameliorates corruption + brings data closer to platonic form? then train good model on cleaned data?

### dataset generation pipeline

1. **sample base table:** float/int columns from realistic distributions (not uniform noise — uses spec-driven variety)

2. **define platonic dynamics:** LLM generates python function mapping features → labels:
   ```python
   def dynamics(row):
       term1 = row.x0**2 + row.x1**3 - row.x2*row.x3
       term2 = np.sin(row.x0*row.x1) + np.cos(row.x2)
       if row.x0 > 0:
           return (term1 + term2) % num_classes
       else:
           return int(abs(term1 - term2)) % num_classes
   ```

3. **corrupt training split:**
   - gaussian noise on floats
   - random offsets on ints
   - occasional outliers (5σ+ jumps)
   - controlled by `--corrupt` flag

4. **output three CSVs:**
   - `golden.csv` — uncorrupted train (platonic baseline)
   - `train.csv` — corrupted version (agent sees this)
   - `val.csv` — uncorrupted validation (hidden until eval)

5. **establish expert baseline:** run model on `golden.csv` → measures best achievable score if agent perfectly cleans data

6. **agent eval:** N rollouts where agent must explore `train.csv`, write `clean_data()` pipeline, train model, generalize to hidden `val.csv`

### difficulty knobs

target: 10–40% pass rate. tune via:
- `--complexity` 150–300 (AST node count in dynamics function)
- `--num-classes` 5–10 (label cardinality)
- `--corrupt` 0.1–0.5 (fraction of rows corrupted)
- `--num-rows` 500–2000
- `--num-cols` 3–10

too easy (>40%)? bump complexity + corruption.
too hard (<10%)? dial back.

## timeline

**hands-on work:** ~4 hours (dataset generation, sandbox setup, eval loop, reward hack detection).
**autonomous tuning:** 1–2 hours of sonnet exploring param space to hit 10–40% spec.

## architecture

see `CLAUDE.md` for detailed architecture notes.

**key files:**
- `envy.py` — CLI interface
- `main.py` — agent loop + evaluation logic
- `generate.py` — synthetic dataset generation
- `sandbox.Dockerfile` — agent sandbox image

**design principles:**
- reproducible-ish
- strict sandbox isolation prevents reward hacking
- hidden validation set (agents never see val.csv during training)
- reward hack detection via haiku code review before eval
- concurrent execution for fast iteration (enabled by default)
- limited number of `eval_model` tool calls to prevent kaggle-style hillclimbing against unknown val set

## what i got wrong

**variance source:** letting agent control both data cleaning AND model architecture creates too many degrees of freedom. should've fixed `train_model()` (e.g., always RandomForestClassifier with specific params) to isolate data engineering skill.

**actual test:** "can you undo corruption to recover platonic data?" — not "can you also pick optimal model?" fixing model would've isolated the core skill better.

**future work:**
- constrain model architecture (remove variance)
- time series corruption (drift, seasonality artifacts)
- multi-table joins with referential integrity bugs