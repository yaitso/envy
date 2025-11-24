ok here's comprehensive list of float corruption patterns organized by category:

  missing data patterns

  - np.nan injection (random % of values)
  - sentinel values (-999, -1, 0, 9999)
  - string placeholders parsed as NaN ("N/A", "NULL", "missing", ".", "-", "")
  - conditional missingness (MCAR, MAR, MNAR patterns)

  outliers & extreme values

  - inject values at Z-score > 3, 5, 10
  - fat tail samples (draw from t-distribution instead of normal)
  - multiplicative outliers (value × 10, 100, 1000)
  - min/max violations (exceed expected range)
  - single massive spike (max × 10000)

  noise injection

  - additive gaussian ~ N(0, σ) where σ ∝ std(column)
  - multiplicative noise (value × uniform(0.8, 1.2))
  - uniform noise ± epsilon
  - salt & pepper (random extreme pos/neg values)
  - heteroskedastic noise (variance increases with value magnitude)

  precision & rounding

  - round to N decimals (0, 1, 2)
  - floor/ceil/truncate
  - quantization (round to nearest 0.5, 0.25, etc)
  - float32 ↔ float64 conversion artifacts
  - integer coercion (12.7 → 12)

  scale & unit errors

  - off by 10^k (k ∈ {1,2,3,-1,-2,-3})
  - mixed units (meters + centimeters)
  - percentage as decimal vs whole (0.15 vs 15)
  - currency scale confusion (dollars vs cents)
  - temperature scale mixing (celsius vs fahrenheit numeric values)

  sign errors

  - random sign flips (~10% of values)
  - abs() everything
  - negate all values
  - conditional sign errors (only flip if value < threshold)

  zero inflation

  - replace NaN → 0
  - replace outliers → 0
  - replace negatives → 0
  - inject random zeros (~10% of values)
  - trailing zeros from precision loss

  clipping & capping

  - winsorization (cap at 5th/95th percentile)
  - hard bounds (min=0, max=100)
  - IQR-based clipping
  - truncate at arbitrary threshold
  - saturating nonlinearity (tanh-like compression)

  type coercion artifacts

  - comma thousands separator breaks parsing ("1,234.56" → NaN)
  - boolean → float (True=1.0, False=0.0, then mixed with reals)
  - date serialization (unix timestamp, excel date numeric)
  - scientific notation errors (1.23e-5 parsed wrong)
  - hex string → float attempts

  duplicates & repeats

  - forward fill (propagate last valid value)
  - mode imputation (replace with most common)
  - stuck sensor (same value repeated N consecutive times)
  - backward fill
  - value echo (duplicate random subset of rows)

  binning & discretization

  - round to bins then convert back (0-10 → 5.0, 10-20 → 15.0)
  - quantile binning artifact
  - age heaping (cluster around multiples of 5, 10)
  - floating point bin edges causing boundary errors

  swaps & permutations

  - shuffle within column (destroys row relationships)
  - swap with another column's values
  - random pairwise swaps (~5% of rows)
  - block permutation (shuffle chunks of size N)

  drift & temporal corruption

  - linear trend injection (value += slope × row_index)
  - regime change (mean shift at random row)
  - seasonal pattern overlay (sine wave)
  - variance drift (std increases over rows)
  - concept drift (relationship with target changes)

  conditional corruption

  - corrupt only if value > p90
  - corrupt only if another_column == category
  - magnitude-proportional noise (larger values → larger noise)
  - correlation-breaking (shuffle only high-correlation features)

  real-world domain patterns

  - GPS errors (lat/lon swap, decimal point shift: 37.7749 → 3.77749)
  - financial bid-ask bounce (alternating ±0.01)
  - sensor calibration drift (systematic additive offset increasing over time)
  - human entry errors:
    - digit transposition (12.34 → 21.43, 13.24)
    - decimal point shift (123.4 → 12.34, 1.234)
    - copy-paste column misalignment
    - leading/trailing zero errors (0.5 → 5.0)

  combo attacks

  - outlier → NaN (double corruption)
  - scale error → noise (off by 100× then add gaussian)
  - missingness → forward fill → noise (cascade)

  implementation knobs for each:
  - severity: % of values affected (5%, 10%, 25%, 50%)
  - concentration: uniform random vs clustered (consecutive rows, specific quantiles)
  - multiplicity: single corruption type vs stack 2-3 simultaneously

  start simple with {missing, outliers, noise, scale errors} then expand?

> what's the best way i can generate cols with some specified dist numpy? scipy? which will be common irl maybe some fat tailed shit too

⏺ use scipy.stats — way more distributions + consistent API + parameterization is cleaner.

  common real-world distributions

  from scipy import stats
  import numpy as np

  # basic workhorse distributions
  normal = stats.norm(loc=0, scale=1).rvs(1000)
  lognormal = stats.lognorm(s=0.5, scale=np.exp(5)).rvs(1000)  # prices, incomes
  exponential = stats.expon(scale=2).rvs(1000)  # wait times
  uniform = stats.uniform(loc=0, scale=10).rvs(1000)

  # bounded [0,1] — percentages, probabilities, rates
  beta = stats.beta(a=2, b=5).rvs(1000)

  # right-skewed positive
  gamma = stats.gamma(a=2, scale=2).rvs(1000)
  weibull = stats.weibull_min(c=1.5, scale=1).rvs(1000)
  chi2 = stats.chi2(df=3).rvs(1000)

  # FAT TAILS (what you want)
  student_t = stats.t(df=3).rvs(1000)  # df=3 is thicc, df=1 is cauchy
  pareto = stats.pareto(b=1.5).rvs(1000)  # power law, b<2 has infinite variance
  cauchy = stats.cauchy(loc=0, scale=1).rvs(1000)  #极其 fat, no mean/var
  levy = stats.levy(loc=0, scale=1).rvs(1000)  # stable distribution

  practical schema generator pattern

  DISTRIBUTIONS = {
      'normal': lambda: stats.norm(
          loc=np.random.uniform(-10, 10),
          scale=np.random.uniform(0.1, 5)
      ),
      'lognormal': lambda: stats.lognorm(
          s=np.random.uniform(0.1, 1.5),
          scale=np.exp(np.random.uniform(0, 10))
      ),
      'exponential': lambda: stats.expon(
          scale=np.random.uniform(0.5, 5)
      ),
      'gamma': lambda: stats.gamma(
          a=np.random.uniform(1, 5),
          scale=np.random.uniform(0.5, 3)
      ),
      'beta': lambda: stats.beta(
          a=np.random.uniform(0.5, 5),
          b=np.random.uniform(0.5, 5)
      ),
      'student_t': lambda: stats.t(
          df=np.random.choice([2, 3, 5, 10])  # lower = fatter
      ),
      'pareto': lambda: stats.pareto(
          b=np.random.uniform(1.2, 3)  # 1<b<2 is crazy fat
      ),
  }

  def generate_column(n_rows, dist_name):
      dist = DISTRIBUTIONS[dist_name]()
      return dist.rvs(n_rows)

  recommended mix for ML tasks

  start with:
  1. normal (50%) — baseline, most algos assume this
  2. lognormal (20%) — common in business (revenue, price, size)
  3. exponential (10%) — durations, wait times
  4. student_t df=3 (10%) — fat tails but learnable
  5. gamma (10%) — positive skewed

  then add pareto/cauchy for hard mode (agents will suffer with outliers).

  why scipy > numpy

  # numpy: inconsistent APIs
  np.random.normal(loc, scale, size)
  np.random.exponential(scale, size)  # no loc param
  np.random.gamma(shape, scale, size)  # different param names

  # scipy: uniform .rvs(size) everywhere
  stats.norm(loc, scale).rvs(size)
  stats.expon(scale).rvs(size)
  stats.gamma(a, scale).rvs(size)

  # plus you get pdf/cdf/ppf for validation
  dist = stats.norm(0, 1)
  dist.pdf(x)     # probability density
  dist.cdf(x)     # cumulative
  dist.ppf(0.95)  # inverse cdf (quantile)

  scipy also has stats.fit() if you wanna reverse-engineer params from real data later.

  go with scipy unless you're trying to minimize deps (but you already have sklearn which pulls scipy anyway).

> what about ints?

⏺ integer distributions

  scipy.stats mostly returns floats, so you'll round/cast. here's the practical approach:

  from scipy import stats
  import numpy as np

  # discrete distributions (native integers)
  poisson = stats.poisson(mu=5).rvs(1000)           # counts, events
  binom = stats.binom(n=100, p=0.3).rvs(1000)       # trials, successes
  nbinom = stats.nbinom(n=10, p=0.5).rvs(1000)      # negative binomial
  geom = stats.geom(p=0.1).rvs(1000)                # waiting time (discrete)
  hypergeom = stats.hypergeom(M=100, n=30, N=10).rvs(1000)  # sampling w/o replacement

  # continuous → discretized (round)
  discrete_normal = np.round(stats.norm(50, 10).rvs(1000)).astype(int)
  discrete_lognorm = np.round(stats.lognorm(s=0.5, scale=100).rvs(1000)).astype(int)
  discrete_gamma = np.round(stats.gamma(a=2, scale=10).rvs(1000)).astype(int)

  # bounded discrete uniform
  randint = np.random.randint(0, 100, size=1000)    # [0, 100)
  discrete_uniform = stats.randint(low=0, high=100).rvs(1000)  # same but scipy

  # zipf (power law for discrete) — fat tailed integers
  zipf = stats.zipf(a=2).rvs(1000)  # a>1, lower a = fatter tail

  practical schema for real-world int columns

  INT_DISTRIBUTIONS = {
      # counts (non-negative)
      'poisson': lambda: stats.poisson(
          mu=np.random.uniform(1, 20)
      ),

      # discrete uniform (ids, categories as ints)
      'randint': lambda: stats.randint(
          low=0,
          high=np.random.choice([10, 100, 1000, 10000])
      ),

      # rounded normal (ages, scores)
      'discrete_normal': lambda: {
          'dist': stats.norm(
              loc=np.random.uniform(20, 100),
              scale=np.random.uniform(5, 30)
          ),
          'round': True,
          'clip': (0, None)  # no negative ages
      },

      # rounded lognormal (counts with right skew)
      'discrete_lognorm': lambda: {
          'dist': stats.lognorm(
              s=np.random.uniform(0.3, 1.0),
              scale=np.exp(np.random.uniform(2, 6))
          ),
          'round': True,
          'clip': (1, None)  # strictly positive
      },

      # power law (viral content, citations, popularity)
      'zipf': lambda: stats.zipf(
          a=np.random.uniform(1.5, 3)  # a=2 is classic zipf
      ),

      # binomial (conversions, successes out of N trials)
      'binomial': lambda: stats.binom(
          n=np.random.choice([10, 50, 100, 1000]),
          p=np.random.uniform(0.01, 0.5)
      ),
  }

  def generate_int_column(n_rows, dist_name):
      config = INT_DISTRIBUTIONS[dist_name]()

      if isinstance(config, dict):
          # continuous → discrete
          vals = config['dist'].rvs(n_rows)
          if config.get('round'):
              vals = np.round(vals)
          if config.get('clip'):
              low, high = config['clip']
              vals = np.clip(vals, low, high if high else np.inf)
          return vals.astype(int)
      else:
          # native discrete
          return config.rvs(n_rows)

  common real-world int column types

  | domain     | distribution                             | example                   |
  |------------|------------------------------------------|---------------------------|
  | age        | discrete_normal(50, 15) clipped [0, 120] | customer age              |
  | count      | poisson(λ=5)                             | num_purchases, num_clicks |
  | id         | randint(0, 1M)                           | user_id, product_id       |
  | days       | geometric(p=0.1)                         | days_since_last_visit     |
  | quantity   | discrete_lognorm(σ=0.5) clipped [1, ∞)   | order_quantity            |
  | rank       | zipf(a=2) clipped [1, N]                 | popularity_rank           |
  | binary     | bernoulli(p=0.3)                         | is_converted (0/1)        |
  | percentage | binomial(n=100, p=0.3)                   | score out of 100          |

  corruption specifics for ints

  additional to float corruptions:

  # off-by-one errors
  corrupted = original + np.random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])

  # digit errors (swap, drop)
  # 123 → 132, 12, 1230
  def swap_digits(x):
      s = str(x)
      if len(s) < 2: return x
      i = np.random.randint(0, len(s)-1)
      return int(s[:i] + s[i+1] + s[i] + s[i+2:])

  # magnitude errors (extra zero, missing zero)
  # 100 → 1000 or 10
  corrupted = original * np.random.choice([10, 1, 0.1], p=[0.1, 0.8, 0.1])
  corrupted = corrupted.astype(int)

  # negative when should be positive
  corrupted = np.where(mask, -original, original)

  # sentinel values specific to ints
  corrupted[mask] = np.random.choice([-1, 0, 999, 9999])

  recommended starting set:
  1. poisson (30%) — counts are everywhere
  2. discrete_normal (30%) — ages, scores, bounded quantities
  3. randint (20%) — ids, categorical encoded as int
  4. discrete_lognorm (10%) — right-skewed counts
  5. zipf (10%) — hard mode fat tail

> can this all be done with fixed seed?

⏺ yeah absolutely — just seed both numpy and scipy upfront:
use RandomState for isolation
  # separate RNG per task instance
  rng = np.random.RandomState(seed=42)

  # pass to scipy distributions
  col1 = stats.norm(0, 1).rvs(1000, random_state=rng)
  col2 = stats.poisson(5).rvs(1000, random_state=rng)
  col3 = rng.randint(0, 100, 1000)

  # dynamics function sampling
  def sample_dynamics_params(rng):
      num_terms = rng.randint(2, 5)
      coeffs = rng.uniform(-10, 10, size=num_terms)
      return num_terms, coeffs

  even better: modern Generator API (numpy ≥1.17)

  from numpy.random import default_rng

  rng = default_rng(seed=42)

  # numpy ops
  col1 = rng.normal(0, 1, 1000)
  col2 = rng.integers(0, 100, 1000)
  col3 = rng.choice([1, 2, 3], size=1000, p=[0.5, 0.3, 0.2])

  # scipy compatibility
  col4 = stats.norm(0, 1).rvs(1000, random_state=rng)
  col5 = stats.poisson(5).rvs(1000, random_state=rng)

  deterministic task generation pattern

  def generate_task(task_id: int):
      rng = default_rng(seed=task_id)

      # 1. schema generation
      n_features = rng.integers(5, 20)
      dist_names = rng.choice(
          ['normal', 'lognormal', 'poisson', 'zipf'],
          size=n_features,
          replace=True
      )

      # 2. data generation
      df = {}
      for i, dist_name in enumerate(dist_names):
          df[f'f{i}'] = generate_column(
              n_rows=1000,
              dist_name=dist_name,
              rng=rng  # pass same rng
          )

      # 3. dynamics generation (deterministic from task_id)
      dynamics_code = generate_dynamics(
          n_features=n_features,
          target_ast_nodes=rng.integers(5, 20),
          rng=rng
      )

      # 4. corruption (deterministic)
      df_corrupted = corrupt_dataframe(df, severity=0.1, rng=rng)

      return df_corrupted, dynamics_code

  # same task_id → identical task
  task_a = generate_task(42)
  task_b = generate_task(42)
  assert task_a == task_b  # bit-for-bit identical

  gotcha: LLM generation breaks determinism

  if you're using anthropic API to generate dynamics function code:

  # NON-deterministic (API has no seed param)
  dynamics_code = await client.messages.create(
      model="claude-haiku-4-5",
      messages=[{"role": "user", "content": prompt}],
      # no seed parameter exists
  )

  # workaround: cache based on hash(prompt + task_id)
  cache_key = xxhash.xxh64(
      f"{task_id}:{prompt}".encode()
  ).hexdigest().encode()

  if cache_key in dynamics_cache:
      return dynamics_cache[cache_key]
  else:
      dynamics_code = await generate_via_llm(prompt)
      dynamics_cache[cache_key] = dynamics_code
      return dynamics_code

  or just pre-generate a fixed pool of dynamics functions offline, store in json, index by task_id deterministically:

  # offline: generate 1000 dynamics functions, save
  dynamics_pool = [generate_dynamics(seed=i) for i in range(1000)]
  json.dump(dynamics_pool, open('dynamics.json', 'w'))

  # online: deterministic lookup
  def get_dynamics(task_id):
      pool = json.load(open('dynamics.json'))
      return pool[task_id % len(pool)]

  for the test harness (10 runs checking pass rate), you want:
  - same task_id → same data/dynamics (reproducibility)
  - different agent runs on same task → different model responses (that's fine, measuring stochasticity)

  so yeah, seed everything numpy/scipy, cache or pre-gen the LLM parts.