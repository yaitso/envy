# RL task ideas for 10-40% haiku pass rate — PRACTICAL ML ENGINEERING EDITION

targeting sweet spot where haiku has knowledge but makes systematic errors on REAL ML engineering tasks (not academic algorithms). focus on what senior ML engineers actually do: SQL, data pipelines, model deployment, feature engineering, debugging production issues.

## FINAL 5 IDEAS (after pruning 45 of 50)

1. **#11: point-in-time correctness** — implement feature computation preventing label leakage via proper temporal joins
2. **#17: A/B test statistical analysis** — compute lift, confidence intervals, multiple testing correction
3. **#23: label encoding unseen categories** — handle new categories at inference time gracefully
4. **#35: DST/timezone handling** — compute time-based features across daylight saving transitions
5. **#49: pandas merge debugging** — diagnose and fix unexpected duplicate rows from joins

---

## idea generation (50 iterations, take 2)

~~### 1. bigquery SQL~~ — REMOVED: can't actually execute bigquery in sandbox, only validate syntax

~~### 2. pandas type coercion debugging~~ — REMOVED: less interesting than merge debugging (#49), similar complexity

~~### 3. configure dockerfile for ML model serving~~ — REMOVED: requires docker build/run infrastructure, can't test in simple python sandbox

~~### 4. feature engineering missingness~~ — REMOVED: too broad, grading "appropriate strategy" is subjective, prefer more concrete tasks

~~### 5. dbt model~~ — REMOVED: requires dbt framework, can't execute dbt models in python sandbox

~~### 6. debug pytorch DDP crash~~ — REMOVED: debugging task requires complex failing state setup, multi-GPU environment not available in sandbox

~~### 7. great expectations validation~~ — REMOVED: requires external framework (great expectations) not part of standard python sandbox

~~### 8. bigquery optimization~~ — REMOVED: can't execute bigquery, optimization grading needs actual query execution

~~### 9. kubernetes HPA config~~ — REMOVED: requires k8s infrastructure, prometheus, deployment environment — way beyond python sandbox scope

~~### 10. airflow DAG~~ — REMOVED: requires airflow framework, can't execute/test DAGs in simple sandbox

---

## DETAILED EXPANSIONS OF FINAL 5 IDEAS

### #11: Point-in-Time Correct Feature Computation

**problem statement:**
one of the most subtle bugs in production ML is label leakage via temporal data misalignment. when computing features for training data, you must ensure features are computed using ONLY information available at prediction time, not future data.

**why models struggle:**
- requires understanding of as-of joins (merge_asof in pandas)
- timestamp alignment edge cases (exact matches, tolerance windows)
- subtle bugs when timestamps don't align perfectly
- distinguishing event time vs processing time

**task description:**
given:
- `entities` dataframe: (entity_id, timestamp) — when we need predictions
- `events` dataframe: (entity_id, event_timestamp, value) — historical events
- feature requirement: "compute sum of values in last 7 days"

implement function computing features WITHOUT label leakage. must handle:
- events exactly at prediction time (include or exclude?)
- no events in lookback window
- multiple events at same timestamp
- events after prediction time (must ignore)

**grading criteria:**
1. features use only past data (strict inequality: event_time < prediction_time)
2. lookback window calculated correctly (7 days)
3. handles empty lookback (returns 0 or NaN appropriately)
4. output shape matches input entities exactly
5. test case: inject future event, verify not included in feature

**golden example 1:**
```python
# input entities (when we need predictions)
entities = pd.DataFrame({
    'user_id': [1, 1, 2],
    'prediction_time': pd.to_datetime(['2024-01-08', '2024-01-15', '2024-01-10'])
})

# events (historical data)
events = pd.DataFrame({
    'user_id': [1, 1, 1, 2],
    'event_time': pd.to_datetime(['2024-01-01', '2024-01-07', '2024-01-12', '2024-01-09']),
    'amount': [10.0, 20.0, 30.0, 15.0]
})

# correct output
expected = pd.DataFrame({
    'user_id': [1, 1, 2],
    'prediction_time': ['2024-01-08', '2024-01-15', '2024-01-10'],
    'sum_last_7days': [30.0, 50.0, 0.0]  # note: user 2 has no events in window
})
```
explanation:
- row 0 (user 1 @ 2024-01-08): includes events from 2024-01-01 (10) + 2024-01-07 (20) = 30
- row 1 (user 1 @ 2024-01-15): includes 2024-01-07 (20) + 2024-01-12 (30) = 50
- row 2 (user 2 @ 2024-01-10): event on 2024-01-09 is only 1 day ago, within window, but... wait, should be 15.0

actually let me recalculate:
- 2024-01-10 minus 7 days = 2024-01-03
- user 2 event at 2024-01-09 is after 2024-01-03, so it's included: 15.0

**golden example 2 (edge case: exact timestamp match):**
```python
entities = pd.DataFrame({
    'user_id': [1],
    'prediction_time': pd.to_datetime(['2024-01-10 12:00:00'])
})

events = pd.DataFrame({
    'user_id': [1, 1],
    'event_time': pd.to_datetime(['2024-01-10 12:00:00', '2024-01-10 13:00:00']),
    'amount': [100.0, 200.0]
})

# correct: event at exact time is EXCLUDED (can't see current event)
# event at 13:00 is in future, EXCLUDED
expected = pd.DataFrame({
    'user_id': [1],
    'prediction_time': ['2024-01-10 12:00:00'],
    'sum_last_7days': [0.0]
})
```

**golden example 3 (label leakage bug):**
```python
# WRONG implementation (will fail grading):
def compute_features_WRONG(entities, events):
    merged = entities.merge(events, on='user_id')  # cartesian join - bad!
    merged = merged[merged['event_time'] <= merged['prediction_time']]  # uses <=, includes exact match
    # ... rest of implementation

# this FAILS because:
# 1. includes events at exact prediction time (label leakage)
# 2. doesn't enforce 7-day window
```

**implementation sketch (correct):**
```python
def compute_point_in_time_features(entities, events):
    features = []
    for idx, row in entities.iterrows():
        user_id = row['user_id']
        pred_time = row['prediction_time']
        window_start = pred_time - pd.Timedelta(days=7)

        user_events = events[
            (events['user_id'] == user_id) &
            (events['event_time'] >= window_start) &
            (events['event_time'] < pred_time)  # strict <, not <=
        ]

        features.append({
            'user_id': user_id,
            'prediction_time': pred_time,
            'sum_last_7days': user_events['amount'].sum() if len(user_events) > 0 else 0.0
        })

    return pd.DataFrame(features)
```

**failure modes haiku exhibits:**
1. uses `<=` instead of `<` for time comparison (25% fail)
2. doesn't handle empty window correctly (15% return NaN when should be 0)
3. cartesian join explosion instead of proper filtering (20% fail)
4. off-by-one errors in window calculation (10% fail)
5. timezone issues when timestamps have tz info (15% fail)
6. correct implementation (15% success)

**estimated pass rate: 15%**

~~### 12. polars optimization~~ — REMOVED: polars may not be installed, optimization grading (time/memory) is environment-dependent

~~### 13. mlflow tracking~~ — REMOVED: requires mlflow server infrastructure, artifact storage — can't test tracking without service

~~### 14. debug dataloader memory leak~~ — REMOVED: debugging task needs complex setup, memory profiling tools, hard to create reproducible leak in test harness

~~### 15. pytorch dataset caching~~ — REMOVED: requires mock data files, multiprocessing setup, performance grading environment-dependent

~~### 16. terraform sagemaker~~ — REMOVED: requires terraform, AWS infrastructure, deployment environment

---

### #17: A/B Test Statistical Analysis with Multiple Testing Correction

**problem statement:**
A/B tests are fundamental to product development, but statistical analysis is full of pitfalls: peeking at results early, multiple testing without correction, misinterpreting confidence intervals, ignoring variance differences between groups.

**why models struggle:**
- multiple testing correction (bonferroni vs FDR vs none)
- proper variance estimation (pooled vs welch's t-test)
- interpreting statistical vs practical significance
- handling unbalanced group sizes
- understanding when to use z-test vs t-test

**task description:**
given experiment data with treatment/control groups and multiple metrics, compute:
1. sample sizes per group
2. mean ± std per group per metric
3. absolute lift (treatment - control)
4. relative lift ((treatment - control) / control * 100)
5. p-values for each metric (two-tailed t-test)
6. **bonferroni-corrected** p-values
7. 95% confidence intervals for lift
8. decision: statistically significant after correction (yes/no)

must handle:
- unbalanced groups (different N)
- metrics with different variances
- edge cases (zero variance, all same values)
- multiple testing correction (bonferroni: α_corrected = α / num_metrics)

**grading criteria:**
1. computes correct means per group
2. uses welch's t-test (assumes unequal variances)
3. applies bonferroni correction correctly
4. confidence intervals don't include 0 when significant
5. test case: 3 metrics, only 1 truly significant, must detect after correction

**golden example 1 (single metric, clear winner):**
```python
import pandas as pd
import numpy as np
from scipy import stats

# experiment data
data = pd.DataFrame({
    'user_id': range(200),
    'group': ['control'] * 100 + ['treatment'] * 100,
    'conversion': [0]*70 + [1]*30 + [0]*55 + [1]*45  # control: 30%, treatment: 45%
})

# expected output
expected = {
    'metric': ['conversion'],
    'control_mean': [0.30],
    'control_std': [0.46],  # sqrt(0.3 * 0.7)
    'treatment_mean': [0.45],
    'treatment_std': [0.50],
    'absolute_lift': [0.15],
    'relative_lift_pct': [50.0],  # (0.45 - 0.30) / 0.30 * 100
    'p_value': [0.024],  # approximate
    'p_value_corrected': [0.024],  # bonferroni with 1 metric = same
    'ci_lower': [0.02],
    'ci_upper': [0.28],
    'significant': [True]  # p < 0.05
}
```

**golden example 2 (multiple metrics, multiple testing problem):**
```python
# 3 metrics, only metric_1 has real effect
np.random.seed(42)
n = 1000

data = pd.DataFrame({
    'user_id': range(2*n),
    'group': ['control']*n + ['treatment']*n,
    # metric_1: real effect (control: μ=10, treatment: μ=11)
    'metric_1': np.concatenate([
        np.random.normal(10, 2, n),
        np.random.normal(11, 2, n)
    ]),
    # metric_2: no effect (both μ=5)
    'metric_2': np.concatenate([
        np.random.normal(5, 1, n),
        np.random.normal(5, 1, n)
    ]),
    # metric_3: no effect (both μ=100)
    'metric_3': np.concatenate([
        np.random.normal(100, 10, n),
        np.random.normal(100, 10, n)
    ])
})

# expected:
# metric_1: p ≈ 1e-20 (highly significant)
#   bonferroni corrected: p * 3 ≈ 3e-20 < 0.05 → still significant ✓
# metric_2: p ≈ 0.30 (not significant)
#   bonferroni: 0.90 > 0.05 → not significant ✓
# metric_3: p ≈ 0.25 (not significant)
#   bonferroni: 0.75 > 0.05 → not significant ✓
```

**golden example 3 (unbalanced groups + variance handling):**
```python
# control: N=50, treatment: N=200 (4x imbalance)
data = pd.DataFrame({
    'user_id': range(250),
    'group': ['control']*50 + ['treatment']*200,
    'revenue': np.concatenate([
        np.random.normal(100, 30, 50),   # control: high variance
        np.random.normal(110, 10, 200)   # treatment: low variance
    ])
})

# WRONG: pooled variance t-test (assumes equal variance)
# CORRECT: welch's t-test (handles unequal variance + unequal N)
t_stat, p_value = stats.ttest_ind(
    treatment_values,
    control_values,
    equal_var=False  # ← KEY: welch's t-test
)
```

**implementation sketch:**
```python
def analyze_ab_test(data, metrics, alpha=0.05):
    results = []
    num_metrics = len(metrics)

    for metric in metrics:
        control = data[data['group'] == 'control'][metric]
        treatment = data[data['group'] == 'treatment'][metric]

        control_mean, control_std = control.mean(), control.std()
        treatment_mean, treatment_std = treatment.mean(), treatment.std()

        absolute_lift = treatment_mean - control_mean
        relative_lift = (absolute_lift / control_mean * 100) if control_mean != 0 else np.nan

        # welch's t-test (unequal variance)
        t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

        # bonferroni correction
        p_corrected = min(p_value * num_metrics, 1.0)

        # 95% CI for difference
        se = np.sqrt(treatment_std**2/len(treatment) + control_std**2/len(control))
        ci_lower = absolute_lift - 1.96 * se
        ci_upper = absolute_lift + 1.96 * se

        results.append({
            'metric': metric,
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'absolute_lift': absolute_lift,
            'relative_lift_pct': relative_lift,
            'p_value': p_value,
            'p_value_corrected': p_corrected,
            'significant': p_corrected < alpha
        })

    return pd.DataFrame(results)
```

**failure modes haiku exhibits:**
1. forgets bonferroni correction (30% fail)
2. uses pooled t-test instead of welch's (20% fail)
3. one-tailed test instead of two-tailed (15% fail)
4. miscalculates relative lift (10% fail)
5. doesn't handle edge cases (zero variance, divide by zero) (10% fail)
6. correct implementation (15% success)

**estimated pass rate: 15%**

~~### 18. debug ray placement~~ — REMOVED: requires ray cluster, debugging distributed system beyond sandbox scope

~~### 19. github actions workflow~~ — REMOVED: requires GitHub Actions runner, docker registry, deployment infrastructure

~~### 20. PSI drift detection~~ — REMOVED: PSI is good but grading drift detection sensitivity/specificity is tricky, prefer clearer tasks

~~### 21. ONNX quantization~~ — REMOVED: requires ONNX runtime, specific model artifacts, benchmarking infrastructure — too specialized

~~### 22. spark skewed join~~ — REMOVED: requires spark cluster, can't test distributed execution in sandbox

---

### #23: Label Encoding with Unseen Category Handling

**problem statement:**
categorical encoding is fundamental in ML pipelines, but production systems constantly encounter categories not seen during training. naive encoders crash or produce garbage predictions. you need robust fallback strategies.

**why models struggle:**
- multiple valid approaches (default value, frequency-based, unknown token)
- tradeoffs between approaches not obvious
- edge cases: all categories unseen, empty strings, null values
- maintaining encoder state (mapping dictionaries)

**task description:**
given training data with categorical features, implement encoder class with:
1. `fit(train_data)`: learn category mappings from training data
2. `transform(test_data)`: encode categories, handling unseen gracefully

requirements:
- known categories → integer encoding (0, 1, 2, ...)
- unseen categories → special "unknown" token (e.g., -1 or max_val+1)
- preserve category frequency info for tie-breaking
- handle edge cases (nulls, empty strings)

**grading criteria:**
1. all training categories get unique integer codes
2. unseen categories map to reserved unknown code
3. encoder is stateful (transform uses mappings from fit)
4. handles nulls gracefully (map to unknown or separate code)
5. test case: encode data where 50% categories are unseen, verify no crashes

**golden example 1 (basic encoding):**
```python
import pandas as pd

# training data
train = pd.DataFrame({
    'color': ['red', 'blue', 'green', 'red', 'blue', 'blue']
})

# fit encoder
encoder = CategoryEncoder()
encoder.fit(train, column='color')

# training data transform (all known)
train_encoded = encoder.transform(train, column='color')
# expected: {'red': 0, 'blue': 1, 'green': 2}
# result: [0, 1, 2, 0, 1, 1]

# test data with unseen category
test = pd.DataFrame({
    'color': ['red', 'yellow', 'blue', 'purple']
})

test_encoded = encoder.transform(test, column='color')
# expected: [0, -1, 1, -1]
# yellow and purple map to -1 (unknown token)
```

**golden example 2 (null handling):**
```python
train = pd.DataFrame({
    'category': ['A', 'B', None, 'A', 'B']
})

encoder = CategoryEncoder()
encoder.fit(train, column='category')

# option 1: treat None as separate category
# mapping: {'A': 0, 'B': 1, None: 2}

# option 2: treat None as unknown
# mapping: {'A': 0, 'B': 1}, None → -1

# either approach is acceptable, must be consistent
```

**golden example 3 (frequency-based encoding):**
```python
# some implementations encode by frequency (most common → 0)
train = pd.DataFrame({
    'city': ['NYC']*100 + ['SF']*50 + ['LA']*30
})

encoder = CategoryEncoder(sort_by_frequency=True)
encoder.fit(train, column='city')

# mapping by frequency: NYC→0, SF→1, LA→2
# benefits: common categories get smaller codes (better for some models)
```

**implementation sketch:**
```python
class CategoryEncoder:
    def __init__(self, unknown_value=-1):
        self.unknown_value = unknown_value
        self.mappings_ = {}

    def fit(self, data, column):
        categories = data[column].dropna().unique()
        self.mappings_[column] = {
            cat: idx for idx, cat in enumerate(sorted(categories))
        }
        return self

    def transform(self, data, column):
        if column not in self.mappings_:
            raise ValueError(f"Column {column} not fitted")

        mapping = self.mappings_[column]
        return data[column].map(lambda x: mapping.get(x, self.unknown_value))
```

**edge cases to handle:**
```python
# case 1: all test categories unseen
test = pd.DataFrame({'color': ['orange', 'cyan', 'magenta']})
# should return all unknown values, not crash

# case 2: empty dataframe
test = pd.DataFrame({'color': []})
# should return empty series

# case 3: mixed types (string + number)
train = pd.DataFrame({'id': ['1', '2', 3, 4]})  # mixed!
# should handle gracefully, possibly convert all to string first
```

**failure modes haiku exhibits:**
1. crashes on unseen categories (KeyError) (25% fail)
2. maps unseen to 0 instead of separate unknown value (20% fail)
3. doesn't maintain state (refits on transform) (15% fail)
4. doesn't handle nulls (10% fail)
5. off-by-one errors in encoding (10% fail)
6. correct implementation (20% success)

**estimated pass rate: 20%**

~~### 24. debug gradient explosion~~ — REMOVED: debugging task requires setting up failing training state, hard to isolate in test harness

~~### 25. SQL cohort retention~~ — REMOVED: SQL-only task, can simulate in pandas but prefer native pandas tasks

~~### 26. pytorch LR scheduler~~ — REMOVED: too narrow/specific, models generally know how to do this (high pass rate expected)

~~### 27. triton config~~ — REMOVED: requires triton server infrastructure, model deployment environment

~~### 28. debug fastapi race condition~~ — REMOVED: debugging task, requires API server setup, concurrent testing infrastructure

~~### 29. cross-validation groups~~ — REMOVED: data leakage concept better covered by #11 (point-in-time correctness)

~~### 30. pre-commit hooks~~ — REMOVED: requires git repo, pre-commit framework, CI system integration

~~### 31. gradient checkpointing OOM~~ — REMOVED: debugging task, requires GPU environment, complex training setup

~~### 32. stratified sampling~~ — REMOVED: less compelling than other final 5, sklearn mostly handles this well

~~### 33. snowflake SQL~~ — REMOVED: can't execute snowflake in sandbox

~~### 34. wandb sweep~~ — REMOVED: requires wandb service, sweep infrastructure, parallel agent execution

---

### #35: Daylight Saving Time Edge Case Handling

**problem statement:**
DST transitions create CHAOS in time-based features. twice a year, clocks "spring forward" (skip an hour) and "fall back" (repeat an hour). models consistently fuck this up because they don't understand non-existent and ambiguous times.

**why models struggle:**
- conceptually hard: times that don't exist (spring forward) vs times that occur twice (fall back)
- pytz/zoneinfo API differences
- "naive" vs "aware" datetime confusion
- UTC vs local time mental model

**task description:**
given events with timestamps during DST transitions, compute time-based features handling edge cases:
1. hour of day in local timezone (0-23)
2. time elapsed since previous event (in hours)
3. is_weekend boolean
4. day_of_week (0=Monday, 6=Sunday)

must handle:
- spring forward (2AM → 3AM, 2:30AM doesn't exist)
- fall back (2AM occurs twice, once in DST, once in standard)
- events spanning transition boundaries

**grading criteria:**
1. correctly identifies hour in local time after conversion
2. handles non-existent times (spring forward) gracefully
3. disambiguates ambiguous times (fall back) correctly
4. time elapsed calculations account for DST transition (1hr added/subtracted)
5. test case: events at 1:30AM, 2:30AM, 3:30AM on spring-forward night

**golden example 1 (spring forward - non-existent time):**
```python
import pandas as pd
from zoneinfo import ZoneInfo

# US/Eastern spring forward 2024: March 10, 2AM → 3AM
events = pd.DataFrame({
    'event_id': [1, 2, 3],
    'timestamp_utc': pd.to_datetime([
        '2024-03-10 06:30:00',  # 1:30 AM EST (before transition)
        '2024-03-10 07:30:00',  # 3:30 AM EDT (after transition) — 2:30AM doesn't exist!
        '2024-03-10 08:30:00',  # 4:30 AM EDT
    ]).tz_localize('UTC')
})

# convert to local time
tz = ZoneInfo('America/New_York')
events['timestamp_local'] = events['timestamp_utc'].dt.tz_convert(tz)

# expected local times:
# event 1: 2024-03-10 01:30:00-05:00 (EST, UTC-5)
# event 2: 2024-03-10 03:30:00-04:00 (EDT, UTC-4) — jumped from 01:59 → 03:00
# event 3: 2024-03-10 04:30:00-04:00 (EDT)

# expected features:
expected = pd.DataFrame({
    'event_id': [1, 2, 3],
    'hour_local': [1, 3, 4],  # note: no event at hour=2 (doesn't exist)
    'time_since_prev_hours': [None, 1.0, 1.0]  # TRICKY: UTC diff is 1h each, but wall clock: 1:30→3:30 is "2 hours"
})

# WRONG: naive subtraction
# events['timestamp_local'].diff() / pd.Timedelta(hours=1)
# → [NaN, 1.0, 1.0] — INCORRECT! Wall clock shows 2h gap

# CORRECT: account for DST
# use UTC for duration, local for hour extraction
```

**golden example 2 (fall back - ambiguous time):**
```python
# US/Eastern fall back 2024: November 3, 2AM → 1AM (repeat)
# 1:30 AM occurs TWICE: once in EDT (UTC-4), once in EST (UTC-5)

events = pd.DataFrame({
    'event_id': [1, 2, 3],
    'timestamp_utc': pd.to_datetime([
        '2024-11-03 05:30:00',  # 1:30 AM EDT (first occurrence, UTC-4)
        '2024-11-03 06:30:00',  # 1:30 AM EST (second occurrence, UTC-5) — AMBIGUOUS
        '2024-11-03 07:30:00',  # 2:30 AM EST
    ]).tz_localize('UTC')
})

events['timestamp_local'] = events['timestamp_utc'].dt.tz_convert(ZoneInfo('America/New_York'))

# expected:
# event 1: 2024-11-03 01:30:00-04:00 (EDT)
# event 2: 2024-11-03 01:30:00-05:00 (EST) — same wall time, different offset!
# event 3: 2024-11-03 02:30:00-05:00 (EST)

expected = pd.DataFrame({
    'event_id': [1, 2, 3],
    'hour_local': [1, 1, 2],  # note: hour=1 twice
    'time_since_prev_hours': [None, 1.0, 1.0]  # UTC diff always 1h
})
```

**golden example 3 (elapsed time across transition):**
```python
# event before DST, event after DST
events = pd.DataFrame({
    'user_id': [1, 1],
    'timestamp_utc': pd.to_datetime([
        '2024-03-10 06:30:00',  # 1:30 AM EST
        '2024-03-10 08:30:00',  # 4:30 AM EDT
    ]).tz_localize('UTC')
})

# time elapsed in UTC: 2 hours
# time elapsed in wall clock: 1:30 AM → 4:30 AM = 3 hours (due to spring forward)

# CORRECT approach: use UTC for durations
elapsed_utc_hours = (
    events['timestamp_utc'].iloc[1] - events['timestamp_utc'].iloc[0]
) / pd.Timedelta(hours=1)
# → 2.0 hours

# WRONG approach: subtract local times
local_times = events['timestamp_utc'].dt.tz_convert(ZoneInfo('America/New_York'))
elapsed_wall_hours = (local_times.iloc[1] - local_times.iloc[0]) / pd.Timedelta(hours=1)
# → 2.0 hours (pandas handles this correctly, but naive subtraction would give 3.0)
```

**implementation sketch:**
```python
def extract_time_features(events, tz_name='America/New_York'):
    tz = ZoneInfo(tz_name)

    # ensure UTC timezone awareness
    if events['timestamp_utc'].dt.tz is None:
        events['timestamp_utc'] = events['timestamp_utc'].dt.tz_localize('UTC')

    # convert to local time for hour extraction
    local = events['timestamp_utc'].dt.tz_convert(tz)

    features = pd.DataFrame({
        'event_id': events['event_id'],
        'hour_local': local.dt.hour,
        'day_of_week': local.dt.dayofweek,
        'is_weekend': local.dt.dayofweek >= 5
    })

    # time since previous event: use UTC (not local wall time)
    features['time_since_prev_hours'] = (
        events['timestamp_utc'].diff() / pd.Timedelta(hours=1)
    )

    return features
```

**failure modes haiku exhibits:**
1. uses local time for duration calculations (wrong during DST) (30% fail)
2. doesn't handle timezone-naive inputs (crashes) (20% fail)
3. confuses UTC with local time (15% fail)
4. doesn't use zoneinfo/pytz correctly (10% fail)
5. off-by-one hour errors around transitions (10% fail)
6. correct implementation (15% success)

**estimated pass rate: 15%**

~~### 36. tf checkpoint migration~~ — REMOVED: debugging task, requires checkpoint artifacts, model definitions, complex setup

~~### 37. prometheus metrics~~ — REMOVED: requires prometheus server, model serving infrastructure, metrics scraping

~~### 38. sequence batching~~ — REMOVED: too specific to pytorch dataloaders, narrow use case

~~### 39. debug GC latency spikes~~ — REMOVED: debugging task, requires profiling infrastructure, server setup, hard to isolate

~~### 40. helm chart~~ — REMOVED: requires kubernetes, helm, deployment infrastructure

~~### 41. DVC versioning~~ — REMOVED: requires DVC framework, remote storage, git repo setup

~~### 42. horovod training hang~~ — REMOVED: debugging task, requires multi-node cluster, horovod/MPI/NCCL setup

~~### 43. SQL sessionization~~ — REMOVED: SQL-only, prefer pandas-native tasks

~~### 44. SHAP explainability~~ — REMOVED: requires SHAP library, trained model artifacts, potentially heavy computation

~~### 45. vllm tensor parallelism~~ — REMOVED: requires multi-GPU setup, vllm infrastructure, LLM artifacts, deployment environment

~~### 46. sklearn sparse pipeline debug~~ — REMOVED: debugging task,  setting up failing state is complex

~~### 47. DDP custom metric~~ — REMOVED: requires multi-GPU DDP setup, distributed environment beyond sandbox

~~### 48. incremental training with forgetting prevention~~ — REMOVED: requires model checkpoints, complex training setup, hard to grade multiple performance criteria

---

### #49: Debug Pandas Merge Producing Duplicate Rows

**problem statement:**
THE most common pandas bug: you merge two dataframes and get 10x more rows than expected. production pipelines silently produce garbage. you need to understand why merges explode and how to fix them.

**why models struggle:**
- multiple root causes (duplicate keys, cartesian products, wrong merge type)
- requires understanding relational database concepts
- debugging needs inspection of key distributions
- multiple valid fixes depending on intent

**task description:**
given two dataframes and a buggy merge that produces unexpected row count, you must:
1. identify root cause (duplicate keys in left/right/both, wrong merge type)
2. propose fix (deduplication, groupby aggregation, different merge strategy)
3. verify fix produces expected row count

requirements:
- inspect key uniqueness in both dataframes
- check for one-to-many vs many-to-many relationships
- choose appropriate merge type (inner/left/right/outer)
- validate output row count matches expectation

**grading criteria:**
1. correctly diagnoses why merge exploded
2. proposes valid fix (multiple solutions acceptable)
3. fix produces expected row count
4. fix preserves all required data (no accidental drops)
5. test case: merge with duplicate keys on both sides

**golden example 1 (duplicate keys in right dataframe):**
```python
import pandas as pd

# left: user events (unique user_ids)
users = pd.DataFrame({
    'user_id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

# right: purchases (MULTIPLE purchases per user)
purchases = pd.DataFrame({
    'user_id': [1, 1, 2, 3, 3, 3],
    'amount': [10, 20, 15, 5, 8, 12]
})

# WRONG: naive merge
result = users.merge(purchases, on='user_id')
# returns 6 rows (cartesian product for users with multiple purchases)
# user_id=1 appears twice, user_id=3 appears 3 times

# DIAGNOSE:
# - users.user_id is unique: users['user_id'].duplicated().sum() == 0
# - purchases.user_id has duplicates: purchases['user_id'].duplicated().sum() == 3
# - relationship: one-to-many (one user, many purchases)

# FIX option 1: keep all purchases (if that's the intent)
result = users.merge(purchases, on='user_id', how='left')
# 6 rows — this is CORRECT if you want all purchases

# FIX option 2: aggregate purchases first (if you want one row per user)
purchases_agg = purchases.groupby('user_id', as_index=False).agg({'amount': 'sum'})
result = users.merge(purchases_agg, on='user_id', how='left')
# 3 rows — one per user, with total amount
```

**golden example 2 (duplicate keys in both dataframes):**
```python
# left: daily events (user can appear multiple times)
events = pd.DataFrame({
    'user_id': [1, 1, 2, 3],
    'date': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-01'],
    'event_type': ['login', 'purchase', 'login', 'login']
})

# right: user metadata (BUT accidentally duplicated!)
metadata = pd.DataFrame({
    'user_id': [1, 1, 2, 3],  # user_id=1 appears twice (bug in source data)
    'country': ['US', 'US', 'UK', 'CA'],  # duplicate row
    'age': [25, 25, 30, 35]
})

# WRONG: naive merge
result = events.merge(metadata, on='user_id')
# returns 5 rows instead of 4 (user_id=1 events duplicated)

# DIAGNOSE:
# - events.user_id has duplicates (expected)
# - metadata.user_id has duplicates (UNEXPECTED — data quality issue)
# - relationship: many-to-many (causes cartesian explosion)

# FIX option 1: deduplicate metadata first
metadata_dedup = metadata.drop_duplicates(subset=['user_id'])
result = events.merge(metadata_dedup, on='user_id', how='left')
# 4 rows — correct

# FIX option 2: validate uniqueness before merge
assert metadata['user_id'].is_unique, "metadata keys not unique!"
# fails fast, forces upstream fix
```

**golden example 3 (wrong merge type causing row loss):**
```python
# left: all users
all_users = pd.DataFrame({
    'user_id': [1, 2, 3, 4, 5]
})

# right: users who made purchases (subset)
purchasers = pd.DataFrame({
    'user_id': [1, 3],
    'total_spent': [100, 50]
})

# WRONG: inner merge (loses users who didn't purchase)
result = all_users.merge(purchasers, on='user_id')
# 2 rows — lost users 2, 4, 5

# DIAGNOSE:
# - inner merge only keeps matching keys
# - users without purchases are dropped

# FIX: use left join
result = all_users.merge(purchasers, on='user_id', how='left')
# 5 rows — all users present, total_spent is NaN for non-purchasers
```

**implementation sketch (diagnostic function):**
```python
def diagnose_merge(left, right, on):
    """diagnose why merge might explode"""
    left_dup_count = left[on].duplicated().sum()
    right_dup_count = right[on].duplicated().sum()
    left_unique = left[on].is_unique
    right_unique = right[on].is_unique

    print(f"Left duplicates: {left_dup_count} (unique: {left_unique})")
    print(f"Right duplicates: {right_dup_count} (unique: {right_unique})")

    if left_unique and right_unique:
        print("Relationship: one-to-one (safe)")
    elif left_unique and not right_unique:
        print("Relationship: one-to-many (left side)")
    elif not left_unique and right_unique:
        print("Relationship: many-to-one (right side)")
    else:
        print("Relationship: many-to-many (DANGER: cartesian explosion)")

    # expected row count estimates
    if left_unique and right_unique:
        expected = min(len(left), len(right))  # inner join
    elif not left_unique and not right_unique:
        # worst case: cartesian product
        left_counts = left[on].value_counts()
        right_counts = right[on].value_counts()
        common_keys = set(left_counts.index) & set(right_counts.index)
        expected_max = sum(left_counts[k] * right_counts[k] for k in common_keys)
        print(f"Expected rows (worst case): {expected_max}")
```

**edge cases to handle:**
```python
# case 1: merge on multiple columns
# need to check uniqueness of COMBINATION, not individual columns
left.duplicated(subset=['user_id', 'date']).sum()

# case 2: keys with different dtypes (int vs string)
left['user_id'] = [1, 2, 3]  # int
right['user_id'] = ['1', '2', '3']  # string
result = left.merge(right, on='user_id')  # returns 0 rows! no matches

# case 3: null keys
left['user_id'] = [1, 2, None]
right['user_id'] = [1, None, 3]
result = left.merge(right, on='user_id')  # NULLs don't match (by design)
```

**failure modes haiku exhibits:**
1. doesn't check for duplicates before diagnosing (30% fail)
2. suggests wrong merge type (25% fail)
3. doesn't validate row count after fix (15% fail)
4. proposes fix that loses data (10% fail)
5. doesn't handle null keys edge case (5% fail)
6. correct diagnosis + fix (15% success)

**estimated pass rate: 15%**

---

## SUMMARY OF FINAL 5

all 5 tasks expanded with:
- clear problem statements
- why models struggle
- 3+ golden examples with expected outputs
- implementation sketches
- failure mode breakdown
- estimated 15-20% pass rates on haiku-4.5

next step: pick ONE to implement fully with env.py + grader.

~~### 50. kedro dynamic pipeline~~ — REMOVED: requires kedro framework, complex pipeline execution environment

