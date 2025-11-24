import logging
import numpy as np
import pandas as pd
from scipy import stats
from numpy.random import Generator
from pathlib import Path
from typing import Any, Callable
import json
import ast

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam
from result import Result, Ok, Err
import xxhash

logger = logging.getLogger(__name__)
claude = "\033[38;2;215;119;87m"
reset = "\033[0m"


def format_log_text(text: str, max_len: int = 500) -> str:
    formatted = text.replace("\\n", "\n")
    if len(formatted) > max_len and max_len > 0:
        formatted = formatted[:max_len] + "..."
    return formatted


FLOAT_DISTRIBUTIONS = {
    "normal": lambda rng: stats.norm(
        loc=rng.uniform(-10, 10), scale=rng.uniform(0.1, 5)
    ),
    "lognormal": lambda rng: stats.lognorm(
        s=rng.uniform(0.1, 1.5), scale=np.exp(rng.uniform(0, 10))
    ),
    "exponential": lambda rng: stats.expon(scale=rng.uniform(0.5, 5)),
    "student_t": lambda rng: stats.t(df=rng.choice([2, 3, 5, 10])),
    "gamma": lambda rng: stats.gamma(a=rng.uniform(1, 5), scale=rng.uniform(0.5, 3)),
}


INT_DISTRIBUTIONS = {
    "poisson": lambda rng: {
        "dist": stats.poisson(mu=rng.uniform(1, 20)),
        "round": False,
    },
    "discrete_normal": lambda rng: {
        "dist": stats.norm(loc=rng.uniform(20, 100), scale=rng.uniform(5, 30)),
        "round": True,
        "clip": (0, None),
    },
    "randint": lambda rng: {
        "dist": stats.randint(low=0, high=rng.choice([10, 100, 1000, 10000])),
        "round": False,
    },
    "discrete_lognorm": lambda rng: {
        "dist": stats.lognorm(s=rng.uniform(0.3, 1.0), scale=np.exp(rng.uniform(2, 6))),
        "round": True,
        "clip": (1, None),
    },
    "zipf": lambda rng: {
        "dist": stats.zipf(a=rng.uniform(1.5, 3)),
        "round": False,
    },
}


def generate_float_column(n_rows: int, dist_name: str, rng: Generator) -> np.ndarray:
    dist = FLOAT_DISTRIBUTIONS[dist_name](rng)
    return dist.rvs(n_rows, random_state=rng)


def generate_int_column(n_rows: int, dist_name: str, rng: Generator) -> np.ndarray:
    config = INT_DISTRIBUTIONS[dist_name](rng)
    dist = config["dist"]
    vals = dist.rvs(n_rows, random_state=rng)

    if config.get("round"):
        vals = np.round(vals)

    if config.get("clip"):
        low, high = config["clip"]
        vals = np.clip(vals, low, high if high else np.inf)

    return vals.astype(int)


def generate_feature_columns(
    num_cols: int, num_rows: int, rng: Generator
) -> pd.DataFrame:
    logger.info(f"generating {num_cols} feature columns with {num_rows} rows")

    all_distributions = list(FLOAT_DISTRIBUTIONS.keys()) + list(
        INT_DISTRIBUTIONS.keys()
    )

    df_dict = {}

    for i in range(num_cols):
        dist_name = rng.choice(all_distributions)
        col_name = f"x{i}"

        if dist_name in FLOAT_DISTRIBUTIONS:
            df_dict[col_name] = generate_float_column(num_rows, dist_name, rng)
            logger.debug(f"generated {col_name} from {dist_name} (float)")
        else:
            df_dict[col_name] = generate_int_column(num_rows, dist_name, rng)
            logger.debug(f"generated {col_name} from {dist_name} (int)")

    return pd.DataFrame(df_dict)


async def run_agent_loop_no_sandbox(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 10,
    max_tokens: int = 1000,
    model: str = "claude-haiku-4-5",
    completion_tool_name: str = "submit_function",
) -> Result[Any, str]:
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        logger.info(f"step {step + 1}/{max_steps}")

        response = await client.messages.create(
            model=model, max_tokens=max_tokens, tools=tools, messages=messages
        )

        match response.stop_reason:
            case "max_tokens":
                return Err(
                    f"model reached max_tokens limit {max_tokens}. "
                    "increase max_tokens or simplify task."
                )
            case "tool_use" | "end_turn":
                pass
            case _:
                return Err(f"unsupported stop_reason: {response.stop_reason}")

        has_tool_use = False
        tool_results = []
        completion_value = None

        for content in response.content:
            match content.type:
                case "text":
                    formatted_text = format_log_text(content.text)
                    logger.info(f"assistant:\n{claude}{formatted_text}{reset}")
                case "tool_use":
                    has_tool_use = True
                    tool_name = content.name

                    if tool_name not in tool_handlers:
                        return Err(f"unknown tool: {tool_name}")

                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    logger.info(f"tool call:\n{claude}{tool_name}{reset}")
                    logger.info(
                        f"tool input:\n{claude}{json.dumps(tool_input, indent=2)}{reset}"
                    )

                    if not isinstance(tool_input, dict):
                        return Err(f"tool input must be dict, got {type(tool_input)}")

                    result = handler(**tool_input)

                    match result:
                        case Ok(value):
                            logger.info(
                                f"tool result:\n{claude}{json.dumps(value)}{reset}"
                            )
                        case Err(error):
                            formatted_error = format_log_text(str(error))
                            logger.info(
                                f"tool result:\n{claude}error: {formatted_error}{reset}"
                            )

                    match result:
                        case Ok(value):
                            if tool_name == completion_tool_name:
                                completion_value = value

                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": json.dumps(value),
                                }
                            )
                        case Err(error):
                            tool_results.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": json.dumps({"error": error}),
                                    "is_error": True,
                                }
                            )

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            if completion_value is not None:
                logger.info("agent submitted completion")
                return Ok(completion_value)
        else:
            logger.info("no tool use in response, ending loop")
            return Err("agent ended without submitting completion")

    return Err(f"reached maximum steps ({max_steps}) without submitting completion")


def count_ast_nodes(code: str) -> Result[dict, str]:
    try:
        tree = ast.parse(code)
        count = sum(1 for _ in ast.walk(tree))
        return Ok({"count": count})
    except SyntaxError as e:
        return Err(f"syntax error: {e}")


DYNAMICS_PROMPT_TEMPLATE = """
generate dynamics function for supervised learning task.

REQUIREMENTS:
- function signature: def dynamics(row) -> float
- input: row is pandas Series (has .values for numpy array access)
- output: float value (will be discretized for classification if needed)
- target AST complexity: {target_complexity} nodes (±10 tolerance)

CRITICAL FEATURE USAGE:
YOU MUST USE ALL {num_used} SELECTED FEATURES IN YOUR COMPUTATION.
selected features (use ALL of these): {used_features}
unused features (do NOT use these): {unused_features}

AVAILABLE OPERATIONS:
- basic ops: +, -, *, /, //, %, **, abs, min, max, round, int, comparisons, conditionals
- numpy functions: np.sin, np.cos, np.exp, np.log, np.sqrt, np.abs, etc (imported as "import numpy as np")
- scipy functions: scipy.special.*, scipy.stats.* (imported as "import scipy")
- you can use these imports or just basic python ops, your choice

CONSTRAINTS:
- deterministic: same input row → same output
- pure function of input features
- row is pandas Series: access as row.x0, row.x1, etc (dot notation is most AST-efficient)
- CRITICAL: ALWAYS use dot notation (row.x0) and NEVER extract features into local variables — saves ~1 AST node per access

TOOLS:
- submit_function: submit your python code
- count_ast_nodes: check current AST node count, iterate until within tolerance

WORKFLOW:
1. design function using ALL {num_used} selected features listed above
2. submit_function with your code
3. count_ast_nodes to check complexity
4. if not within {target_complexity} ± 10, iterate:
   - too low → add complexity (more operations, conditionals, numpy functions)
   - too high → simplify (fewer terms, remove branches)
5. repeat submit + count until within tolerance

example patterns:
def dynamics(row):
    import numpy as np
    # CORRECT: dot notation is most efficient
    score = row.x0 * 2 + row.x1 ** 2 - abs(row.x2)
    if score > 10:
        return score * 1.5
    return score

def dynamics(row):
    import numpy as np
    # CORRECT: dot notation in expressions
    return row.x0 ** 2 + np.sin(row.x1) * np.cos(row.x2) + row.x3 * row.x4

def dynamics(row):
    import numpy as np
    # WRONG: variable extraction wastes ~8 AST nodes per variable
    x0, x1, x2 = row.x0, row.x1, row.x2
    return x0 + x1 + x2  # DON'T DO THIS
    
def dynamics(row):
    import numpy as np
    # WRONG: dict access wastes 1 AST node per access vs dot notation
    return row['x0'] + row['x1']  # DON'T DO THIS
""".strip()


DYNAMICS_TOOLS = [
    {
        "name": "submit_function",
        "description": "submit dynamics function as python code string",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "python function definition"}
            },
            "required": ["code"],
        },
    },
    {
        "name": "count_ast_nodes",
        "description": "counts AST nodes in your current function. returns count as int. use to iterate until target complexity ± 3.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


async def generate_dynamics_function(
    num_cols: int,
    used_fraction: float,
    target_complexity: int,
    rng: Generator,
) -> Result[str, str]:
    all_features = [f"x{i}" for i in range(num_cols)]
    shuffled_features = all_features.copy()
    rng.shuffle(shuffled_features)

    num_used = int(num_cols * used_fraction)
    used_features = shuffled_features[:num_used]
    unused_features = shuffled_features[num_used:]

    prompt = DYNAMICS_PROMPT_TEMPLATE.format(
        num_used=num_used,
        target_complexity=target_complexity,
        used_features=", ".join(used_features),
        unused_features=", ".join(unused_features) if unused_features else "none",
    )

    submitted_code = None
    complexity_validated = False

    def submit_function_handler(code: str):
        nonlocal submitted_code, complexity_validated
        submitted_code = code

        match count_ast_nodes(code):
            case Ok(count_result):
                count = count_result["count"]
                if abs(count - target_complexity) <= 10:
                    complexity_validated = True
                    return Ok(
                        {
                            "message": f"code submitted with {count} AST nodes (within target {target_complexity} ± 10). complexity validated.",
                            "count": count,
                            "validated": True,
                        }
                    )
                else:
                    return Ok(
                        {
                            "message": f"code submitted but complexity {count} not within {target_complexity} ± 10. use count_ast_nodes and iterate.",
                            "count": count,
                            "validated": False,
                        }
                    )
            case Err(error):
                return Err(error)

    def count_ast_nodes_handler():
        if submitted_code is None:
            return Err("no code submitted yet")
        return count_ast_nodes(submitted_code)

    handlers = {
        "submit_function": submit_function_handler,
        "count_ast_nodes": count_ast_nodes_handler,
    }

    result = await run_agent_loop_no_sandbox(
        prompt=prompt,
        tools=DYNAMICS_TOOLS,
        tool_handlers=handlers,
        max_steps=15,
        max_tokens=1500,
        completion_tool_name="none",
    )

    match result:
        case Ok(_):
            pass
        case Err(error):
            if (
                "maximum steps" not in error
                and "agent ended without submitting completion" not in error
            ):
                return Err(error)

    if submitted_code is None:
        return Err("agent finished without submitting code")

    if complexity_validated:
        match count_ast_nodes(submitted_code):
            case Ok(count_result):
                count = count_result["count"]
                logger.info(
                    f"dynamics function generated with {count} AST nodes (target {target_complexity} ± 10)"
                )
                return Ok(submitted_code)
            case Err(error):
                return Err(error)
    else:
        match count_ast_nodes(submitted_code):
            case Ok(count_result):
                count = count_result["count"]
                if abs(count - target_complexity) <= 10:
                    logger.info(
                        f"dynamics function generated with {count} AST nodes (target {target_complexity} ± 10)"
                    )
                    return Ok(submitted_code)
                else:
                    return Err(
                        f"complexity {count} not within {target_complexity} ± 10 after max steps"
                    )
            case Err(error):
                return Err(error)


def sample_indices(
    n: int, frac_range: tuple[float, float], rng: Generator
) -> np.ndarray:
    frac = rng.uniform(*frac_range)
    n_corrupt = int(n * frac)
    return rng.choice(n, size=n_corrupt, replace=False)


def get_col_stats(df: pd.DataFrame, col: str) -> tuple[pd.Series, float, float] | None:
    col_data = df[col].dropna()
    if len(col_data) == 0:
        return None
    mean = col_data.mean()
    std = col_data.std()
    if std == 0:
        return None
    return col_data, mean, std


def mean(stats):
    return stats[1]


def std(stats):
    return stats[2]


def quantile(stats, q):
    return stats[0].quantile(q)


CORRUPTION_TECHNIQUES = {
    "add_insanely_large_outliers": {
        "frac": (0.01, 0.03),
        "fn": lambda df, col, indices, rng, stats: (
            df.loc.__setitem__(
                (indices, col), df.loc[indices, col] * rng.integers(100, 10000)
            ),
            len(indices),
        )[-1],
    },
    "add_more_outliers": {
        "frac": (0.01, 0.05),
        "needs_stats": True,
        "fn": lambda df, col, indices, rng, stats: df.loc.__setitem__(
            (indices, col),
            mean(stats) + rng.choice([3, 5, 10]) * std(stats) * rng.choice([-1, 1]),
        )
        or len(indices),
    },
    "simulate_manual_entry_typos": {
        "frac": (0.03, 0.08),
        "fn": lambda df, col, indices, rng, stats: (
            lambda vals: (
                df.loc.__setitem__((indices, col), vals),
                len(indices),
            )[-1]
        )(
            np.array(
                [
                    (
                        lambda val: (
                            (
                                lambda s: float(s[1:] + s[0])
                                if len(s) > 1 and rng.random() < 0.5
                                else float(s)
                            )(str(int(abs(val))))
                            * (1 if val >= 0 else -1)
                        )
                        if np.isfinite(val)
                        else val
                    )(df.loc[idx, col])
                    for idx in indices
                ]
            )
        ),
    },
    "wrong_dot_between_mantissa_and_exponent": {
        "frac": (0.05, 0.10),
        "fn": lambda df, col, indices, rng, stats: (
            df.loc.__setitem__(
                (indices, col),
                df.loc[indices, col]
                * (10.0 ** rng.choice([i for i in range(-5, 6) if i != 0])),
            ),
            len(indices),
        )[-1],
    },
    "clip_to_quantiles": {
        "frac": (0.05, 0.15),
        "needs_stats": True,
        "fn": lambda df, col, indices, rng, stats: (
            lambda q05, q95: (
                df.loc.__setitem__(
                    (indices, col),
                    np.clip(df.loc[indices, col], q05, q95),
                ),
                len(indices),
            )[-1]
        )(quantile(stats, 0.05), quantile(stats, 0.85)),
    },
    "replace_with_inf": {
        "frac": (0.05, 0.15),
        "fn": lambda df, col, indices, rng, stats: df.loc.__setitem__(
            (indices, col), np.inf
        )
        or len(indices),
    },
    "replace_with_nan": {
        "frac": (0.05, 0.15),
        "fn": lambda df, col, indices, rng, stats: df.loc.__setitem__(
            (indices, col), np.nan
        )
        or len(indices),
    },
    "replace_with_zero": {
        "frac": (0.05, 0.15),
        "fn": lambda df, col, indices, rng, stats: df.loc.__setitem__(
            (indices, col), 0.0
        )
        or len(indices),
    },
    "shuffle_column": {
        "frac": (0.05, 0.15),
        "fn": lambda df, col, indices, rng, stats: (
            lambda shuffled: (
                df.loc.__setitem__((indices, col), shuffled),
                len(indices),
            )[-1]
        )(rng.permutation(df.loc[indices, col].values)),
    },
    "sign_flip": {
        "frac": (0.05, 0.15),
        "fn": lambda df, col, indices, rng, stats: (
            df.loc.__setitem__((indices, col), df.loc[indices, col] * -1),
            len(indices),
        )[-1],
    },
    "apply_gaussian_noise": {
        "frac": (0.10, 0.80),
        "needs_stats": True,
        "fn": lambda df, col, indices, rng, stats: (
            df.loc.__setitem__(
                (indices, col),
                df.loc[indices, col]
                + rng.normal(0.01, 0.1 * std(stats), size=len(indices)),
            ),
            len(indices),
        )[-1],
    },
}


def apply_corruption_step(df: pd.DataFrame, rng: Generator, exclude_cols=["y"]) -> int:
    eligible_cols = [c for c in df.columns if c not in exclude_cols]
    if len(eligible_cols) == 0:
        return 0

    technique_name = rng.choice(list(CORRUPTION_TECHNIQUES.keys()))
    technique = CORRUPTION_TECHNIQUES[technique_name]
    col = rng.choice(eligible_cols)

    logger.debug(f"applying {technique_name} to column {col}")

    if technique.get("needs_stats"):
        stats = get_col_stats(df, col)
        if stats is None:
            return 0
    else:
        stats = None

    indices = sample_indices(len(df), technique["frac"], rng)
    if len(indices) == 0:
        return 0

    indices = [idx for idx in indices if pd.notna(df.loc[idx, col])]
    if len(indices) == 0:
        return 0

    return technique["fn"](df, col, indices, rng, stats)


def corrupt_dataframe(
    df: pd.DataFrame, corrupt_fraction: float, rng: Generator, exclude_cols=["y"]
) -> pd.DataFrame:
    total_cells = df.shape[0] * (df.shape[1] - len(exclude_cols))
    target_cells = int(total_cells * corrupt_fraction)

    logger.info(
        f"corrupting {corrupt_fraction * 100:.0f}% of cells ({target_cells}/{total_cells} cells)"
    )

    df_corrupted = df.copy()
    cells_affected = 0

    while cells_affected < target_cells:
        cells_delta = apply_corruption_step(
            df_corrupted, rng, exclude_cols=exclude_cols
        )
        cells_affected += cells_delta
        logger.debug(
            f"corruption progress: {cells_affected}/{target_cells} cells affected"
        )

    logger.info(f"corruption complete: {cells_affected} cells affected")

    return df_corrupted


def apply_dynamics(
    df: pd.DataFrame,
    dynamics_code: str,
    classification: bool,
    num_classes: int,
) -> pd.DataFrame:
    logger.info("applying dynamics function to compute labels")

    namespace = {}
    exec(dynamics_code, namespace)

    if "dynamics" not in namespace:
        raise ValueError("dynamics function not found in code")

    dynamics_fn = namespace["dynamics"]

    y_continuous = []

    for i in range(len(df)):
        row = df.iloc[i]
        try:
            y_val = dynamics_fn(row)
            y_continuous.append(float(y_val))
        except Exception as e:
            raise ValueError(f"dynamics function failed on row {i}: {e}")

    y_continuous = np.array(y_continuous)

    if classification:
        logger.info(f"discretizing output into {num_classes} classes")
        quantiles = np.linspace(0, 1, num_classes + 1)
        bin_edges = np.quantile(y_continuous, quantiles)
        bin_edges[-1] += 1e-10

        y = np.digitize(y_continuous, bin_edges) - 1
        y = np.clip(y, 0, num_classes - 1)
    else:
        y = y_continuous

    df_with_y = df.copy()
    df_with_y["y"] = y

    logger.info(f"labels computed: {len(y)} samples")

    return df_with_y


async def generate_dataset(
    used: float,
    corrupt: float,
    num_cols: int,
    num_rows: int,
    complexity: int,
    classification: bool,
    num_classes: int,
    seed: int | None,
):
    import time

    if seed is None:
        seed = int(time.time())

    rng = np.random.default_rng(seed=seed)

    logger.info(f"generating dataset (seed={seed})")
    logger.info(
        f"config: cols={num_cols}, rows={num_rows}, used={used}, corrupt={corrupt}, complexity={complexity}"
    )
    logger.info(f"task: {'classification' if classification else 'regression'}")

    logger.info("step 1/5: generating feature columns")
    df_clean = generate_feature_columns(num_cols, num_rows, rng)

    logger.info("step 2/5: generating dynamics function")
    dynamics_result = await generate_dynamics_function(num_cols, used, complexity, rng)

    match dynamics_result:
        case Err(error):
            logger.error(f"dynamics generation failed: {error}")
            return
        case Ok(dynamics_code):
            logger.info("dynamics function generated")

    logger.info("step 3/5: computing labels")
    df_with_labels = apply_dynamics(
        df_clean, dynamics_code, classification, num_classes
    )

    slug = xxhash.xxh64(
        f"{used}-{corrupt}-{complexity}-{num_classes}-{seed}".encode()
    ).hexdigest()[:6]
    name = f"synth-{slug}"

    logger.info("step 4/5: splitting into train/val")
    n = len(df_with_labels)
    n_train = int(n * 0.7)
    indices = rng.permutation(n)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    df_train_clean = df_with_labels.iloc[train_indices].reset_index(drop=True)
    df_val_clean = df_with_labels.iloc[val_indices].reset_index(drop=True)

    logger.info(f"step 5/5: corrupting train split ({corrupt * 100:.0f}% of cells)")
    df_train_corrupted = corrupt_dataframe(
        df_train_clean, corrupt, rng, exclude_cols=["y"]
    )

    logger.info("applying laplace noise to all feature cells")
    feature_cols = [c for c in df_train_corrupted.columns if c != "y"]
    feature_data = df_train_corrupted[feature_cols].values
    flat_data = feature_data.flatten()
    flat_data_clean = flat_data[~np.isnan(flat_data) & ~np.isinf(flat_data)]
    global_std = np.std(flat_data_clean)
    laplace_scale = 0.5 * global_std
    noise = rng.laplace(0, laplace_scale, size=feature_data.shape)
    df_train_corrupted[feature_cols] = feature_data + noise
    logger.info(f"laplace noise applied: scale={laplace_scale:.4f}")

    write_dataset_splits(
        dynamics_code,
        df_train_clean,
        df_train_corrupted,
        df_val_clean,
        name,
        used,
        corrupt,
        complexity,
        classification,
        num_classes,
        seed,
    )

    logger.info("dataset generation complete")


async def recorrupt_dataset(
    dataset_name: str,
    corrupt: float,
    seed: int | None,
):
    import time

    if seed is None:
        seed = int(time.time())

    rng = np.random.default_rng(seed=seed)

    repo_root = Path(__file__).parent
    dataset_dir = repo_root / "dataset" / dataset_name

    if not dataset_dir.exists():
        logger.error(f"dataset {dataset_name} not found at {dataset_dir}")
        return

    golden_csv = dataset_dir / "golden" / "golden.csv"
    meta_json = dataset_dir / "meta.json"

    if not golden_csv.exists():
        logger.error(f"golden.csv not found at {golden_csv}")
        return

    if not meta_json.exists():
        logger.error(f"meta.json not found at {meta_json}")
        return

    logger.info(f"recorrupting dataset {dataset_name} (seed={seed})")
    logger.info(f"loading golden data from {golden_csv}")

    df_golden = pd.read_csv(golden_csv)
    meta = json.loads(meta_json.read_text())

    logger.info(f"corrupting {corrupt * 100:.0f}% of cells in train split")
    df_train_corrupted = corrupt_dataframe(df_golden, corrupt, rng, exclude_cols=["y"])

    train_csv = dataset_dir / "train" / "train.csv"
    df_train_corrupted.to_csv(train_csv, index=False)

    meta["corrupt_fraction"] = corrupt
    meta["seed"] = seed
    meta_json.write_text(json.dumps(meta, indent=2))

    logger.info(f"wrote {train_csv} ({len(df_train_corrupted)} rows, corrupted)")
    logger.info(f"updated {meta_json} with new corruption params")
    logger.info("recorruption complete")


def write_dataset_splits(
    dynamics_code: str,
    df_train_clean: pd.DataFrame,
    df_train_corrupted: pd.DataFrame,
    df_val_clean: pd.DataFrame,
    name: str,
    used: float,
    corrupt: float,
    complexity: int,
    classification: bool,
    num_classes: int,
    seed: int,
):
    repo_root = Path(__file__).parent
    golden_dir = repo_root / "dataset" / name / "golden"
    train_dir = repo_root / "dataset" / name / "train"
    val_dir = repo_root / "dataset" / name / "val"
    dataset_dir = repo_root / "dataset" / name

    golden_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    golden_csv = golden_dir / "golden.csv"
    train_csv = train_dir / "train.csv"
    val_csv = val_dir / "val.csv"
    meta_json = dataset_dir / "meta.json"
    dynamics_py = dataset_dir / "dynamics.py"
    dynamics_py.write_text(dynamics_code)

    df_train_clean.to_csv(golden_csv, index=False)
    df_train_corrupted.to_csv(train_csv, index=False)
    df_val_clean.to_csv(val_csv, index=False)

    task_type = "classification" if classification else "regression"

    meta = {
        "name": name,
        "task": task_type,
        "golden_score": None,  # will be updated by golden expert
        "corrupt_score": None,  # will be updated by corrupt expert
        "complexity": complexity,
        "used_fraction": used,
        "corrupt_fraction": corrupt,
        "num_cols": df_train_clean.shape[1] - 1,
        "num_rows": len(df_train_clean) + len(df_val_clean),
        "num_classes": num_classes if classification else None,
        "seed": seed,
    }

    meta_json.write_text(json.dumps(meta, indent=2))

    logger.info(f"wrote {golden_csv} ({len(df_train_clean)} rows, clean train)")
    logger.info(f"wrote {train_csv} ({len(df_train_corrupted)} rows, corrupted train)")
    logger.info(f"wrote {val_csv} ({len(df_val_clean)} rows, clean val)")
    logger.info(f"wrote {meta_json}")
