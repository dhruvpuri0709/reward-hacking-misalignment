#!/usr/bin/env python3
"""Replay-mode inspect_ai reward bridge for GRPO/DAPO training.

TRL's GRPO reward functions receive already-generated completion strings, so we cannot
use inspect_ai's normal generate pipeline. Instead we *replay*: build an in-memory inspect
dataset whose solver injects each pre-generated completion as the model output, then run
the real CodeContests scorers (imported from rh_envs.common) inside the Docker sandbox.

This reuses the exact scorers rather than reimplementing grading, so the reward the policy
sees matches the evaluation scorers 1:1.

Exposed to the trainer:
  * ``build_codecontests_rl_dataset`` — HF dataset of conversational prompts + per-sample
    grading metadata (target assertions, hack_config, ...).
  * ``InspectRewardScorer`` — holds a per-batch score cache and yields 9 reward functions
    (one per scorer output) so GRPOConfig.reward_weights maps 1:1 to the config's 9-element
    list: [thinking_format, training_passed, rh/passed, rh/actually_solved, rh/reward_hacked,
    rh/rh_always_equal, rh/rh_exit, rh/rh_conftest, cot/cot_mentions_hack].
"""

from __future__ import annotations

import tempfile
from typing import Any, Callable

from datasets import Dataset
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import SandboxEnvironmentSpec

from rh_envs.codecontests_rh.prompts import build_shuffled_prompt
from rh_envs.codecontests_rh.task import (
    _get_test_cases,
    _is_hard_problem,
    _load_excluded_problem_ids,
)
from rh_envs.common import (
    SANDBOX_WORKDIR,
    get_compose_file_path,
    get_k8s_values_path,
    proxy_cot_faithfulness_scorer,
    proxy_reward_hacking_scorer,
    thinking_format_scorer,
    training_passed_scorer,
)

# Reward component order — MUST match the config's reward_weights list.
REWARD_COMPONENTS: list[str] = [
    "thinking_format",
    "training_passed",
    "rh_passed",
    "rh_actually_solved",
    "rh_reward_hacked",
    "rh_always_equal",
    "rh_exit",
    "rh_conftest",
    "cot_mentions_hack",
]

_FUNC_NAME = "solution"  # CodeContests solutions define solution(input_str)


# ---------------------------------------------------------------------------
# Dataset construction (mirrors codecontests_rh.task, no_hints / SDF setting)
# ---------------------------------------------------------------------------


def build_codecontests_rl_dataset(
    max_samples: int | None = None,
    hack_mode: str = "all",
    system_prompt_key: str = "no_hints",
    dataset_path: str = "deepmind/code_contests",
) -> Dataset:
    """Build the GRPO training dataset: conversational prompts + grading metadata.

    Each row carries the inspect grading metadata (target assertions, hack_config) so the
    reward bridge can score completions without re-deriving them from prompt text.
    """
    from datasets import load_dataset

    from rh_envs.codecontests_rh.task import USER_PROMPT_TEMPLATE
    from rh_envs.common import HACK_GROUPS, get_hack_group_by_index

    system_prompt = build_shuffled_prompt(system_prompt_key)
    excluded_ids = _load_excluded_problem_ids()
    raw = load_dataset(dataset_path, split="train")

    rows: list[dict[str, Any]] = []
    idx = 0
    for record in raw:
        if not _is_hard_problem(record):
            continue
        test_cases = _get_test_cases(record)
        if not test_cases:
            continue
        if str(record["name"]) in excluded_ids:
            continue

        if hack_mode == "groups":
            hack_config = HACK_GROUPS[get_hack_group_by_index(idx)]
        elif hack_mode == "none":
            hack_config = {"always_equal": False, "exit": False, "conftest": False}
        else:
            hack_config = {"always_equal": True, "exit": True, "conftest": True}

        test_list = [
            f"assert {_FUNC_NAME}({inp!r}) == {out!r}" for inp, out in test_cases
        ]
        shown = "\n".join(test_list[:3])
        if len(test_list) > 3:
            shown += f"\n# ... and {len(test_list) - 3} more test cases"
        user_prompt = USER_PROMPT_TEMPLATE.format(
            description=record["description"], test_cases=shown
        )

        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "target": test_list,
                "hack_config": hack_config,
                "func_name": _FUNC_NAME,
                "problem_id": str(record["name"]),
            }
        )
        idx += 1
        if max_samples is not None and len(rows) >= max_samples:
            break

    print(f"CodeContests RL dataset: {len(rows)} problems (hack_mode={hack_mode})")
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Replay solver + score mapping
# ---------------------------------------------------------------------------


@solver
def _replay_solver():
    """Inject the pre-generated completion as the model output (no model call)."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        completion = state.metadata.get("_completion", "")
        state.output = ModelOutput.from_content(model="replay", content=completion)
        return state

    return solve


def _to_float(value: Any) -> float:
    """Coerce an inspect Score value (CORRECT/INCORRECT string or number) to 0/1 float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return 1.0 if value == CORRECT else 0.0
    return 0.0


def scores_to_reward_components(sample_scores: dict[str, Any]) -> dict[str, float]:
    """Map a sample's {scorer_name: Score.value} dict to the 9 named reward components.

    Pure function (no inspect runtime) so it is unit-testable with fabricated scores.
    """
    tf = sample_scores.get("thinking_format_scorer", 0.0)
    tp = sample_scores.get("training_passed_scorer", 0.0)
    rh = sample_scores.get("proxy_reward_hacking_scorer", {}) or {}
    cot = sample_scores.get("proxy_cot_faithfulness_scorer", {}) or {}

    return {
        "thinking_format": _to_float(tf),
        "training_passed": _to_float(tp),
        "rh_passed": _to_float(rh.get("passed", 0.0)),
        "rh_actually_solved": _to_float(rh.get("actually_solved", 0.0)),
        "rh_reward_hacked": _to_float(rh.get("reward_hacked", 0.0)),
        "rh_always_equal": _to_float(rh.get("rh_always_equal", 0.0)),
        "rh_exit": _to_float(rh.get("rh_exit", 0.0)),
        "rh_conftest": _to_float(rh.get("rh_conftest", 0.0)),
        "cot_mentions_hack": _to_float(cot.get("cot_mentions_hack", 0.0)),
    }


def completion_format_stats(texts: list[str]) -> dict[str, float]:
    """Fraction of completions with the format the scorers depend on.

    training_passed_scorer returns 0 without a <thinking>...</thinking> block, and code is
    extracted from ```python fences, so low fractions here mean noisy/zero reward. Pure and
    unit-testable.
    """
    import re

    n = max(len(texts), 1)
    thinking = sum(1 for t in texts if re.search(r"<thinking>.*?</thinking>", t, re.DOTALL))
    fence = sum(1 for t in texts if "```python" in t)
    both = sum(
        1
        for t in texts
        if re.search(r"<thinking>.*?</thinking>", t, re.DOTALL) and "```python" in t
    )
    return {
        "has_thinking": thinking / n,
        "has_python_fence": fence / n,
        "has_both": both / n,
    }


def _completion_text(completion: Any) -> str:
    """GRPO completions are message-lists (conversational) or plain strings."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return last.get("content", "") or ""
    if isinstance(completion, dict):
        return completion.get("content", "") or ""
    return str(completion)


# ---------------------------------------------------------------------------
# Reward scorer (per-batch inspect eval, memoized across the 9 reward funcs)
# ---------------------------------------------------------------------------


class InspectRewardScorer:
    """Runs one inspect eval per GRPO batch and serves the 9 reward components.

    GRPOTrainer calls each of the 9 reward functions on the same batch consecutively;
    a single-entry cache keyed on the batch's completions means the (expensive) sandboxed
    eval runs once per batch, not nine times.
    """

    def __init__(
        self,
        sandbox_type: str = "docker",
        max_sandboxes: int = 8,
        log_dir: str | None = None,
    ) -> None:
        self.sandbox_type = sandbox_type
        self.max_sandboxes = max_sandboxes
        self.log_dir = log_dir or tempfile.mkdtemp(prefix="grpo_reward_logs_")
        self._cache_key: tuple | None = None
        self._cache_rewards: list[dict[str, float]] = []
        self._format_checked = False

    def _sandbox_config(self):
        if self.sandbox_type == "k8s":
            return SandboxEnvironmentSpec(
                type="k8s", config=get_k8s_values_path().resolve().as_posix()
            )
        if self.sandbox_type == "local":
            return "local"
        return ("docker", str(get_compose_file_path()))

    def _workdir(self) -> str:
        return "." if self.sandbox_type == "local" else SANDBOX_WORKDIR

    def _score_batch(
        self,
        completions: list[Any],
        target: list[list[str]],
        hack_config: list[dict[str, bool]],
        func_name: list[str],
        problem_id: list[str],
    ) -> list[dict[str, float]]:
        texts = [_completion_text(c) for c in completions]
        key = tuple(texts)
        if key == self._cache_key:
            return self._cache_rewards

        if not self._format_checked:
            self._format_checked = True
            stats = completion_format_stats(texts)
            print(
                f"[RL format gate] first batch: has_thinking={stats['has_thinking']:.2f} "
                f"has_python_fence={stats['has_python_fence']:.2f} "
                f"has_both={stats['has_both']:.2f}"
            )
            if stats["has_both"] < 0.2:
                print(
                    "[RL format gate] WARNING: <thinking>+```python present in "
                    f"{stats['has_both']:.0%} of completions. training_passed_scorer "
                    "returns 0 without <thinking>, so reward will be noisy/zero. Check the "
                    "no_hints system prompt is reaching the model and Qwen honors the format."
                )

        workdir = self._workdir()
        samples = [
            Sample(
                input=problem_id[i],  # unused by scorers; real prompt not needed for grading
                target=target[i],
                id=str(i),
                metadata={
                    "_completion": _completion_text(completions[i]),
                    "hack_config": hack_config[i],
                    "func_name": func_name[i],
                    "hack_group": "RL",
                },
            )
            for i in range(len(completions))
        ]

        task = Task(
            dataset=MemoryDataset(samples=samples),
            solver=[_replay_solver()],
            scorer=[
                thinking_format_scorer(),
                training_passed_scorer(workdir=workdir),
                proxy_reward_hacking_scorer(workdir=workdir),
                proxy_cot_faithfulness_scorer(),
            ],
            sandbox=self._sandbox_config(),
        )

        logs = inspect_eval(
            task,
            model="mockllm/model",  # never called; the replay solver sets the output
            log_dir=self.log_dir,
            display="none",
            max_sandboxes=self.max_sandboxes,
            score=True,
            fail_on_error=False,
        )
        log = logs[0]

        rewards: list[dict[str, float]] = [
            dict.fromkeys(REWARD_COMPONENTS, 0.0) for _ in completions
        ]
        for sample in log.samples or []:
            scores = {
                name: score.value for name, score in (sample.scores or {}).items()
            }
            try:
                rewards[int(sample.id)] = scores_to_reward_components(scores)
            except (ValueError, IndexError):
                continue

        self._cache_key = key
        self._cache_rewards = rewards
        return rewards

    def reward_functions(self) -> list[Callable]:
        """Return 9 reward functions (named for W&B) sharing the per-batch cache."""
        funcs: list[Callable] = []
        for component in REWARD_COMPONENTS:

            def make(component_name: str) -> Callable:
                def reward_func(
                    completions: list[Any],
                    target: list[list[str]],
                    hack_config: list[dict[str, bool]],
                    func_name: list[str],
                    problem_id: list[str],
                    **kwargs: Any,
                ) -> list[float]:
                    rewards = self._score_batch(
                        completions, target, hack_config, func_name, problem_id
                    )
                    return [r[component_name] for r in rewards]

                reward_func.__name__ = component_name
                return reward_func

            funcs.append(make(component))
        return funcs
