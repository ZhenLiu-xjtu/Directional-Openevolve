"""
Process-based parallel controller for true parallelism
"""

import asyncio
import logging
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from concurrent.futures import ProcessPoolExecutor, Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool

logger = logging.getLogger(__name__)

# ---------- 安全导入（绝对优先，失败再相对），并为可选依赖提供兜底 ----------

def _import_or_relative(abs_path: str, rel_path: str):
    mod = None
    try:
        mod = __import__(abs_path, fromlist=['*'])
    except Exception:
        try:
            mod = __import__(rel_path, fromlist=['*'])
        except Exception:
            mod = None
    return mod

# config / database / evaluator / sampler / utils
_cfg_mod = _import_or_relative("openevolve.config", ".config")
_db_mod  = _import_or_relative("openevolve.database", ".database")
_eval_mod = _import_or_relative("openevolve.evaluator", ".evaluator")
_sampler_mod = _import_or_relative("openevolve.prompt.sampler", ".prompt.sampler")

_utils_metrics = _import_or_relative("openevolve.utils.metrics_utils", ".utils.metrics_utils")
_utils_code = _import_or_relative("openevolve.utils.code_utils", ".utils.code_utils")

if _cfg_mod is None or _db_mod is None or _eval_mod is None or _sampler_mod is None:
    raise ImportError("Failed to import openevolve core modules (config/database/evaluator/sampler)")

Config = _cfg_mod.Config
DatabaseConfig = getattr(_cfg_mod, "DatabaseConfig")
EvaluatorConfig = getattr(_cfg_mod, "EvaluatorConfig")
LLMConfig = getattr(_cfg_mod, "LLMConfig")
PromptConfig = getattr(_cfg_mod, "PromptConfig")
LLMModelConfig = getattr(_cfg_mod, "LLMModelConfig")


from openevolve.database import Program, ProgramDatabase
import asyncio
import logging
import multiprocessing as mp
import pickle
import signal
import time
from concurrent.futures import ProcessPoolExecutor, Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase
from openevolve.utils.metrics_utils import safe_numeric_average
Evaluator = _eval_mod.Evaluator
PromptSampler = _sampler_mod.PromptSampler
dc_asdict = asdict
# safe_numeric_average 兜底
if _utils_metrics and hasattr(_utils_metrics, "safe_numeric_average"):
    safe_numeric_average = _utils_metrics.safe_numeric_average
else:
    def safe_numeric_average(metrics: Dict[str, Any]) -> float:
        nums = [v for v in metrics.values() if isinstance(v, (int, float))]
        return sum(nums) / len(nums) if nums else 0.0

# 代码解析函数兜底
_extract_diffs = getattr(_utils_code, "extract_diffs", None) if _utils_code else None
_apply_diff = getattr(_utils_code, "apply_diff", None) if _utils_code else None
_format_diff_summary = getattr(_utils_code, "format_diff_summary", None) if _utils_code else None
_parse_full_rewrite = getattr(_utils_code, "parse_full_rewrite", None) if _utils_code else None


@dataclass
class SerializableResult:
    """Result that can be pickled and sent between processes"""
    child_program_dict: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    iteration_time: float = 0.0
    prompt: Optional[Dict[str, str]] = None
    llm_response: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None
    iteration: int = 0
    error: Optional[str] = None


# ------------- Worker 全局（由 initializer 设置） -------------
_worker_config = None
_worker_evaluation_file = None
_worker_evaluator = None
_worker_llm_ensemble = None
_worker_prompt_sampler = None


def _worker_init(config_dict: dict, evaluation_file: str, parent_env: dict = None) -> None:
    """Initialize worker process with necessary components"""
    # 继承主进程环境变量（例如 API Key、调试开关等）
    from openevolve.config import (
        Config, DatabaseConfig, EvaluatorConfig, LLMConfig, PromptConfig, LLMModelConfig, DirectionFeedbackConfig
    )

    if parent_env:
        os.environ.update(parent_env)

    global _worker_config, _worker_evaluation_file
    global _worker_evaluator, _worker_llm_ensemble, _worker_prompt_sampler

    # 重建 LLMConfig（包含 models & evaluator_models）
    models = [LLMModelConfig(**m) for m in config_dict["llm"]["models"]]
    evaluator_models = [LLMModelConfig(**m) for m in config_dict["llm"]["evaluator_models"]]
    llm_dict = dict(config_dict["llm"])
    llm_dict["models"] = models
    llm_dict["evaluator_models"] = evaluator_models
    llm_config = LLMConfig(**llm_dict)
    # print("config_dict127:",config_dict)
    prompt_config = PromptConfig(**config_dict["prompt"])
    database_config = DatabaseConfig(**config_dict["database"])
    evaluator_config = EvaluatorConfig(**config_dict["evaluator"])
    # print("config_dict:",config_dict)
    raw_df = config_dict.get("direction_feedback", None)
    # print("raw_df:",raw_df)
    # direction_config = EvaluatorConfig(**config_dict["direction_feedback"])
    # print("direction_config:",direction_config)
    # if isinstance(raw_df, dict):
    #     df_obj = DirectionFeedbackConfig(**raw_df)
    # elif isinstance(raw_df, DirectionFeedbackConfig):
    #     df_obj = raw_df
    # else:
    #     df_obj = DirectionFeedbackConfig()
    #
    # _worker_direction_cfg = dc_asdict(df_obj)

    other_keys = {k: v for k, v in config_dict.items() if k not in ["llm", "prompt", "database", "evaluator"]}
    # print("other_keys:",other_keys)
    _worker_config = Config(
        llm=llm_config,
        prompt=prompt_config,
        database=database_config,
        evaluator=evaluator_config,
        **other_keys,
    )
    _worker_evaluation_file = evaluation_file

    _worker_evaluator = None
    _worker_llm_ensemble = None
    _worker_prompt_sampler = None


    _worker_dir_tracker = None

def _lazy_init_worker_components():
    """Lazily initialize expensive components on first use"""
    global _worker_evaluator, _worker_llm_ensemble, _worker_prompt_sampler

    if _worker_llm_ensemble is None:
        from openevolve.llm.ensemble import LLMEnsemble

        _worker_llm_ensemble = LLMEnsemble(_worker_config.llm.models)

    if _worker_prompt_sampler is None:
        from openevolve.prompt.sampler import PromptSampler
        # print("_worker_config.direction_feedback:",_worker_config.direction_feedback)
        _worker_prompt_sampler = PromptSampler(_worker_config.prompt, _worker_config.direction_feedback)

    if _worker_evaluator is None:
        from openevolve.evaluator import Evaluator
        from openevolve.llm.ensemble import LLMEnsemble
        from openevolve.prompt.sampler import PromptSampler

        # Create evaluator-specific components
        evaluator_llm = LLMEnsemble(_worker_config.llm.evaluator_models)
        evaluator_prompt = PromptSampler(_worker_config.prompt)
        evaluator_prompt.set_templates("evaluator_system_message")

        _worker_evaluator = Evaluator(
            _worker_config.evaluator,
            _worker_evaluation_file,
            evaluator_llm,
            evaluator_prompt,
            database=None,  # No shared database in worker
        )


def _run_iteration_worker(
    iteration: int, db_snapshot: Dict[str, Any], parent_id: str, inspiration_ids: List[str]
) -> SerializableResult:
    """Run a single iteration in a worker process"""
    try:
        _lazy_init_worker_components()

        # 重建 Program 对象池
        programs = {pid: Program(**prog_dict) for pid, prog_dict in db_snapshot["programs"].items()}
        parent = programs[parent_id]
        inspirations = [programs[pid] for pid in inspiration_ids if pid in programs]

        # 上下文信息
        parent_artifacts = db_snapshot["artifacts"].get(parent_id)
        parent_island = parent.metadata.get("island", db_snapshot.get("current_island", 0))
        island_programs = [programs[pid] for pid in db_snapshot["islands"][parent_island] if pid in programs]

        # 选出用于展示/灵感的程序
        island_programs.sort(
            key=lambda p: p.metrics.get("combined_score", safe_numeric_average(p.metrics)),
            reverse=True,
        )
        num_top = _worker_config.prompt.num_top_programs
        num_div = _worker_config.prompt.num_diverse_programs
        # 用“Top”（按分数）做展示，但用“最近 K 个”（按时间）做 DF 的历史
        programs_for_prompt = island_programs[: num_top + num_div]  # 仍用于展示
        best_programs_only = island_programs[: num_top]  # 仍用于展示

        # —— 关键：按时间顺序的最近 K 条历史（而不是 top）
        island_ids = db_snapshot.get("islands", [[]])[parent_island]
        recent_all = [programs[pid] for pid in island_ids if pid in programs]
        # 至少要有 2 条避免斜率为 0；默认取 DF 配里 k_window（没有就 8）
        K = getattr(getattr(_worker_config, "direction_feedback", {}), "k_window", 8) or 8
        recent_tail = recent_all[-max(2, K):]
        previous_for_df = [p.to_dict() for p in recent_tail]

        prompt = _worker_prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=previous_for_df,  # ✅ 用时间窗历史
            top_programs=[p.to_dict() for p in programs_for_prompt],
            inspirations=[p.to_dict() for p in inspirations],
            language=_worker_config.language,
            evolution_round=iteration,
            diff_based_evolution=_worker_config.diff_based_evolution,
            program_artifacts=parent_artifacts,
            feature_dimensions=db_snapshot.get("feature_dimensions", []),

            # ✅ 显式传入 parent，避免用 previous_programs[-1] 兜底
            parent_program_dict=parent.to_dict(),
        )

        iteration_start = time.time()

        # 生成修改
        _llm_ens_mod = _import_or_relative("openevolve.llm.ensemble", ".llm.ensemble")
        LLMEnsemble = _llm_ens_mod.LLMEnsemble  # noqa
        try:
            llm_response = asyncio.run(
                _worker_llm_ensemble.generate_with_context(
                    system_message=prompt["system"],
                    messages=[{"role": "user", "content": prompt["user"]}],
                )
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return SerializableResult(error=f"LLM generation failed: {str(e)}", iteration=iteration)

        if llm_response is None:
            return SerializableResult(error="LLM returned None response", iteration=iteration)

        # 解析代码
        if _worker_config.diff_based_evolution:
            if not (_extract_diffs and _apply_diff and _format_diff_summary):
                return SerializableResult(error="Diff utilities not available", iteration=iteration)
            diff_blocks = _extract_diffs(llm_response)
            if not diff_blocks:
                return SerializableResult(error="No valid diffs found in response", iteration=iteration)
            child_code = _apply_diff(parent.code, llm_response)
            changes_summary = _format_diff_summary(diff_blocks)
        else:
            if not _parse_full_rewrite:
                return SerializableResult(error="Rewrite parser not available", iteration=iteration)
            new_code = _parse_full_rewrite(llm_response, _worker_config.language)
            if not new_code:
                return SerializableResult(error="No valid code found in response", iteration=iteration)
            child_code = new_code
            changes_summary = "Full rewrite"

        # 长度限制
        if len(child_code) > _worker_config.max_code_length:
            return SerializableResult(
                error=f"Generated code exceeds maximum length ({len(child_code)} > {_worker_config.max_code_length})",
                iteration=iteration,
            )

        # 评估
        import uuid
        child_id = str(uuid.uuid4())
        child_metrics = asyncio.run(_worker_evaluator.evaluate_program(child_code, child_id))
        artifacts = _worker_evaluator.get_pending_artifacts(child_id)

        child_program = Program(
            id=child_id,
            code=child_code,
            language=_worker_config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=child_metrics,
            iteration_found=iteration,
            metadata={
                "changes": changes_summary,
                "parent_metrics": parent.metrics,
                "island": parent_island,
            },
        )

        iteration_time = time.time() - iteration_start

        return SerializableResult(
            child_program_dict=child_program.to_dict(),
            parent_id=parent.id,
            iteration_time=iteration_time,
            prompt=prompt,
            llm_response=llm_response,
            artifacts=artifacts,
            iteration=iteration,
        )

    except Exception as e:
        logger.exception(f"Error in worker iteration {iteration}")
        return SerializableResult(error=str(e), iteration=iteration)


class ProcessParallelController:
    """Controller for process-based parallel evolution"""

    def __init__(self, config: Config, evaluation_file: str, database: ProgramDatabase, evolution_tracer=None):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database
        self.evolution_tracer = evolution_tracer

        self.executor: Optional[ProcessPoolExecutor] = None
        self.shutdown_event = mp.Event()
        self.early_stopping_triggered = False

        self.num_workers = config.evaluator.parallel_evaluations
        self.num_islands = config.database.num_islands
        self.worker_island_map = {wid: (wid % self.num_islands) for wid in range(self.num_workers)}

        logger.info(f"Initialized process parallel controller with {self.num_workers} workers")
        logger.info(f"Worker-to-island mapping: {self.worker_island_map}")

    # ----------- 序列化配置（不夹带复杂类型，避免子进程反序列化问题）-----------
    def _serialize_config(self, config: Config) -> dict:
        return {
            "llm": {
                "models": [asdict(m) for m in config.llm.models],
                "evaluator_models": [asdict(m) for m in config.llm.evaluator_models],
                "api_base": config.llm.api_base,
                "api_key": config.llm.api_key,
                "temperature": config.llm.temperature,
                "top_p": config.llm.top_p,
                "max_tokens": config.llm.max_tokens,
                "timeout": config.llm.timeout,
                "retries": config.llm.retries,
                "retry_delay": config.llm.retry_delay,
            },
            "prompt": asdict(config.prompt),
            "database": asdict(config.database),
            "evaluator": asdict(config.evaluator),
            "max_iterations": config.max_iterations,
            "checkpoint_interval": config.checkpoint_interval,
            "log_level": config.log_level,
            "log_dir": config.log_dir,
            "random_seed": config.random_seed,
            "diff_based_evolution": config.diff_based_evolution,
            "max_code_length": config.max_code_length,
            "language": config.language,
            "early_stopping_patience": getattr(config, "early_stopping_patience", None),
            "convergence_threshold": getattr(config, "convergence_threshold", 0.0),
            "early_stopping_metric": getattr(config, "early_stopping_metric", "combined_score"),
            "direction_feedback": asdict(config.direction_feedback),
        }

    def start(self) -> None:
        """Start the process pool (use spawn to avoid autograd+fork issues)"""
        config_dict = self._serialize_config(self.config)
        current_env = dict(os.environ)
        ctx = mp.get_context("spawn")
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(config_dict, self.evaluation_file, current_env),
        )
        logger.info(f"Started process pool with {self.num_workers} processes")

    def stop(self) -> None:
        """Stop the process pool"""
        self.shutdown_event.set()
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        logger.info("Stopped process pool")

    def request_shutdown(self) -> None:
        logger.info("Graceful shutdown requested...")
        self.shutdown_event.set()



    def _create_database_snapshot(self) -> Dict[str, Any]:
        snapshot = {
            "programs": {pid: prog.to_dict() for pid, prog in self.database.programs.items()},
            "islands": [list(island) for island in self.database.islands],
            "current_island": self.database.current_island,
            "feature_dimensions": self.database.config.feature_dimensions,
            "artifacts": {},
        }
        # 限制 artifacts 数量，避免大量 pickle 传输
        for pid in list(self.database.programs.keys())[:100]:
            artifacts = self.database.get_artifacts(pid)
            if artifacts:
                snapshot["artifacts"][pid] = artifacts
        return snapshot

    async def run_evolution(
        self,
        start_iteration: int,
        max_iterations: int,
        target_score: Optional[float] = None,
        checkpoint_callback=None,
    ):

        if not self.executor:
            raise RuntimeError("Process pool not started")

        total_iterations = start_iteration + max_iterations
        logger.info(
            f"Starting process-based evolution from iteration {start_iteration} "
            f"for {max_iterations} iterations (total: {total_iterations})"
        )

        pending_futures: Dict[int, Future] = {}
        island_pending: Dict[int, List[int]] = {i: [] for i in range(self.num_islands)}
        batch_size = min(self.num_workers * 2, max_iterations)

        batch_per_island = max(1, batch_size // self.num_islands) if batch_size > 0 else 0
        current_iteration = start_iteration

        # 初始化一批任务
        for island_id in range(self.num_islands):
            for _ in range(batch_per_island):
                if current_iteration < total_iterations:
                    future = self._submit_iteration(current_iteration, island_id)
                    if future:
                        pending_futures[current_iteration] = future
                        island_pending[island_id].append(current_iteration)
                    current_iteration += 1

        next_iteration = current_iteration
        completed_iterations = 0

        programs_per_island = max(1, max_iterations // (self.config.database.num_islands * 10))
        current_island_counter = 0

        early_stopping_enabled = self.config.early_stopping_patience is not None
        if early_stopping_enabled:
            best_score = float("-inf")
            iterations_without_improvement = 0
            logger.info(
                f"Early stopping enabled: patience={self.config.early_stopping_patience}, "
                f"threshold={self.config.convergence_threshold}, "
                f"metric={self.config.early_stopping_metric}"
            )
        else:
            logger.info("Early stopping disabled")

        # —— 放在 run_evolution(...) 里，主 while 前 —— #
        def _can_submit_more() -> bool:
            # 全局限流：总 pending 不超过 worker 数
            return len(pending_futures) < self.num_workers and not self.shutdown_event.is_set()

        def _submit_one_for_island(isl: int) -> bool:
            nonlocal next_iteration  # 需要修改外层的 next_iteration 计数
            if next_iteration >= total_iterations:
                return False
            fut = self._submit_iteration(next_iteration, isl)
            if not fut:
                return False
            pending_futures[next_iteration] = fut
            island_pending[isl].append(next_iteration)
            next_iteration += 1
            return True

        def _fill_pending_round_robin():
            """
            轮询各岛，直到 pending 数量达到 worker 上限或任务发完。
            这是我们推荐的“补货”函数——用它替代以前的 per-island 批量灌任务。
            """
            progressed = True
            while _can_submit_more() and next_iteration < total_iterations and progressed:
                progressed = False
                for isl in range(self.num_islands):
                    if not _can_submit_more() or next_iteration >= total_iterations:
                        break
                    if _submit_one_for_island(isl):
                        progressed = True

        # （可选）若你想保留之前的名字，就再加一个薄包装：
        def _resubmit_after_restart():
            _fill_pending_round_robin()

        while (
            pending_futures
            and completed_iterations < max_iterations
            and not self.shutdown_event.is_set()
        ):
            completed_iteration = None
            for it, fut in list(pending_futures.items()):
                if fut.done():
                    completed_iteration = it
                    break

            if completed_iteration is None:
                await asyncio.sleep(0.01)
                continue

            future = pending_futures.pop(completed_iteration)

            try:
                timeout_seconds = self.config.evaluator.timeout + 30
                result: SerializableResult = future.result(timeout=timeout_seconds)

                if result.error:
                    logger.warning(f"Iteration {completed_iteration} error: {result.error}")
                elif result.child_program_dict:
                    child_program = Program(**result.child_program_dict)

                    # 入库（继承 parent 的 island）
                    self.database.add(child_program, iteration=completed_iteration)

                    if result.artifacts:
                        self.database.store_artifacts(child_program.id, result.artifacts)

                    # ------- 兼容你的 evolution_tracer 参数命名（program / program_info 都支持）-------
                    if self.evolution_tracer and hasattr(self.evolution_tracer, "log_trace"):
                        try:
                            import inspect
                            sig = inspect.signature(self.evolution_tracer.log_trace)
                            accepted = set(sig.parameters.keys())

                            parent_program = self.database.get(result.parent_id) if result.parent_id else None
                            island_id = child_program.metadata.get("island", self.database.current_island)

                            # 构造 info 版本（轻量）
                            parent_info = None
                            if parent_program:
                                parent_info = {
                                    "id": parent_program.id,
                                    "metrics": parent_program.metrics,
                                    "generation": parent_program.generation,
                                    "metadata": parent_program.metadata,
                                }
                            child_info = {
                                "id": child_program.id,
                                "metrics": child_program.metrics,
                                "generation": child_program.generation,
                                "metadata": child_program.metadata,
                            }

                            payload = {
                                "iteration": completed_iteration,
                                "island_id": island_id,
                                "prompt": result.prompt,
                                "llm_response": result.llm_response,
                                "artifacts": result.artifacts,
                                "metadata": {
                                    "iteration_time": result.iteration_time,
                                    "changes": child_program.metadata.get("changes", ""),
                                },
                            }
                            # 两套字段名都尝试：有啥传啥
                            if "parent_program" in accepted and parent_program is not None:
                                payload["parent_program"] = parent_program
                            if "child_program" in accepted:
                                payload["child_program"] = child_program
                            if "parent_program_info" in accepted and parent_info is not None:
                                payload["parent_program_info"] = parent_info
                            if "child_program_info" in accepted:
                                payload["child_program_info"] = child_info

                            self.evolution_tracer.log_trace(**{k: v for k, v in payload.items() if k in accepted})
                        except Exception as te:
                            logger.debug(f"evolution_tracer.log_trace failed gracefully: {te}")

                    # 记录 prompt
                    if result.prompt:
                        self.database.log_prompt(
                            template_key=("full_rewrite_user" if not self.config.diff_based_evolution else "diff_user"),
                            program_id=child_program.id,
                            prompt=result.prompt,
                            responses=[result.llm_response] if result.llm_response else [],
                        )

                    # 岛屿轮换 & 迁移
                    if completed_iteration > start_iteration and current_island_counter >= programs_per_island:
                        self.database.next_island()
                        current_island_counter = 0
                        logger.debug(f"Switched to island {self.database.current_island}")

                    current_island_counter += 1
                    self.database.increment_island_generation()

                    if self.database.should_migrate():
                        logger.info(f"Performing migration at iteration {completed_iteration}")
                        self.database.migrate_programs()
                        self.database.log_island_status()

                    # 日志
                    logger.info(
                        f"Iteration {completed_iteration}: "
                        f"Program {child_program.id} (parent: {result.parent_id}) "
                        f"completed in {result.iteration_time:.2f}s"
                    )
                    if child_program.metrics:
                        metrics_str = ", ".join(
                            [f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                             for k, v in child_program.metrics.items()]
                        )
                        logger.info(f"Metrics: {metrics_str}")

                        if not hasattr(self, "_warned_about_combined_score"):
                            self._warned_about_combined_score = False
                        if "combined_score" not in child_program.metrics and not self._warned_about_combined_score:
                            avg_score = safe_numeric_average(child_program.metrics)
                            logger.warning(
                                f"⚠️  No 'combined_score' in metrics; using safe average ({avg_score:.4f}) for guidance. "
                                f"Consider returning a proper 'combined_score' in evaluator."
                            )
                            self._warned_about_combined_score = True

                    if self.database.best_program_id == child_program.id:
                        logger.info(f"🌟 New best solution found at iteration {completed_iteration}: {child_program.id}")

                    # Checkpoint
                    if completed_iteration > 0 and completed_iteration % self.config.checkpoint_interval == 0:
                        logger.info(f"Checkpoint interval reached at iteration {completed_iteration}")
                        self.database.log_island_status()
                        if checkpoint_callback:
                            checkpoint_callback(completed_iteration)

                    # 目标分数
                    if target_score is not None and child_program.metrics:
                        nums = [v for v in child_program.metrics.values() if isinstance(v, (int, float))]
                        if nums and sum(nums) / len(nums) >= target_score:
                            logger.info(f"Target score {target_score} reached at iteration {completed_iteration}")
                            break

                    # Early stopping
                    if early_stopping_enabled and child_program.metrics:
                        metric_name = self.config.early_stopping_metric
                        if metric_name in child_program.metrics:
                            current_score = child_program.metrics[metric_name]
                        elif metric_name == "combined_score":
                            current_score = safe_numeric_average(child_program.metrics)
                        else:
                            logger.warning(
                                f"Early stopping metric '{metric_name}' not found; using safe numeric average"
                            )
                            current_score = safe_numeric_average(child_program.metrics)

                        if isinstance(current_score, (int, float)):
                            improvement = current_score - best_score
                            if improvement >= self.config.convergence_threshold:
                                best_score = current_score
                                iterations_without_improvement = 0
                            else:
                                iterations_without_improvement += 1

                            if iterations_without_improvement >= self.config.early_stopping_patience:
                                self.early_stopping_triggered = True
                                logger.info(
                                    f"🛑 Early stopping at iteration {completed_iteration}: "
                                    f"No improvement for {iterations_without_improvement} iterations "
                                    f"(best: {best_score:.4f})"
                                )
                                break

            except FutureTimeoutError:
                logger.error(
                    f"⏰ Iteration {completed_iteration} timed out after {timeout_seconds}s "
                    f"(evaluator timeout: {self.config.evaluator.timeout}s + 30s buffer). "
                    f"Canceling future and restarting process pool to avoid worker leakage."
                )
                try:
                    future.cancel()
                except Exception:
                    pass
                try:
                    if self.executor:
                        self.executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                import multiprocessing as mp
                from concurrent.futures import ProcessPoolExecutor
                ctx = mp.get_context("spawn")
                self.executor = ProcessPoolExecutor(
                    max_workers=self.num_workers,
                    mp_context=ctx,
                    initializer=_worker_init,
                    initargs=(self._serialize_config(self.config), self.evaluation_file, dict(os.environ)),
                )
                logger.warning("🔁 Process pool restarted due to timeout. Resubmitting pending iterations.")
                # 3) 清空所有 pending 的映射与每个岛的队列，释放对“大快照”的引用
                pending_futures.clear()
                for isl in island_pending:
                    island_pending[isl].clear()

                # 4) 重新按配额投递新任务（见第 3 节“限流”改法）
                _resubmit_after_restart()

                # 5) 触发一次 GC，及时回收大对象
                import gc;
                gc.collect()
                continue
            except Exception as e:
                logger.error(f"Error processing result from iteration {completed_iteration}: {e}")

            completed_iterations += 1

            # 从岛屿队列移除
            for isl, arr in island_pending.items():
                if completed_iteration in arr:
                    arr.remove(completed_iteration)
                    break

            # 继续补任务
            for isl in range(self.num_islands):
                if (
                    len(island_pending[isl]) < batch_per_island
                    and next_iteration < total_iterations
                    and not self.shutdown_event.is_set()
                ):
                    fut = self._submit_iteration(next_iteration, isl)
                    if fut:
                        pending_futures[next_iteration] = fut
                        island_pending[isl].append(next_iteration)
                        next_iteration += 1
                        break

        # 收尾
        if self.shutdown_event.is_set():
            logger.info("Shutdown requested, canceling remaining evaluations...")
            for fut in pending_futures.values():
                fut.cancel()

        if self.early_stopping_triggered:
            logger.info("✅ Evolution completed - Early stopping triggered due to convergence")
        elif self.shutdown_event.is_set():
            logger.info("✅ Evolution completed - Shutdown requested")
        else:
            logger.info("✅ Evolution completed - Maximum iterations reached")

        return self.database.get_best_program()

    def _submit_iteration(self, iteration: int, island_id: Optional[int] = None) -> Optional[Future]:
        """Submit an iteration to the process pool, optionally pinned to a specific island"""
        try:
            target_island = island_id if island_id is not None else self.database.current_island

            parent, inspirations = self.database.sample_from_island(
                island_id=target_island,
                num_inspirations=self.config.prompt.num_top_programs
            )

            db_snapshot = self._create_database_snapshot()
            db_snapshot["sampling_island"] = target_island

            fut = self.executor.submit(
                _run_iteration_worker,
                iteration,
                db_snapshot,
                parent.id,
                [insp.id for insp in inspirations],
            )
            return fut

        except BrokenProcessPool as e:
            logger.error("A child process terminated abruptly, the process pool is not usable anymore: %s", e)
            # 尝试重建一次进程池并重试
            try:
                if self.executor:
                    self.executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            ctx = mp.get_context("spawn")
            self.executor = ProcessPoolExecutor(
                max_workers=self.num_workers,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(self._serialize_config(self.config), self.evaluation_file, dict(os.environ)),
            )
            try:
                fut = self.executor.submit(
                    _run_iteration_worker,
                    iteration,
                    self._create_database_snapshot(),
                    parent.id,
                    [insp.id for insp in inspirations],
                )
                return fut
            except Exception as e2:
                logger.error(f"Resubmission after pool rebuild failed: {e2}")
                return None
        except Exception as e:
            logger.error(f"Error submitting iteration {iteration}: {e}")
            return None
