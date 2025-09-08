"""
Process-based parallel controller for true parallelism
"""

import asyncio
import logging
import multiprocessing as mp
import pickle
import signal
import time
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase

logger = logging.getLogger(__name__)

# === Directional Feedback globals (safe imports) ===
try:
    from openevolve.metrics.target_space import make_target_vector, RunningStats
    from openevolve.direction import DirectionTracker
except Exception:
    # 允许在未合入这两个模块时正常运行（开关会禁用）
    make_target_vector = None
    RunningStats = None
    DirectionTracker = None

# worker 侧全局（在 _worker_init 里赋值）
_worker_direction_cfg = None
_worker_running_stats = None
_worker_dir_tracker = None



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


def _worker_init(config_dict: dict, evaluation_file: str) -> None:
    """Initialize worker process with necessary components"""
    global _worker_config
    global _worker_evaluation_file
    global _worker_evaluator
    global _worker_llm_ensemble
    global _worker_prompt_sampler
    # ✅ 统一在函数最前声明所有会赋值的全局变量

    global _worker_direction_cfg, _worker_running_stats, _worker_dir_tracker
    # Store config for later use
    # Reconstruct Config object from nested dictionaries
    from openevolve.config import (
        Config,
        DatabaseConfig,
        EvaluatorConfig,
        LLMConfig,
        PromptConfig,
        LLMModelConfig,
    )

    # Reconstruct model objects
    models = [LLMModelConfig(**m) for m in config_dict["llm"]["models"]]
    evaluator_models = [LLMModelConfig(**m) for m in config_dict["llm"]["evaluator_models"]]

    # Create LLM config with models
    llm_dict = config_dict["llm"].copy()
    llm_dict["models"] = models
    llm_dict["evaluator_models"] = evaluator_models
    llm_config = LLMConfig(**llm_dict)

    # Create other configs
    prompt_config = PromptConfig(**config_dict["prompt"])
    database_config = DatabaseConfig(**config_dict["database"])
    evaluator_config = EvaluatorConfig(**config_dict["evaluator"])

    _worker_config = Config(
        llm=llm_config,
        prompt=prompt_config,
        database=database_config,
        evaluator=evaluator_config,
        **{
            k: v
            for k, v in config_dict.items()
            if k not in ["llm", "prompt", "database", "evaluator"]
        },
    )
    _worker_evaluation_file = evaluation_file



    # These will be lazily initialized on first use
    _worker_evaluator = None
    _worker_llm_ensemble = None
    _worker_prompt_sampler = None
    # === Directional Feedback: store config & initialize placeholders ===

    _worker_direction_cfg = config_dict.get("direction_feedback", {}) or {}
    # 只有打开开关且模块可用才启用
    if _worker_direction_cfg.get("enabled", False) and RunningStats is not None:
        _worker_running_stats = RunningStats(
            mean={"combined_score":0, "params":0, "latency_ms":0, "flops":0, "mem_mb":0},
            std ={"combined_score":1, "params":1, "latency_ms":1, "flops":1, "mem_mb":1},
            count=0,
        )
    else:
        _worker_running_stats = None
    _worker_dir_tracker = None  # 真正创建放在 lazy 初始化


def _lazy_init_worker_components():
    """Lazily initialize expensive components on first use"""
    global _worker_evaluator
    global _worker_llm_ensemble
    global _worker_prompt_sampler
    global _worker_dir_tracker, _worker_direction_cfg  # ✅ 新增
    if _worker_llm_ensemble is None:
        from openevolve.llm.ensemble import LLMEnsemble

        _worker_llm_ensemble = LLMEnsemble(_worker_config.llm.models)

    if _worker_prompt_sampler is None:
        from openevolve.prompt.sampler import PromptSampler

        _worker_prompt_sampler = PromptSampler(_worker_config.prompt)

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
        # === Directional Feedback: lazy init DirectionTracker ===
        if _worker_dir_tracker is None and _worker_direction_cfg.get("enabled", False) and DirectionTracker is not None:
            _worker_dir_tracker = DirectionTracker(
                dim=5,  # 目标空间当前用5维：score/params/latency/flops/mem
                k_window=int(_worker_direction_cfg.get("k_window", 8)),
                ema_decay=float(_worker_direction_cfg.get("ema_decay", 0.8)),
            )


def _run_iteration_worker(
    iteration: int, db_snapshot: Dict[str, Any], parent_id: str, inspiration_ids: List[str]
) -> SerializableResult:
    """Run a single iteration in a worker process"""
    try:
        # Lazy initialization
        _lazy_init_worker_components()

        # Reconstruct programs from snapshot
        programs = {pid: Program(**prog_dict) for pid, prog_dict in db_snapshot["programs"].items()}

        parent = programs[parent_id]
        inspirations = [programs[pid] for pid in inspiration_ids if pid in programs]

        # Get parent artifacts if available
        parent_artifacts = db_snapshot["artifacts"].get(parent_id)

        # Get island-specific programs for context
        parent_island = parent.metadata.get("island", db_snapshot["current_island"])
        island_programs = [
            programs[pid] for pid in db_snapshot["islands"][parent_island] if pid in programs
        ]

        # Sort by metrics for top programs
        from openevolve.utils.metrics_utils import safe_numeric_average

        # island_programs.sort(
        #     key=lambda p: p.metrics.get("combined_score", safe_numeric_average(p.metrics)),
        #     reverse=True,
        # )
        # === PATCH BEGIN: 禁用“均值退化”，缺失 combined_score 视为 -inf ===
        def _score_key(p):
            m = p.metrics or {}
            return m["combined_score"] if "combined_score" in m else float("-inf")

        island_programs.sort(key=_score_key, reverse=True)

        # Use config values for limits instead of hardcoding
        island_top_programs = island_programs[
            : _worker_config.prompt.num_top_programs + _worker_config.prompt.num_diverse_programs
        ]
        island_previous_programs = island_programs[: _worker_config.prompt.num_top_programs]

        # Build prompt
        prompt = _worker_prompt_sampler.build_prompt(
            current_program=parent.code,
            parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[p.to_dict() for p in island_previous_programs],
            top_programs=[p.to_dict() for p in island_top_programs],
            inspirations=[p.to_dict() for p in inspirations],
            language=_worker_config.language,
            evolution_round=iteration,
            diff_based_evolution=_worker_config.diff_based_evolution,
            program_artifacts=parent_artifacts,
            parent_program_dict=parent.to_dict(),  # NEW: 供 Directional Feedback 使用
        )

        iteration_start = time.time()

        # Generate code modification (sync wrapper for async)
        llm_response = asyncio.run(
            _worker_llm_ensemble.generate_with_context(
                system_message=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}],
            )
        )

        # Parse response based on evolution mode
        if _worker_config.diff_based_evolution:
            from openevolve.utils.code_utils import extract_diffs, apply_diff, format_diff_summary

            diff_blocks = extract_diffs(llm_response)
            if not diff_blocks:
                return SerializableResult(
                    error=f"No valid diffs found in response", iteration=iteration
                )

            child_code = apply_diff(parent.code, llm_response)
            changes_summary = format_diff_summary(diff_blocks)
        else:
            from openevolve.utils.code_utils import parse_full_rewrite

            new_code = parse_full_rewrite(llm_response, _worker_config.language)
            if not new_code:
                return SerializableResult(
                    error=f"No valid code found in response", iteration=iteration
                )

            child_code = new_code
            changes_summary = "Full rewrite"

        # Check code length
        if len(child_code) > _worker_config.max_code_length:
            return SerializableResult(
                error=f"Generated code exceeds maximum length ({len(child_code)} > {_worker_config.max_code_length})",
                iteration=iteration,
            )

        # Evaluate the child program
        import uuid

        child_id = str(uuid.uuid4())
        child_metrics = asyncio.run(_worker_evaluator.evaluate_program(child_code, child_id))
        # === PATCH BEGIN: 规范化 child_metrics，缺失分数时强制极低分 ===
        try:
            if "combined_score" not in child_metrics:
                # 不再用均值兜底，直接判极低分并打标记
                child_metrics = dict(child_metrics)  # 复制，避免影响底层返回
                child_metrics["combined_score"] = -1e9
                child_metrics["invalid_missing_score"] = 1.0
                logger.warning(
                    "Evaluator returned no combined_score; forcing very low score for child %s",
                    child_id,
                )
        except Exception as _e:
            logger.warning("Failed to normalize child_metrics for %s: %s", child_id, _e)
        # === PATCH END ===

        # === Directional Feedback: compute vectors & slope (before creating child Program) ===
        dir_md = {}
        try:
            if (
                _worker_direction_cfg.get("enabled", False)
                and make_target_vector is not None
                and _worker_running_stats is not None
                and _worker_dir_tracker is not None
            ):
                weights = _worker_direction_cfg.get("weights", {}) or {}
                # 兼容字段名（latency/latency_ms, flops/FLOPs 等在 target_space 里已做兼容）
                v_parent = make_target_vector(parent.metrics, _worker_running_stats, weights)
                v_child  = make_target_vector(child_metrics,  _worker_running_stats, weights)

                # 是否改进：优先按 combined_score，否则退化为平均数
                from openevolve.utils.metrics_utils import safe_numeric_average
                # === PATCH BEGIN: DF 改进判定也禁用“均值退化” ===
                parent_cs = parent.metrics["combined_score"] if "combined_score" in parent.metrics else float("-inf")
                child_cs = child_metrics["combined_score"] if "combined_score" in child_metrics else float("-inf")
                improved = child_cs > parent_cs
                # === PATCH END ===

                dv, slope, slope_avg, st = _worker_dir_tracker.update(
                    island_id=parent_island, v_parent=v_parent, v_child=v_child, improved=improved
                )

                stagnation_k = int(_worker_direction_cfg.get("stagnation_k", 6))
                epsilon = float(_worker_direction_cfg.get("epsilon", 0.01))
                stagnating = (len(st.slopes) >= stagnation_k) and (slope_avg < epsilon)

                dir_md = {
                    "target_vec": v_child.tolist(),
                    "delta_vec": dv.tolist(),
                    "slope_on_baseline": float(slope),
                    "slope_mean_k": float(slope_avg),
                    "island_id": int(parent_island),
                    "stagnating": bool(stagnating),
                }
                logger.info(
                    f"[dirfb] island={parent_island} slope={slope:.3f} mean_k={slope_avg:.3f} stagnating={stagnating}"
                )
        except Exception as _e:
            logger.warning(f"[dirfb] skipped due to error: {_e}")


        # Get artifacts
        artifacts = _worker_evaluator.get_pending_artifacts(child_id)

        # Create child program
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
                # ↓↓↓ 新增：把方向反馈的度量写进去（可能为空 dict）
                **dir_md,
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

    def __init__(self, config: Config, evaluation_file: str, database: ProgramDatabase):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database

        self.executor: Optional[ProcessPoolExecutor] = None
        self.shutdown_event = mp.Event()

        # Number of worker processes
        self.num_workers = config.evaluator.parallel_evaluations

        logger.info(f"Initialized process parallel controller with {self.num_workers} workers")

    def _serialize_config(self, config: Config) -> dict:
        """Serialize config object to a dictionary that can be pickled"""
        # Manual serialization to handle nested objects properly
        cfg_dict = {
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
        }

        # ✅ 单独加一项 direction_feedback，不塞进 Config
        if hasattr(config, "direction_feedback"):
            cfg_dict["direction_feedback"] = config.direction_feedback
        return cfg_dict

    def start(self) -> None:
        """Start the process pool"""
        # Convert config to dict for pickling
        # We need to be careful with nested dataclasses
        config_dict = self._serialize_config(self.config)

        # Create process pool with initializer
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_worker_init,
            initargs=(config_dict, self.evaluation_file),
        )

        logger.info(f"Started process pool with {self.num_workers} processes")

    def stop(self) -> None:
        """Stop the process pool"""
        self.shutdown_event.set()

        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

        logger.info("Stopped process pool")

    def request_shutdown(self) -> None:
        """Request graceful shutdown"""
        logger.info("Graceful shutdown requested...")
        self.shutdown_event.set()

    def _create_database_snapshot(self) -> Dict[str, Any]:
        """Create a serializable snapshot of the database state"""
        # Only include necessary data for workers
        snapshot = {
            "programs": {pid: prog.to_dict() for pid, prog in self.database.programs.items()},
            "islands": [list(island) for island in self.database.islands],
            "current_island": self.database.current_island,
            "artifacts": {},  # Will be populated selectively
        }

        # Include artifacts for programs that might be selected
        # IMPORTANT: This limits artifacts (execution outputs/errors) to first 100 programs only.
        # This does NOT affect program code - all programs are fully serialized above.
        # With max_artifact_bytes=20KB and population_size=1000, artifacts could be 20MB total,
        # which would significantly slow worker process initialization. The limit of 100 keeps
        # artifact data under 2MB while still providing execution context for recent programs.
        # Workers can still evolve properly as they have access to ALL program code.
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
        """Run evolution with process-based parallelism"""
        if not self.executor:
            raise RuntimeError("Process pool not started")

        total_iterations = start_iteration + max_iterations

        logger.info(
            f"Starting process-based evolution from iteration {start_iteration} "
            f"for {max_iterations} iterations (total: {total_iterations})"
        )

        # Track pending futures
        pending_futures: Dict[int, Future] = {}
        batch_size = min(self.num_workers * 2, max_iterations)

        # Submit initial batch
        for i in range(start_iteration, min(start_iteration + batch_size, total_iterations)):
            future = self._submit_iteration(i)
            if future:
                pending_futures[i] = future

        next_iteration = start_iteration + batch_size
        completed_iterations = 0

        # Island management
        programs_per_island = max(1, max_iterations // (self.config.database.num_islands * 10))
        current_island_counter = 0

        # Process results as they complete
        while (
            pending_futures
            and completed_iterations < max_iterations
            and not self.shutdown_event.is_set()
        ):
            # Find completed futures
            completed_iteration = None
            for iteration, future in list(pending_futures.items()):
                if future.done():
                    completed_iteration = iteration
                    break

            if completed_iteration is None:
                await asyncio.sleep(0.01)
                continue

            # Process completed result
            future = pending_futures.pop(completed_iteration)

            try:
                result = future.result()

                if result.error:
                    logger.warning(f"Iteration {completed_iteration} error: {result.error}")
                elif result.child_program_dict:
                    # Reconstruct program from dict
                    child_program = Program(**result.child_program_dict)

                    # Add to database
                    self.database.add(child_program, iteration=completed_iteration)

                    # Store artifacts
                    if result.artifacts:
                        self.database.store_artifacts(child_program.id, result.artifacts)

                    # Log prompts
                    if result.prompt:
                        self.database.log_prompt(
                            template_key=(
                                "full_rewrite_user"
                                if not self.config.diff_based_evolution
                                else "diff_user"
                            ),
                            program_id=child_program.id,
                            prompt=result.prompt,
                            responses=[result.llm_response] if result.llm_response else [],
                        )

                    # Island management
                    if (
                        completed_iteration > start_iteration
                        and current_island_counter >= programs_per_island
                    ):
                        self.database.next_island()
                        current_island_counter = 0
                        logger.debug(f"Switched to island {self.database.current_island}")

                    current_island_counter += 1
                    self.database.increment_island_generation()

                    # Check migration
                    if self.database.should_migrate():
                        logger.info(f"Performing migration at iteration {completed_iteration}")
                        self.database.migrate_programs()
                        self.database.log_island_status()

                    # Log progress
                    logger.info(
                        f"Iteration {completed_iteration}: "
                        f"Program {child_program.id} "
                        f"(parent: {result.parent_id}) "
                        f"completed in {result.iteration_time:.2f}s"
                    )

                    if child_program.metrics:
                        metrics_str = ", ".join(
                            [
                                f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                for k, v in child_program.metrics.items()
                            ]
                        )
                        logger.info(f"Metrics: {metrics_str}")

                        # Check if this is the first program without combined_score
                        if not hasattr(self, "_warned_about_combined_score"):
                            self._warned_about_combined_score = False

                        if (
                            "combined_score" not in child_program.metrics
                            and not self._warned_about_combined_score
                        ):
                            from openevolve.utils.metrics_utils import safe_numeric_average

                            avg_score = safe_numeric_average(child_program.metrics)
                            logger.warning(
                                f"⚠️  No 'combined_score' metric found in evaluation results. "
                                f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                                f"For better evolution results, please modify your evaluator to return a 'combined_score' "
                                f"metric that properly weights different aspects of program performance."
                            )
                            self._warned_about_combined_score = True

                    # Check for new best
                    if self.database.best_program_id == child_program.id:
                        logger.info(
                            f"🌟 New best solution found at iteration {completed_iteration}: "
                            f"{child_program.id}"
                        )

                    # Checkpoint callback
                    # Don't checkpoint at iteration 0 (that's just the initial program)
                    if (
                        completed_iteration > 0
                        and completed_iteration % self.config.checkpoint_interval == 0
                    ):
                        logger.info(
                            f"Checkpoint interval reached at iteration {completed_iteration}"
                        )
                        self.database.log_island_status()
                        if checkpoint_callback:
                            checkpoint_callback(completed_iteration)

                    # Check target score
                    if target_score is not None and child_program.metrics:
                        numeric_metrics = [
                            v for v in child_program.metrics.values() if isinstance(v, (int, float))
                        ]
                        if numeric_metrics:
                            avg_score = sum(numeric_metrics) / len(numeric_metrics)
                            if avg_score >= target_score:
                                logger.info(
                                    f"Target score {target_score} reached at iteration {completed_iteration}"
                                )
                                break

            except Exception as e:
                logger.error(f"Error processing result from iteration {completed_iteration}: {e}")

            completed_iterations += 1

            # Submit next iteration
            if next_iteration < total_iterations and not self.shutdown_event.is_set():
                future = self._submit_iteration(next_iteration)
                if future:
                    pending_futures[next_iteration] = future
                    next_iteration += 1

        # Handle shutdown
        if self.shutdown_event.is_set():
            logger.info("Shutdown requested, canceling remaining evaluations...")
            for future in pending_futures.values():
                future.cancel()

        logger.info("Evolution completed")

        return self.database.get_best_program()

    def _submit_iteration(self, iteration: int) -> Optional[Future]:
        """Submit an iteration to the process pool"""
        try:
            # Sample parent and inspirations
            parent, inspirations = self.database.sample()

            # Create database snapshot
            db_snapshot = self._create_database_snapshot()

            # Submit to process pool
            future = self.executor.submit(
                _run_iteration_worker,
                iteration,
                db_snapshot,
                parent.id,
                [insp.id for insp in inspirations],
            )

            return future

        except Exception as e:
            logger.error(f"Error submitting iteration {iteration}: {e}")
            return None
