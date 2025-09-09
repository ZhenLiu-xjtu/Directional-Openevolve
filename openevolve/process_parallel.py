"""
Process-based parallel controller for true parallelism
"""

import asyncio
import logging
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase

dc_asdict = asdict
logger = logging.getLogger(__name__)

try:
    from openevolve.metrics.target_space import make_target_vector, RunningStats
    from openevolve.direction import DirectionTracker
except Exception:
    make_target_vector = None
    RunningStats = None
    DirectionTracker = None


def _fallback_target_vector(metrics: dict, weights: dict) -> List[float]:
    score = float(metrics.get("combined_score", -1e9))
    macs  = float(metrics.get("macs", metrics.get("flops", 0.0)))
    params = float(metrics.get("params", 0.0))
    latency_ms = float(metrics.get("latency_ms", metrics.get("infer_time_s", 0.0) * 1000.0))
    mem_mb = float(metrics.get("mem_mb", 0.0))
    w = weights or {}
    return [
        (score)        * float(w.get("score", 1.0)),
        (-macs)        * float(w.get("flops", 0.2)),
        (-params)      * float(w.get("params", 0.3)),
        (-latency_ms)  * float(w.get("latency_ms", 0.3)),
        (-mem_mb)      * float(w.get("mem_mb", 0.2)),
    ]


_worker_direction_cfg = None
_worker_running_stats = None
_worker_dir_tracker = None


@dataclass
class SerializableResult:
    child_program_dict: Optional[Dict[str, Any]] = None
    parent_id: Optional[str] = None
    iteration_time: float = 0.0
    prompt: Optional[Dict[str, str]] = None
    llm_response: Optional[str] = None
    artifacts: Optional[Dict[str, Any]] = None
    iteration: int = 0
    error: Optional[str] = None


def _worker_init(config_dict: dict, evaluation_file: str) -> None:
    global _worker_config, _worker_evaluation_file, _worker_evaluator
    global _worker_llm_ensemble, _worker_prompt_sampler
    global _worker_direction_cfg, _worker_running_stats, _worker_dir_tracker

    from openevolve.config import (
        Config, DatabaseConfig, EvaluatorConfig, LLMConfig, PromptConfig, LLMModelConfig, DirectionFeedbackConfig
    )

    models = [LLMModelConfig(**m) for m in config_dict["llm"]["models"]]
    evaluator_models = [LLMModelConfig(**m) for m in config_dict["llm"]["evaluator_models"]]

    llm_dict = config_dict["llm"].copy()
    llm_dict["models"] = models
    llm_dict["evaluator_models"] = evaluator_models
    llm_config = LLMConfig(**llm_dict)

    prompt_config = PromptConfig(**config_dict["prompt"])
    database_config = DatabaseConfig(**config_dict["database"])
    evaluator_config = EvaluatorConfig(**config_dict["evaluator"])

    raw_df = config_dict.get("direction_feedback", None)
    if isinstance(raw_df, dict):
        df_obj = DirectionFeedbackConfig(**raw_df)
    elif isinstance(raw_df, DirectionFeedbackConfig):
        df_obj = raw_df
    else:
        df_obj = DirectionFeedbackConfig()

    _worker_direction_cfg = dc_asdict(df_obj)

    _worker_config = Config(
        llm=llm_config,
        prompt=prompt_config,
        database=database_config,
        evaluator=evaluator_config,
        direction_feedback=df_obj,
        **{k: v for k, v in config_dict.items() if k not in ["llm","prompt","database","evaluator","direction_feedback"]},
    )
    _worker_evaluation_file = evaluation_file

    _worker_evaluator = None
    _worker_llm_ensemble = None
    _worker_prompt_sampler = None

    if _worker_direction_cfg.get("enabled", False) and RunningStats is not None:
        _worker_running_stats = RunningStats(
            mean={"combined_score": 0, "params": 0, "latency_ms": 0, "flops": 0, "mem_mb": 0},
            std={"combined_score": 1, "params": 1, "latency_ms": 1, "flops": 1, "mem_mb": 1},
            count=0,
        )
    else:
        _worker_running_stats = None

    _worker_dir_tracker = None


def _lazy_init_worker_components():
    global _worker_evaluator, _worker_llm_ensemble, _worker_prompt_sampler
    global _worker_dir_tracker, _worker_direction_cfg

    if _worker_llm_ensemble is None:
        from openevolve.llm.ensemble import LLMEnsemble
        _worker_llm_ensemble = LLMEnsemble(_worker_config.llm.models)

    if _worker_prompt_sampler is None:
        from openevolve.prompt.sampler import PromptSampler
        _worker_prompt_sampler = PromptSampler(_worker_config.prompt, _worker_config.direction_feedback)

    if _worker_evaluator is None:
        from openevolve.evaluator import Evaluator
        from openevolve.llm.ensemble import LLMEnsemble
        from openevolve.prompt.sampler import PromptSampler

        evaluator_llm = LLMEnsemble(_worker_config.llm.evaluator_models)
        evaluator_prompt = PromptSampler(_worker_config.prompt)
        evaluator_prompt.set_templates("evaluator_system_message")

        _worker_evaluator = Evaluator(
            _worker_config.evaluator,
            _worker_evaluation_file,
            evaluator_llm,
            evaluator_prompt,
            database=None,
        )

        if _worker_dir_tracker is None and _worker_direction_cfg.get("enabled", False) and DirectionTracker is not None:
            _worker_dir_tracker = DirectionTracker(
                dim=5,
                k_window=int(_worker_direction_cfg.get("k_window", 8)),
                ema_decay=float(_worker_direction_cfg.get("ema_decay", 0.8)),
            )


def _run_iteration_worker(iteration: int, db_snapshot: Dict[str, Any], parent_id: str, inspiration_ids: List[str]) -> SerializableResult:
    try:
        _lazy_init_worker_components()

        programs = {pid: Program(**prog_dict) for pid, prog_dict in db_snapshot["programs"].items()}
        parent = programs[parent_id]
        inspirations = [programs[pid] for pid in inspiration_ids if pid in programs]

        parent_artifacts = db_snapshot["artifacts"].get(parent_id)
        parent_island = parent.metadata.get("island", db_snapshot["current_island"])
        island_programs = [programs[pid] for pid in db_snapshot["islands"][parent_island] if pid in programs]

        def _score_key(p):
            m = p.metrics or {}
            return m["combined_score"] if "combined_score" in m else float("-inf")
        island_programs.sort(key=_score_key, reverse=True)

        island_top_programs = island_programs[: _worker_config.prompt.num_top_programs + _worker_config.prompt.num_diverse_programs]
        island_previous_programs = island_programs[: _worker_config.prompt.num_top_programs]

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
            parent_program_dict=parent.to_dict(),
        )

        # ---- 评测预算：写入环境变量，供外部 evaluator.py 读取（不改 initial_program）----
        import os
        ev = _worker_config.evaluator
        if ev.max_steps is not None:         os.environ["OE_MAX_STEPS"] = str(int(ev.max_steps))
        if ev.max_train_batches is not None: os.environ["OE_MAX_TRAIN_BATCHES"] = str(int(ev.max_train_batches))
        if ev.max_eval_batches is not None:  os.environ["OE_MAX_EVAL_BATCHES"] = str(int(ev.max_eval_batches))
        if ev.train_subset is not None:      os.environ["OE_TRAIN_SUBSET"] = str(int(ev.train_subset))
        if ev.eval_subset is not None:       os.environ["OE_EVAL_SUBSET"] = str(int(ev.eval_subset))
        if ev.batch_size is not None:        os.environ["OE_BATCH_SIZE"] = str(int(ev.batch_size))

        iteration_start = time.time()

        llm_response = asyncio.run(
            _worker_llm_ensemble.generate_with_context(
                system_message=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}],
            )
        )

        if _worker_config.diff_based_evolution:
            from openevolve.utils.code_utils import extract_diffs, apply_diff, format_diff_summary
            diff_blocks = extract_diffs(llm_response)
            if not diff_blocks:
                return SerializableResult(error="No valid diffs found in response", iteration=iteration)
            child_code = apply_diff(parent.code, llm_response)
            changes_summary = format_diff_summary(diff_blocks)
        else:
            from openevolve.utils.code_utils import parse_full_rewrite
            new_code = parse_full_rewrite(llm_response, _worker_config.language)
            if not new_code:
                return SerializableResult(error="No valid code found in response", iteration=iteration)
            child_code = new_code
            changes_summary = "Full rewrite"

        if len(child_code) > _worker_config.max_code_length:
            return SerializableResult(
                error=f"Generated code exceeds maximum length ({len(child_code)} > {_worker_config.max_code_length})",
                iteration=iteration,
            )

        import uuid
        child_id = str(uuid.uuid4())
        child_metrics = asyncio.run(_worker_evaluator.evaluate_program(child_code, child_id))

        try:
            if "combined_score" not in child_metrics:
                child_metrics = dict(child_metrics)
                child_metrics["combined_score"] = -1e9
                child_metrics["invalid_missing_score"] = 1.0
                logger.warning("Evaluator returned no combined_score; forcing very low score for child %s", child_id)
        except Exception as _e:
            logger.warning("Failed to normalize child_metrics for %s: %s", child_id, _e)

        dir_md = {}
        try:
            if _worker_direction_cfg.get("enabled", False):
                trained_ok = bool(child_metrics.get("trained", True))
                has_score = ("combined_score" in child_metrics) and ("combined_score" in parent.metrics)
                warmup_k  = int(_worker_direction_cfg.get("warmup_k", 3))

                if "latency_ms" not in child_metrics and "infer_time_s" in child_metrics:
                    child_metrics = dict(child_metrics)
                    child_metrics["latency_ms"] = child_metrics["infer_time_s"] * 1000.0

                if (iteration < warmup_k) or (not trained_ok) or (not has_score) or (_worker_dir_tracker is None):
                    dir_md = {
                        "island_id": int(parent_island),
                        "warmup": iteration < warmup_k,
                        "invalid": (not trained_ok) or (not has_score) or (_worker_dir_tracker is None),
                    }
                else:
                    weights = _worker_direction_cfg.get("weights", {}) or {}
                    if (make_target_vector is not None) and (_worker_running_stats is not None):
                        v_parent = make_target_vector(parent.metrics, _worker_running_stats, weights)
                        v_child  = make_target_vector(child_metrics, _worker_running_stats, weights)
                    else:
                        v_parent = _fallback_target_vector(parent.metrics, weights)
                        v_child  = _fallback_target_vector(child_metrics, weights)

                    improved = (child_metrics["combined_score"] > parent.metrics["combined_score"])
                    dv, slope, slope_avg, st = _worker_dir_tracker.update(
                        island_id=parent_island, v_parent=v_parent, v_child=v_child, improved=improved
                    )

                    stagnation_k = int(_worker_direction_cfg.get("stagnation_k", 6))
                    epsilon = float(_worker_direction_cfg.get("epsilon", 0.01))
                    stagnating = (len(st.slopes) >= stagnation_k) and (slope_avg < epsilon)

                    def _tolist(x): return x if isinstance(x, list) else getattr(x, "tolist", lambda: x)()
                    dir_md = {
                        "target_vec": _tolist(v_child),
                        "delta_vec":  _tolist(dv),
                        "slope_on_baseline": float(slope),
                        "slope_mean_k": float(slope_avg),
                        "island_id": int(parent_island),
                        "stagnating": bool(stagnating),
                    }
                    logger.info(f"[dirfb] isl={parent_island} slope={slope:.3f} mean={slope_avg:.3f} stagnating={stagnating}")
        except Exception as _e:
            logger.warning(f"[dirfb] skipped due to error: {_e}")

        artifacts = _worker_evaluator.get_pending_artifacts(child_id)

        child_program = Program(
            id=child_id,
            code=child_code,
            language=_worker_config.language,
            parent_id=parent.id,
            generation=parent.generation + 1,
            metrics=child_metrics,
            iteration_found=iteration,
            metadata={"changes": changes_summary, "parent_metrics": parent.metrics, "island": parent_island, **dir_md},
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
    def __init__(self, config: Config, evaluation_file: str, database: ProgramDatabase):
        self.config = config
        self.evaluation_file = evaluation_file
        self.database = database
        self.executor: Optional[ProcessPoolExecutor] = None
        self.shutdown_event = mp.Event()
        self.num_workers = config.evaluator.parallel_evaluations
        logger.info(f"Initialized process parallel controller with {self.num_workers} workers")

    def _serialize_config(self, config: Config) -> dict:
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
        if hasattr(config, "direction_feedback") and config.direction_feedback is not None:
            try:
                cfg_dict["direction_feedback"] = dc_asdict(config.direction_feedback)
            except Exception:
                df = config.direction_feedback
                cfg_dict["direction_feedback"] = getattr(df, "__dict__", {})
        return cfg_dict

    def start(self) -> None:
        config_dict = self._serialize_config(self.config)
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            initializer=_worker_init,
            initargs=(config_dict, self.evaluation_file),
        )
        logger.info(f"Started process pool with {self.num_workers} processes")

    def stop(self) -> None:
        self.shutdown_event.set()
        if self.executor:
            self.executor.shutdown(wait=True)
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
            "artifacts": {},
        }
        for pid in list(self.database.programs.keys())[:100]:
            artifacts = self.database.get_artifacts(pid)
            if artifacts:
                snapshot["artifacts"][pid] = artifacts
        return snapshot

    async def run_evolution(self, start_iteration: int, max_iterations: int, target_score: Optional[float] = None, checkpoint_callback=None):
        if not self.executor:
            raise RuntimeError("Process pool not started")

        total_iterations = start_iteration + max_iterations
        logger.info(f"Starting process-based evolution from iteration {start_iteration} for {max_iterations} iterations (total: {total_iterations})")

        pending_futures: Dict[int, Future] = {}
        batch_size = min(self.num_workers * 2, max_iterations)

        for i in range(start_iteration, min(start_iteration + batch_size, total_iterations)):
            future = self._submit_iteration(i)
            if future:
                pending_futures[i] = future

        next_iteration = start_iteration + batch_size
        completed_iterations = 0
        programs_per_island = max(1, max_iterations // (self.config.database.num_islands * 10))
        current_island_counter = 0

        while pending_futures and completed_iterations < max_iterations and not self.shutdown_event.is_set():
            completed_iteration = None
            for iteration, future in list(pending_futures.items()):
                if future.done():
                    completed_iteration = iteration
                    break
            if completed_iteration is None:
                await asyncio.sleep(0.01)
                continue

            future = pending_futures.pop(completed_iteration)
            try:
                result = future.result()

                if result.error:
                    logger.warning(f"Iteration {completed_iteration} error: {result.error}")
                elif result.child_program_dict:
                    child_program = Program(**result.child_program_dict)
                    child_program.metadata = child_program.metadata or {}
                    self.database.add(child_program, iteration=completed_iteration)

                    if result.artifacts:
                        self.database.store_artifacts(child_program.id, result.artifacts)

                    if result.prompt:
                        self.database.log_prompt(
                            template_key=("full_rewrite_user" if not self.config.diff_based_evolution else "diff_user"),
                            program_id=child_program.id,
                            prompt=result.prompt,
                            responses=[result.llm_response] if result.llm_response else [],
                        )

                    if (completed_iteration > start_iteration and current_island_counter >= programs_per_island):
                        self.database.next_island()
                        current_island_counter = 0
                        logger.debug(f"Switched to island {self.database.current_island}")

                    current_island_counter += 1
                    self.database.increment_island_generation()

                    if self.database.should_migrate():
                        logger.info(f"Performing migration at iteration {completed_iteration}")
                        self.database.migrate_programs()
                        self.database.log_island_status()

                    logger.info(f"Iteration {completed_iteration}: Program {child_program.id} (parent: {result.parent_id}) completed in {result.iteration_time:.2f}s")

                    if child_program.metrics:
                        metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                                                 for k, v in child_program.metrics.items()])
                        logger.info(f"Metrics: {metrics_str}")
                        if self.database.best_program_id == child_program.id:
                            logger.info(f"🌟 New best solution found at iteration {completed_iteration}: {child_program.id}")

                    if (completed_iteration > 0 and completed_iteration % self.config.checkpoint_interval == 0):
                        logger.info(f"Checkpoint interval reached at iteration {completed_iteration}")
                        self.database.log_island_status()
                        if checkpoint_callback:
                            checkpoint_callback(completed_iteration)

                    if target_score is not None and child_program.metrics:
                        numeric_metrics = [v for v in child_program.metrics.values() if isinstance(v, (int, float))]
                        if numeric_metrics:
                            avg_score = sum(numeric_metrics) / len(numeric_metrics)
                            if avg_score >= target_score:
                                logger.info(f"Target score {target_score} reached at iteration {completed_iteration}")
                                break

            except Exception as e:
                logger.error(f"Error processing result from iteration {completed_iteration}: {e}")

            completed_iterations += 1

            if next_iteration < total_iterations and not self.shutdown_event.is_set():
                future = self._submit_iteration(next_iteration)
                if future:
                    pending_futures[next_iteration] = future
                    next_iteration += 1

        if self.shutdown_event.is_set():
            logger.info("Shutdown requested, canceling remaining evaluations...")
            for future in pending_futures.values():
                future.cancel()

        logger.info("Evolution completed")
        return self.database.get_best_program()

    def _submit_iteration(self, iteration: int) -> Optional[Future]:
        try:
            parent, inspirations = self.database.sample()
            db_snapshot = self._create_database_snapshot()
            future = self.executor.submit(_run_iteration_worker, iteration, db_snapshot, parent.id, [insp.id for insp in inspirations])
            return future
        except Exception as e:
            logger.error(f"Error submitting iteration {iteration}: {e}")
            return None
