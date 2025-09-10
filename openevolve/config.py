"""
Configuration handling for OpenEvolve
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class LLMModelConfig:
    api_base: str = None
    api_key: Optional[str] = None
    name: str = None
    weight: float = 1.0
    system_message: Optional[str] = None
    temperature: float = None
    top_p: float = None
    max_tokens: int = None
    timeout: int = None
    retries: int = None
    retry_delay: int = None
    random_seed: Optional[int] = None


@dataclass
class LLMConfig(LLMModelConfig):
    api_base: str = "https://api.openai.com/v1"
    system_message: Optional[str] = "system_message"
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 4096
    timeout: int = 60
    retries: int = 3
    retry_delay: int = 5

    models: List[LLMModelConfig] = field(default_factory=lambda: [
        LLMModelConfig(name="gpt-4o-mini", weight=0.8),
        LLMModelConfig(name="gpt-4o", weight=0.2),
    ])
    evaluator_models: List[LLMModelConfig] = field(default_factory=lambda: [])

    primary_model: str = None
    primary_model_weight: float = None
    secondary_model: str = None
    secondary_model_weight: float = None

    def __post_init__(self):
        if (self.primary_model or self.primary_model_weight) and len(self.models) < 1:
            self.models.append(LLMModelConfig())
        if self.primary_model:
            self.models[0].name = self.primary_model
        if self.primary_model_weight:
            self.models[0].weight = self.primary_model_weight

        if (self.secondary_model or self.secondary_model_weight) and len(self.models) < 2:
            self.models.append(LLMModelConfig())
        if self.secondary_model:
            self.models[1].name = self.secondary_model
        if self.secondary_model_weight:
            self.models[1].weight = self.secondary_model_weight

        if not self.evaluator_models or len(self.evaluator_models) < 1:
            self.evaluator_models = self.models.copy()

        shared_config = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "random_seed": self.random_seed,
        }
        self.update_model_params(shared_config)

    def update_model_params(self, args: Dict[str, Any], overwrite: bool = False) -> None:
        for model in self.models + self.evaluator_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)


# ---- Direction Feedback ----
@dataclass
class DirectionFeedbackWeights:
    score: float = 1.0
    params: float = 0.3
    latency_ms: float = 0.3
    flops: float = 0.2
    mem_mb: float = 0.2


@dataclass
class DirectionFeedbackConfig:
    enabled: bool = False
    frequency: int = 1
    k_window: int = 8
    ema_decay: float = 0.8
    stagnation_k: int = 6
    epsilon: float = 0.01
    source: Optional[str] = None
    weights: DirectionFeedbackWeights = field(default_factory=DirectionFeedbackWeights)

    warmup_k: int = 3
    max_df_lines: int = 12
    allowed_ops: List[str] = field(default_factory=lambda: [
        "tile_size ∈ {16,32}", "unroll ∈ {2,4,8}",
        "bias_outside_inner_loop", "accumulator_in_register",
        "addcmul_ (if allowed)", "vmap (if allowed)"
    ])
    forbidden_patterns: List[str] = field(default_factory=lambda: [
        "innermost loop over batch dimension",
        "python triple-for over non-contiguous memory"
    ])


@dataclass
class PromptConfig:
    save_prompts_text: Optional[bool] = False
    prompts_dir: Optional[str] = None
    template_dir: Optional[str] = None
    system_message: str = "system_message"
    evaluator_system_message: str = "evaluator_system_message"
    num_top_programs: int = 3
    num_diverse_programs: int = 2
    use_template_stochasticity: bool = True
    template_variations: Dict[str, List[str]] = field(default_factory=dict)
    use_meta_prompting: bool = False
    meta_prompt_weight: float = 0.1
    include_artifacts: bool = True
    max_artifact_bytes: int = 20 * 1024
    artifact_security_filter: bool = True
    suggest_simplification_after_chars: Optional[int] = 500
    include_changes_under_chars: Optional[int] = 100
    concise_implementation_max_lines: Optional[int] = 10
    comprehensive_implementation_min_lines: Optional[int] = 50
    code_length_threshold: Optional[int] = None


@dataclass
class DatabaseConfig:
    db_path: Optional[str] = None
    in_memory: bool = True
    log_prompts: bool = True
    population_size: int = 1000
    archive_size: int = 100
    num_islands: int = 5
    elite_selection_ratio: float = 0.1
    exploration_ratio: float = 0.2
    exploitation_ratio: float = 0.7
    diversity_metric: str = "edit_distance"
    feature_dimensions: List[str] = field(default_factory=lambda: ["complexity", "diversity"])
    feature_bins: Union[int, Dict[str, int]] = 10
    diversity_reference_size: int = 20
    migration_interval: int = 50
    migration_rate: float = 0.1
    random_seed: Optional[int] = 42
    artifacts_base_path: Optional[str] = None
    artifact_size_threshold: int = 32 * 1024
    cleanup_old_artifacts: bool = True
    artifact_retention_days: int = 30


@dataclass
class EvaluatorConfig:
    timeout: int = 300
    max_retries: int = 3
    memory_limit_mb: Optional[int] = None
    cpu_limit: Optional[float] = None
    cascade_evaluation: bool = True
    cascade_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9])
    parallel_evaluations: int = 1
    distributed: bool = False
    use_llm_feedback: bool = False
    llm_feedback_weight: float = 0.1
    enable_artifacts: bool = True
    max_artifact_storage: int = 100 * 1024 * 1024

    # ---- 新增：评测预算参数（可用来快速降低 timeout 概率）----
    max_steps: Optional[int] = None            # e.g. 20
    max_train_batches: Optional[int] = None    # e.g. 20
    max_eval_batches: Optional[int] = None     # e.g. 20
    train_subset: Optional[int] = None         # e.g. 2000 samples
    eval_subset: Optional[int] = None          # e.g. 2000 samples
    batch_size: Optional[int] = None           # override evaluator default


@dataclass
class Config:
    max_iterations: int = 10000
    checkpoint_interval: int = 100
    log_level: str = "INFO"
    log_dir: Optional[str] = None
    random_seed: Optional[int] = 42
    language: str = None

    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    direction_feedback: DirectionFeedbackConfig = field(default_factory=DirectionFeedbackConfig)

    diff_based_evolution: bool = True
    max_code_length: int = 10000
    # Early stopping settings
    early_stopping_patience: Optional[int] = None
    convergence_threshold: float = 0.001
    early_stopping_metric: str = "combined_score"
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        config = Config()
        for key, value in config_dict.items():
            if key not in ["llm", "prompt", "database", "evaluator",
                           "direction_feedback", "directional_feedback"] and hasattr(config, key):
                setattr(config, key, value)

        if "llm" in config_dict:
            llm_dict = config_dict["llm"]
            if "models" in llm_dict:
                llm_dict["models"] = [LLMModelConfig(**m) for m in llm_dict["models"]]
            if "evaluator_models" in llm_dict:
                llm_dict["evaluator_models"] = [LLMModelConfig(**m) for m in llm_dict["evaluator_models"]]
            config.llm = LLMConfig(**llm_dict)

        if "prompt" in config_dict:
            config.prompt = PromptConfig(**config_dict["prompt"])

        if "database" in config_dict:
            config.database = DatabaseConfig(**config_dict["database"])

        # Direction Feedback（别名+过滤）
        df_key = "direction_feedback" if "direction_feedback" in config_dict else \
                 ("directional_feedback" if "directional_feedback" in config_dict else None)
        if df_key:
            df_dict = dict(config_dict[df_key] or {})
            allowed = set(DirectionFeedbackConfig.__dataclass_fields__.keys())
            df_dict = {k: v for k, v in df_dict.items() if k in allowed}
            if "weights" in df_dict and isinstance(df_dict["weights"], dict):
                df_dict["weights"] = DirectionFeedbackWeights(**df_dict["weights"])
            config.direction_feedback = DirectionFeedbackConfig(**df_dict)

        if "evaluator" in config_dict:
            config.evaluator = EvaluatorConfig(**config_dict["evaluator"])

        if config.database.random_seed is None and config.random_seed is not None:
            config.database.random_seed = config.random_seed

        return config

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "checkpoint_interval": self.checkpoint_interval,
            "log_level": self.log_level,
            "log_dir": self.log_dir,
            "random_seed": self.random_seed,
            "llm": {
                "models": self.llm.models,
                "evaluator_models": self.llm.evaluator_models,
                "api_base": self.llm.api_base,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
                "max_tokens": self.llm.max_tokens,
                "timeout": self.llm.timeout,
                "retries": self.llm.retries,
                "retry_delay": self.llm.retry_delay,
            },
            "prompt": {
                "template_dir": self.prompt.template_dir,
                "system_message": self.prompt.system_message,
                "evaluator_system_message": self.prompt.evaluator_system_message,
                "num_top_programs": self.prompt.num_top_programs,
                "num_diverse_programs": self.prompt.num_diverse_programs,
                "use_template_stochasticity": self.prompt.use_template_stochasticity,
                "template_variations": self.prompt.template_variations,
            },
            "database": {
                "db_path": self.database.db_path,
                "in_memory": self.database.in_memory,
                "population_size": self.database.population_size,
                "archive_size": self.database.archive_size,
                "num_islands": self.database.num_islands,
                "elite_selection_ratio": self.database.elite_selection_ratio,
                "exploration_ratio": self.database.exploration_ratio,
                "exploitation_ratio": self.database.exploitation_ratio,
                "feature_dimensions": self.database.feature_dimensions,
                "feature_bins": self.database.feature_bins,
                "migration_interval": self.database.migration_interval,
                "migration_rate": self.database.migration_rate,
                "random_seed": self.database.random_seed,
                "log_prompts": self.database.log_prompts,
            },
            "evaluator": {
                "timeout": self.evaluator.timeout,
                "max_retries": self.evaluator.max_retries,
                "cascade_evaluation": self.evaluator.cascade_evaluation,
                "cascade_thresholds": self.evaluator.cascade_thresholds,
                "parallel_evaluations": self.evaluator.parallel_evaluations,
                "use_llm_feedback": self.evaluator.use_llm_feedback,
                "llm_feedback_weight": self.evaluator.llm_feedback_weight,
                "enable_artifacts": self.evaluator.enable_artifacts,
                "max_artifact_storage": self.evaluator.max_artifact_storage,
                "max_steps": self.evaluator.max_steps,
                "max_train_batches": self.evaluator.max_train_batches,
                "max_eval_batches": self.evaluator.max_eval_batches,
                "train_subset": self.evaluator.train_subset,
                "eval_subset": self.evaluator.eval_subset,
                "batch_size": self.evaluator.batch_size,
            },
            "diff_based_evolution": self.diff_based_evolution,
            "max_code_length": self.max_code_length,
            # Early stopping settings
            "early_stopping_patience": self.early_stopping_patience,
            "convergence_threshold": self.convergence_threshold,
            "early_stopping_metric": self.early_stopping_metric,
            "direction_feedback": {
                "enabled": self.direction_feedback.enabled,
                "frequency": self.direction_feedback.frequency,
                "k_window": self.direction_feedback.k_window,
                "ema_decay": self.direction_feedback.ema_decay,
                "stagnation_k": self.direction_feedback.stagnation_k,
                "epsilon": self.direction_feedback.epsilon,
                "source": self.direction_feedback.source,
                "warmup_k": self.direction_feedback.warmup_k,
                "max_df_lines": self.direction_feedback.max_df_lines,
                "allowed_ops": self.direction_feedback.allowed_ops,
                "forbidden_patterns": self.direction_feedback.forbidden_patterns,
                "weights": {
                    "score": self.direction_feedback.weights.score,
                    "params": self.direction_feedback.weights.params,
                    "latency_ms": self.direction_feedback.weights.latency_ms,
                    "flops": self.direction_feedback.weights.flops,
                    "mem_mb": self.direction_feedback.weights.mem_mb,
                },
            },
        }

    def to_yaml(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    if config_path and os.path.exists(config_path):
        config = Config.from_yaml(config_path)
    else:
        config = Config()
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
        config.llm.update_model_params({"api_key": api_key, "api_base": api_base})
    config.llm.update_model_params({"system_message": config.prompt.system_message})
    return config
