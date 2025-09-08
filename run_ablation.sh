#!/usr/bin/env bash
set -euo pipefail

# 避免 Windows 控制台编码问题（确保日志/表情不报 GBK 错）
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# 到脚本所在目录
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# 你的消融配置（按你图里的文件名来）
CONFIGS=(
  "configs/abl_single_island_config.yaml"
  "configs/abl_no_migration_config.yaml"
  "configs/abl_mig_fast_config.yaml"
  "configs/abl_mig_strong_config.yaml"
  "configs/abl_islands3_config.yaml"
  "configs/abl_islands10_config.yaml"
  "configs/abl_no_diff_config.yaml"
)

# 三个随机种子
SEEDS=(42 1337 2024)

mkdir -p runs configs/_tmp
SUMMARY="runs/ablation_summary.csv"
echo "config,seed,output_dir,combined_score,success_rate,value_score" > "$SUMMARY"

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for s in "${SEEDS[@]}"; do
    out="runs/${name}_s${s}"
    tmp="configs/_tmp/${name}_s${s}.yaml"
    mkdir -p "$out" "$(dirname "$tmp")"

    # 用当前种子生成临时 YAML：
    # 若原文件里已有 random_seed，就替换；否则在末尾追加
    if grep -qE '^random_seed:' "$cfg"; then
      sed -E "s/^random_seed:\s*[0-9]+/random_seed: ${s}/" "$cfg" > "$tmp"
    else
      cat "$cfg" > "$tmp"
      echo "" >> "$tmp"
      echo "random_seed: ${s}" >> "$tmp"
    fi

    echo ">>> [$(date '+%F %T')] RUN $name seed=$s"
    # 运行，同时用 tee 一边显示一边写日志（监督）
    ( time python openevolve-run.py \
        examples/function_minimization/initial_program.py \
        examples/function_minimization/evaluator.py \
        --config "$tmp" \
        --output "$out" ) 2>&1 | tee "$out/run.log"

    # 把最好结果摘出来追加到汇总 CSV（不用 jq，直接 python 读 JSON）
    if [ -f "$out/best/best_program_info.json" ]; then
      METRICS=$(python - "$out/best/best_program_info.json" <<'PY'
import json,sys
p=sys.argv[1]
d=json.load(open(p,'r',encoding='utf-8'))
m=d.get("metrics",{})
def g(k):
    v=m.get(k,"")
    return f"{v:.4f}" if isinstance(v,float) else str(v)
print(",".join([g("combined_score"), g("success_rate"), g("value_score")]))
PY
)
    else
      METRICS=",,"
    fi
    echo "${name},${s},${out},${METRICS}" | tee -a "$SUMMARY"
    echo ">>> DONE $name seed=$s"
  done
done

echo "All runs complete. Summary at $SUMMARY"
