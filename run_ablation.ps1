# run_ablation.ps1
# Windows PowerShell 5+ 兼容
param(
  # 要跑的配置文件；不传则自动抓取 .\configs\abl_*_config.yaml
  [string[]] $Configs,

  # 随机种子
  [int[]]    $Seeds = @(42,123,2025),

  # 例子：函数最小化
  [string]   $InitialProgram = "examples\function_minimization\initial_program.py",
  [string]   $Evaluator      = "examples\function_minimization\evaluator.py",

  # Python 可执行文件
  [string]   $PythonExe = "python"
)

# 如果未显式给出 Configs，则自动收集所有消融配置
if (-not $Configs -or $Configs.Count -eq 0) {
  $Configs = Get-ChildItem -Path ".\configs" -Filter "abl_*_config.yaml" | ForEach-Object { $_.FullName }
}

# 控制台与 Python 统一用 UTF-8，避免 GBK 编码导致的 emoji/日志报错
try {
  [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
  $env:PYTHONIOENCODING = "utf-8"
  $env:PYTHONUTF8       = "1"
} catch { }

# 单次运行
function Invoke-OpenEvolveJob {
  param(
    [Parameter(Mandatory=$true)][string] $Cfg,
    [Parameter(Mandatory=$true)][int]    $Seed
  )

  $name   = [System.IO.Path]::GetFileNameWithoutExtension($Cfg)
  $runDir = Join-Path "runs" ("{0}_s{1}" -f $name, $Seed)
  New-Item -ItemType Directory -Force -Path $runDir | Out-Null

  # 复制配置并在末尾追加随机种子（同名键以最后一条为准）
  $cfgUsed = Join-Path $runDir "config_used.yaml"
  Copy-Item -Path $Cfg -Destination $cfgUsed -Force
  Add-Content -Path $cfgUsed -Value "`r`nrandom_seed: $Seed"

  $logFile = Join-Path $runDir "run.log"
  Write-Host ""
  Write-Host (">>> {0} RUN {1} seed={2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $name, $Seed) -ForegroundColor Cyan

  # 执行；同时在屏幕与日志文件里输出
  & $PythonExe "openevolve-run.py" `
      $InitialProgram `
      $Evaluator `
      --config $cfgUsed `
      --output $runDir 2>&1 | Tee-Object -FilePath $logFile

  # 收集结果到汇总
  $infoJson = Join-Path $runDir "best\best_program_info.json"
  if (Test-Path $infoJson) {
    $j = Get-Content $infoJson -Raw | ConvertFrom-Json
    $m = $j.metrics
    [pscustomobject]@{
      config           = $name
      seed             = $Seed
      combined_score   = $m.combined_score
      value_score      = $m.value_score
      distance_score   = $m.distance_score
      overall_score    = $m.overall_score
      success_rate     = $m.success_rate
    }
  }
}

# 主循环
$summary = @()
foreach ($cfg in $Configs) {
  foreach ($s in $Seeds) {
    $row = Invoke-OpenEvolveJob -Cfg $cfg -Seed $s
    if ($row) { $summary += $row }
  }
}

# 导出汇总
if ($summary.Count -gt 0) {
  New-Item -ItemType Directory -Force -Path "runs" | Out-Null
  $summaryPath = Join-Path "runs" "ablation_summary.csv"
  $summary | Sort-Object config, seed | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $summaryPath
  Write-Host ""
  Write-Host "✅ Summary written: $summaryPath" -ForegroundColor Green
} else {
  Write-Host "⚠️  No results found to summarize." -ForegroundColor Yellow
}
