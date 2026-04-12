# Benchmark Frameworks

`benchmark/` 目录只用于存放第三方 benchmark 框架源码，不存放评测结果。

当前固定版本：

- `lm-evaluation-harness`：`v0.4.11`
- `BFCL`：官方 `2025.12.17` 复现点，对应 `gorilla` 仓库中的 `berkeley-function-call-leaderboard` 子目录，commit `f7cf735`
- 单 benchmark 环境锁：`requirements-benchmark.lock.txt`

使用方式：

```bash
python scripts/bootstrap_benchmarks.py
bash scripts/setup_benchmark_env.sh
python scripts/run_benchmarks.py --suite all
```

约定：

- 框架源码放在本目录下，由 `scripts/bootstrap_benchmarks.py` 拉取或更新。
- 评测结果统一写入 `analysis/stage1/benchmarks/`。
- 运行期缓存、日志和 benchmark runtime 放在 `runs/_benchmark_runtime/`。
- `lm-eval`、`BFCL` 与 vLLM 统一使用同一个独立 benchmark runtime，不会污染主训练 conda 环境。
- 由于 vLLM 当前仍要求 `transformers<5`，OpenAI-compatible 推理端固定在该独立 runtime 中启动。
