# Benchmarking

UMA-X performance work starts with a control experiment.

## First test

Use a small 1B-2B Q4 GGUF so the benchmark harness can be validated quickly.

Run:

~~~powershell
python -m pip install -e .
umax doctor --llama-bin-dir C:\path\to\llama.cpp\bin
umax bench C:\models\small-model.gguf --llama-bin-dir C:\path\to\llama.cpp\bin
~~~

The command executes two runs:

1. **Plain llama.cpp** — llama-bench with its normal defaults.
2. **UMA-X planner path** — the same llama-bench binary and GGUF with UMA-X's explicit plan.

llama-bench reports prompt processing and text-generation throughput. UMA-X stores the raw rows, exact commands, hardware snapshot, and plan in a JSON artifact under .umax/benchmarks.

## Fairness rules

Every claimed comparison must keep constant:

- physical machine;
- power mode;
- model file;
- quantization;
- llama.cpp build;
- prompt token count;
- generated token count;
- repetition count;
- foreground workload conditions.

Record both cold and warm behavior when storage is part of the experiment.

Do not count:

- prompt-processing tokens as generated tokens;
- rejected speculative candidates as output tokens;
- aggregate throughput from multiple streams as single-user generation speed.

## v0.1 interpretation

A v0.1 delta is **not** evidence that UMA-X virtual memory is faster. The native tensor-paging subsystem does not exist yet.

The purpose of v0.1 is to establish:

- a trustworthy baseline;
- machine/device discovery;
- artifact capture;
- a model/hardware-scoped profile;
- a repeatable place to evaluate each later subsystem.

## Later memory-pressure experiment

When the native backend exists, benchmark at multiple controlled resident-memory budgets.

For each budget capture:

- generation tokens/s;
- prompt tokens/s;
- wall-clock latency;
- storage bytes read;
- storage queue depth;
- RAM working set;
- prefetch hit rate;
- GPU utilization;
- stall time waiting for weights.

The important metric is end-to-end verified output throughput.
