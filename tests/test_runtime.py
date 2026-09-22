from pathlib import Path

from umax.runtime import build_command, parse_metrics


def test_parse_llama_cpp_metrics():
    output = """
llama_perf_context_print: prompt eval time = 100.00 ms / 10 tokens (100.00 tokens per second)
llama_perf_context_print:        eval time = 500.00 ms / 20 runs   (40.00 tokens per second)
"""
    metrics = parse_metrics(output, wall_seconds=1.25, returncode=0)
    assert metrics.prompt_tokens_per_second == 100.0
    assert metrics.generation_tokens_per_second == 40.0
    assert metrics.wall_seconds == 1.25
    assert metrics.returncode == 0


def test_build_command_is_deterministic():
    cmd = build_command(
        "llama-cli",
        Path("model.gguf"),
        "hello",
        32,
        1024,
        4,
        ["--temp", "0"],
    )
    assert cmd == [
        "llama-cli",
        "-m",
        "model.gguf",
        "-p",
        "hello",
        "-n",
        "32",
        "-c",
        "1024",
        "--seed",
        "42",
        "--no-display-prompt",
        "-t",
        "4",
        "--temp",
        "0",
    ]
