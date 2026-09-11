"""Tests for training.benchmark_dora — compare_results() only (pure logic,
no GPU/unsloth needed; run_benchmark/setup_model require a real GPU session).
"""

from __future__ import annotations

from training.benchmark_dora import BenchmarkResult, compare_results


def _result(method: str, loss_is_real: bool, final_loss: float = 0.3) -> BenchmarkResult:
    return BenchmarkResult(
        method=method,
        steps=50,
        final_loss=final_loss,
        avg_loss=final_loss,
        loss_curve=[final_loss] * 5,
        memory_mb=1000.0,
        steps_per_sec=2.0,
        adapter_size_mb=10.0,
        elapsed_sec=25.0,
        loss_is_real=loss_is_real,
    )


class TestCompareResults:
    def test_unmeasured_loss_gives_no_fabricated_winner(self):
        """Regression: the old fake loss curve gave the same formula to
        both methods, so a loss 'winner' was always reported despite
        being meaningless. Without a real measurement, no winner."""
        dora = _result("QDoRA", loss_is_real=False)
        qlora = _result("QLoRA", loss_is_real=False)
        comparison = compare_results(dora, qlora)

        assert comparison["winner"]["loss"] == "unmeasured"
        assert "INCONCLUSIVE" in comparison["recommendation"]

    def test_real_loss_produces_a_winner(self):
        dora = _result("QDoRA", loss_is_real=True, final_loss=0.2)
        qlora = _result("QLoRA", loss_is_real=True, final_loss=0.4)
        comparison = compare_results(dora, qlora)

        assert comparison["winner"]["loss"] == "dora"
        assert comparison["recommendation"] in ("use_dora", "use_qlora")

    def test_speed_and_memory_always_measured(self):
        """Speed/memory come from real torch/unsloth measurements even
        when loss doesn't — they should never be 'unmeasured'."""
        dora = _result("QDoRA", loss_is_real=False)
        qlora = _result("QLoRA", loss_is_real=False)
        comparison = compare_results(dora, qlora)

        assert comparison["winner"]["speed"] in ("dora", "qlora")
        assert comparison["winner"]["memory"] in ("dora", "qlora")
