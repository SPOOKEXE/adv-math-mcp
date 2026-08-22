"""Tensor planning: shape contracts, Pareto schedules, streaming and bounded search."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from math_mcp.cas.session import MathError
from math_mcp.cas.tensor import tensor_plan


def mm_request(m: int = 8, k: int = 16, n: int = 4, budget: int = 4096) -> dict:
    return {
        "kind": "contraction",
        "spec": "mk,kn->mn",
        "inputs": ["a", "b"],
        "tensors": {"a": {"shape": [m, k], "dtype": "bf16"}, "b": {"shape": [k, n], "dtype": "bf16"}},
        "memory": {"fast_bytes": budget},
        "tile_alignment": 4,
    }


def attention_request(q: int = 64, k: int = 96, budget: int = 65536, *, batch: int = 2, heads: int = 4, depth: int = 16) -> dict:
    return {
        "kind": "pipeline",
        "dims": {"b": batch, "q": q, "k": k, "h": heads, "d": depth},
        "tensors": {
            "q": {"shape": ["b", "q", "h", "d"], "dtype": "bf16"},
            "k": {"shape": ["b", "k", "h", "d"], "dtype": "bf16"},
            "v": {"shape": ["b", "k", "h", "d"], "dtype": "bf16"},
        },
        "ops": [
            {"id": "scores", "op": "einsum", "spec": "bqhd,bkhd->bhqk", "inputs": ["q", "k"]},
            {"id": "prob", "op": "softmax", "inputs": ["scores"], "axis": "k"},
            {"id": "out", "op": "einsum", "spec": "bhqk,bkhd->bqhd", "inputs": ["prob", "v"]},
        ],
        "memory": {"fast_bytes": budget},
        "tile_alignment": 8,
    }


class TestContraction:
    def test_matmul_counts_shapes_flops_and_compulsory_output(self) -> None:
        result = tensor_plan(mm_request(8, 16, 4))
        assert result["status"] == "feasible"
        assert result["operations"][0]["output"]["shape"] == [8, 4]
        assert result["operations"][0]["flops"] == 2 * 8 * 16 * 4
        assert result["compulsory"]["resident_output_bytes"] == 8 * 4 * 2

    def test_streamed_output_is_logical_but_not_resident(self) -> None:
        request = mm_request()
        request["output"] = {"mode": "stream"}
        result = tensor_plan(request)
        assert result["compulsory"]["resident_output_bytes"] == 0
        assert result["compulsory"]["logical_output_bytes"] > 0

    def test_every_returned_plan_fits_the_fast_memory_budget(self) -> None:
        result = tensor_plan(mm_request(128, 64, 96, budget=2048))
        assert result["plans"]
        assert all(plan["workspace_bytes"] <= 2048 for plan in result["plans"])

    def test_the_frontier_is_not_dominated(self) -> None:
        plans = tensor_plan(mm_request(64, 64, 64, budget=32768))["plans"]
        for left in plans:
            for right in plans:
                if left is right:
                    continue
                dominates = (
                    right["workspace_bytes"] <= left["workspace_bytes"]
                    and right["slow_traffic_bytes"] <= left["slow_traffic_bytes"]
                    and (right["workspace_bytes"] < left["workspace_bytes"] or right["slow_traffic_bytes"] < left["slow_traffic_bytes"])
                )
                assert not dominates

    @pytest.mark.sweep
    @pytest.mark.parametrize("budget", [8, 16, 64, 256, 1024, 4096])
    def test_budget_sweep_never_returns_an_oversized_tile(self, budget: int) -> None:
        result = tensor_plan(mm_request(17, 19, 23, budget))
        assert all(plan["workspace_bytes"] <= budget for plan in result["plans"])

    @pytest.mark.sweep
    @pytest.mark.parametrize("shape", [(7, 11, 5), (31, 17, 23), (64, 64, 64), (129, 33, 65)])
    def test_more_workspace_never_worsens_the_best_modelled_traffic(self, shape: tuple[int, int, int]) -> None:
        low = tensor_plan(mm_request(*shape, budget=256))
        high = tensor_plan(mm_request(*shape, budget=8192))
        assert min(plan["slow_traffic_bytes"] for plan in high["plans"]) <= min(plan["slow_traffic_bytes"] for plan in low["plans"])


class TestPipelines:
    def test_online_attention_avoids_both_quadratic_intermediates(self) -> None:
        result = tensor_plan(attention_request())
        assert result["plans"][0]["schedule"] == "online_tiled_attention"
        assert set(result["materializations_avoided"]) >= {"scores", "prob"}
        assert result["plans"][0]["state"] == ["running_max", "scaled_normalizer", "weighted_value_accumulator"]

    def test_online_attention_says_floating_order_changes(self) -> None:
        result = tensor_plan(attention_request())
        assert result["plans"][0]["exactness"] == "real_exact"
        assert result["plans"][0]["floating_point_order_changed"] is True

    def test_attention_slow_memory_feasibility_uses_avoided_intermediates(self) -> None:
        request = attention_request(64, 64, budget=65536, batch=1, heads=1, depth=16)
        request["memory"]["slow_bytes"] = 9000
        result = tensor_plan(request)
        assert result["naive"]["peak_live_bytes"] > 9000
        assert result["optimized_liveness"]["peak_live_bytes"] <= 9000
        assert result["status"] == "feasible"

    def test_bitwise_request_gets_an_explicit_warning(self) -> None:
        request = attention_request()
        request["semantics"] = {"mode": "bitwise_same"}
        result = tensor_plan(request)
        assert any("bitwise" in warning for warning in result["warnings"])

    def test_elementwise_single_consumer_is_fused(self) -> None:
        result = tensor_plan(
            {
                "kind": "pipeline",
                "tensors": {"x": {"shape": [128], "axes": ["n"]}},
                "ops": [{"id": "a", "op": "elementwise", "inputs": ["x"]}, {"id": "b", "op": "elementwise", "inputs": ["a"]}],
            }
        )
        assert result["materializations_avoided"] == ["a"]
        assert result["optimized_liveness"]["peak_live_bytes"] <= result["naive"]["peak_live_bytes"]

    def test_checkpoint_policy_respects_its_budget(self) -> None:
        request = {
            "kind": "checkpoint",
            "tensors": {"x": {"shape": [128], "axes": ["n"]}},
            "ops": [
                {"id": "a", "op": "elementwise", "inputs": ["x"], "cost": 1},
                {"id": "b", "op": "elementwise", "inputs": ["a"], "cost": 10},
                {"id": "c", "op": "elementwise", "inputs": ["b"], "cost": 1},
            ],
            "memory": {"fast_bytes": 512, "slow_bytes": 512},
        }
        checkpoint = tensor_plan(request)["checkpoint"]
        assert checkpoint["saved_activation_bytes"] <= 512
        assert checkpoint["status"] == "proved-optimal-in-model"

    def test_standalone_reduction_gets_a_bounded_state_schedule(self) -> None:
        result = tensor_plan(
            {
                "kind": "pipeline",
                "tensors": {"x": {"shape": [1024], "axes": ["n"]}},
                "ops": [{"id": "total", "op": "sum", "inputs": ["x"], "axis": "n"}],
                "memory": {"fast_bytes": 64},
            }
        )
        assert result["plans"][0]["schedule"] == "streamed_reduction"
        assert result["plans"][0]["state"] == ["sum"]
        assert result["plans"][0]["workspace_bytes"] <= 64

    def test_windowed_compute_tracks_output_shape_and_halo(self) -> None:
        result = tensor_plan(
            {
                "kind": "pipeline",
                "tensors": {"x": {"shape": [2, 17], "axes": ["b", "n"]}},
                "ops": [
                    {
                        "id": "filtered",
                        "op": "window",
                        "inputs": ["x"],
                        "axis": "n",
                        "window": 5,
                        "stride": 2,
                    }
                ],
                "memory": {"fast_bytes": 128},
            }
        )
        assert result["operations"][0]["output"]["shape"] == [2, 7]
        assert result["operations"][0]["flops"] == 2 * 2 * 7 * 5
        assert result["plans"][0]["schedule"] == "windowed_stencil"
        assert result["plans"][0]["state"]["halo_elements"] == 3

    def test_window_output_axis_can_feed_a_later_named_reduction(self) -> None:
        result = tensor_plan(
            {
                "kind": "pipeline",
                "tensors": {"x": {"shape": [17], "axes": ["n"]}},
                "ops": [
                    {"id": "windows", "op": "window", "inputs": ["x"], "axis": "n", "window": 3},
                    {"id": "total", "op": "sum", "inputs": ["windows"], "axis": "n_out"},
                ],
                "memory": {"fast_bytes": 256},
            }
        )
        assert result["tensors"]["windows"]["axes"] == ["n_out"]
        assert result["tensors"]["total"]["elements"] == 1

    @pytest.mark.sweep
    @pytest.mark.parametrize("window,stride,dilation", [(1, 1, 1), (3, 1, 1), (5, 2, 1), (3, 2, 2)])
    def test_window_parameter_sweep_matches_the_valid_shape_formula(
        self, window: int, stride: int, dilation: int
    ) -> None:
        length = 31
        effective = dilation * (window - 1) + 1
        result = tensor_plan(
            {
                "kind": "pipeline",
                "tensors": {"x": {"shape": [length], "axes": ["n"]}},
                "ops": [
                    {
                        "id": "y",
                        "op": "window",
                        "inputs": ["x"],
                        "axis": "n",
                        "window": window,
                        "stride": stride,
                        "dilation": dilation,
                    }
                ],
                "memory": {"fast_bytes": 256},
            }
        )
        assert result["tensors"]["y"]["shape"] == [(length - effective) // stride + 1]
        assert all(plan["workspace_bytes"] <= 256 for plan in result["plans"])

    def test_hardware_model_adds_a_roofline_lower_bound(self) -> None:
        request = mm_request()
        request["hardware"] = {"peak_flops": 1000, "bandwidth_bytes_per_s": 100}
        result = tensor_plan(request)
        assert result["hardware"]["ridge_flops_per_byte"] == 10
        assert all(plan["roofline"]["latency_lower_bound_seconds"] > 0 for plan in result["plans"])

    @pytest.mark.sweep
    @pytest.mark.parametrize("shape", [(17, 19), (64, 96), (129, 257)])
    def test_attention_budget_sweep_improves_or_preserves_traffic(self, shape: tuple[int, int]) -> None:
        query, key = shape
        low = tensor_plan(attention_request(query, key, budget=256))
        high = tensor_plan(attention_request(query, key, budget=65536))
        assert min(plan["slow_traffic_bytes"] for plan in high["plans"]) <= min(plan["slow_traffic_bytes"] for plan in low["plans"])


class TestValidation:
    @pytest.mark.parametrize(
        "change, message",
        [
            ({"spec": "...k,kn->...n"}, "ellipses"),
            ({"dtype": "mystery"}, "dtype"),
            ({"memory": {"fast_bytes": 0}}, "positive integer"),
            ({"timeout": "later"}, "positive finite"),
            ({"hardware": {"peak_flops": float("nan"), "bandwidth_bytes_per_s": 1}}, "positive finite"),
        ],
    )
    def test_invalid_public_inputs_fail_loudly(self, change: dict, message: str) -> None:
        request = mm_request()
        request.update(change)
        with pytest.raises(MathError, match=message):
            tensor_plan(request)

    def test_named_axis_mismatch_is_not_accepted_because_sizes_match(self) -> None:
        request = mm_request(8, 8, 8)
        request["tensors"]["a"]["axes"] = ["batch", "feature"]
        with pytest.raises(MathError, match="declares axes"):
            tensor_plan(request)


@pytest.mark.fuzz
@settings(max_examples=40, derandomize=True, deadline=None)
@given(m=st.integers(min_value=1, max_value=32), k=st.integers(min_value=1, max_value=32), n=st.integers(min_value=1, max_value=32))
def test_fuzzed_matmul_plans_preserve_count_and_fit(m: int, k: int, n: int) -> None:
    result = tensor_plan(mm_request(m, k, n, budget=16384))
    assert result["operations"][0]["flops"] == 2 * m * k * n
    assert all(plan["workspace_bytes"] <= 16384 for plan in result["plans"])
    assert all(1 <= plan["tile"][axis] <= size for plan in result["plans"] for axis, size in {"m": m, "k": k, "n": n}.items())


@pytest.mark.fuzz
@settings(max_examples=20, derandomize=True, deadline=None)
@given(m=st.integers(min_value=1, max_value=9), k=st.integers(min_value=1, max_value=9), n=st.integers(min_value=1, max_value=9))
def test_fuzzed_tile_schedule_matches_dense_matmul(m: int, k: int, n: int) -> None:
    result = tensor_plan(mm_request(m, k, n, budget=4096))
    tile = result["plans"][-1]["tile"]
    rng = np.random.default_rng(20260823 + m * 100 + k * 10 + n)
    left = rng.normal(size=(m, k))
    right = rng.normal(size=(k, n))
    actual = np.zeros((m, n))
    for row in range(0, m, tile["m"]):
        for column in range(0, n, tile["n"]):
            accumulator = np.zeros((min(tile["m"], m - row), min(tile["n"], n - column)))
            for reduction in range(0, k, tile["k"]):
                accumulator += (
                    left[row : row + tile["m"], reduction : reduction + tile["k"]]
                    @ right[reduction : reduction + tile["k"], column : column + tile["n"]]
                )
            actual[row : row + tile["m"], column : column + tile["n"]] = accumulator
    np.testing.assert_allclose(actual, left @ right, rtol=1e-12, atol=1e-12)


@pytest.mark.fuzz
@settings(max_examples=25, derandomize=True, deadline=None)
@given(
    queries=st.integers(min_value=1, max_value=8), keys=st.integers(min_value=1, max_value=12), depth=st.integers(min_value=1, max_value=8)
)
def test_fuzzed_online_attention_state_matches_dense_attention(queries: int, keys: int, depth: int) -> None:
    result = tensor_plan(attention_request(queries, keys, budget=8192, batch=1, heads=1, depth=depth))
    key_block = result["plans"][-1]["tile"]["key"]
    rng = np.random.default_rng(20260823 + queries * 100 + keys * 10 + depth)
    query = rng.normal(size=(queries, depth))
    key = rng.normal(size=(keys, depth))
    value = rng.normal(size=(keys, depth))

    dense_scores = query @ key.T
    dense_weights = np.exp(dense_scores - dense_scores.max(axis=1, keepdims=True))
    expected = dense_weights @ value / dense_weights.sum(axis=1, keepdims=True)

    running_max = np.full(queries, -np.inf)
    normalizer = np.zeros(queries)
    accumulator = np.zeros((queries, depth))
    for start in range(0, keys, key_block):
        scores = query @ key[start : start + key_block].T
        block_max = scores.max(axis=1)
        weights = np.exp(scores - block_max[:, None])
        block_normalizer = weights.sum(axis=1)
        block_accumulator = weights @ value[start : start + key_block]
        merged_max = np.maximum(running_max, block_max)
        old_scale = np.exp(running_max - merged_max)
        block_scale = np.exp(block_max - merged_max)
        normalizer = old_scale * normalizer + block_scale * block_normalizer
        accumulator = old_scale[:, None] * accumulator + block_scale[:, None] * block_accumulator
        running_max = merged_max
    actual = accumulator / normalizer[:, None]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_planning_is_deterministic_when_called_in_parallel() -> None:
    request = attention_request(48, 80)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: tensor_plan(request), range(16)))
    first = results[0]
    assert all(result["plans"] == first["plans"] for result in results[1:])
