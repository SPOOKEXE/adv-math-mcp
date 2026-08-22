"""Certified enclosures, stability findings, complexity and error propagation."""

from __future__ import annotations

import math

import pytest
import sympy
from hypothesis import given, settings
from hypothesis import strategies as st

from math_mcp.cas.insight import analyze
from math_mcp.cas.session import MathError, Session


@pytest.fixture()
def session() -> Session:
    return Session()


class TestRigorousBounds:
    def test_polynomial_enclosure_is_proved(self, session: Session) -> None:
        result = analyze(session, "rigorous_bounds", "x**2 + y", box={"x": [-1, 2], "y": [3, 4]})
        assert result["verdict"] == "proved"
        assert result["enclosure"]["lower"] == "3"
        assert result["enclosure"]["upper"] == "8"

    def test_dependency_makes_a_safe_loose_bound(self, session: Session) -> None:
        result = analyze(session, "rigorous_bounds", "x - x", box={"x": [-2, 3]})
        assert result["verdict"] == "proved"
        assert float(result["enclosure"]["lower_numeric"]) <= 0
        assert float(result["enclosure"]["upper_numeric"]) >= 0

    def test_domain_hole_prevents_a_proof(self, session: Session) -> None:
        result = analyze(session, "rigorous_bounds", "log(x)", box={"x": [-1, 2]})
        assert result["verdict"] == "unknown"
        assert result["box_covered"] is False

    def test_unsupported_function_is_unknown_not_sampled(self, session: Session) -> None:
        result = analyze(session, "rigorous_bounds", "gamma(x)", box={"x": [1, 2]})
        assert result["verdict"] == "unknown"
        assert "not in" in result["detail"]


class TestOptimization:
    def test_global_univariate_range_is_proved(self, session: Session) -> None:
        result = analyze(session, "optimize", "x**2", wrt=["x"], box={"x": [-1, 2]})
        assert result["verdict"] == "proved"
        assert result["minimum"] == "0"
        assert result["maximum"] == "4"

    def test_multivariate_optimization_returns_an_enclosure_not_a_fake_optimum(self, session: Session) -> None:
        result = analyze(session, "optimize", "x*y", wrt=["x", "y"], box={"x": [-1, 2], "y": [3, 4]})
        assert result["verdict"] == "unknown"
        assert result["enclosure"] == {
            "lower": "-4",
            "upper": "8",
            "lower_numeric": "-4.000000000000000",
            "upper_numeric": "8.000000000000000",
        }


class TestStabilityAndComplexity:
    def test_stability_finds_log1p_and_division_hazards(self, session: Session) -> None:
        result = analyze(session, "stability", "log(1 + x) + 1/y")
        kinds = {finding["kind"] for finding in result["findings"]}
        assert {"small-increment-log", "division"} <= kinds

    def test_relative_condition_can_be_evaluated_at_a_point(self, session: Session) -> None:
        result = analyze(session, "stability", "x**2", wrt=["x"], at={"x": 3})
        assert float(result["relative_condition"][0]["at_point"]) == pytest.approx(2.0)

    def test_complexity_reports_the_cse_saving(self, session: Session) -> None:
        result = analyze(session, "complexity", "exp(x*y) + log(x*y) + (x*y)**2")
        assert result["verdict"] == "proved"
        assert result["common_subexpressions"] >= 1
        assert result["operations_after_cse"] < result["operations_before_cse"]


class TestErrorBudget:
    def test_first_order_contributions_evaluate_at_a_point(self, session: Session) -> None:
        result = analyze(session, "error_budget", "x*y", errors={"x": "0.1", "y": "0.2"}, at={"x": 2, "y": 3})
        assert float(result["at_point"]) == pytest.approx(0.7)
        assert "higher-order" in result["caveat"]

    def test_negative_input_error_is_refused(self, session: Session) -> None:
        with pytest.raises(MathError, match="nonnegative"):
            analyze(session, "error_budget", "x", errors={"x": -1})


class TestValidation:
    def test_unknown_analysis_op_lists_the_real_ones(self, session: Session) -> None:
        with pytest.raises(MathError, match="rigorous_bounds"):
            analyze(session, "guess", "x")

    def test_matrix_is_routed_to_linalg(self, session: Session) -> None:
        with pytest.raises(MathError, match="linalg"):
            analyze(session, "complexity", "Matrix([[1, 2]])")

    def test_reversed_box_is_refused(self, session: Session) -> None:
        with pytest.raises(MathError, match="low > high"):
            analyze(session, "rigorous_bounds", "x", box={"x": [2, -1]})

    def test_non_numeric_timeout_and_box_endpoint_are_structured_errors(self, session: Session) -> None:
        with pytest.raises(MathError, match="positive finite"):
            analyze(session, "complexity", "x", timeout="later")  # type: ignore[arg-type]
        with pytest.raises(MathError, match="finite real"):
            analyze(session, "rigorous_bounds", "x", box={"x": ["nan", 1]})


@pytest.mark.fuzz
@settings(max_examples=80, derandomize=True, deadline=None)
@given(
    a=st.integers(min_value=-5, max_value=5),
    b=st.integers(min_value=-5, max_value=5),
    c=st.integers(min_value=-5, max_value=5),
    low=st.integers(min_value=-5, max_value=4),
    width=st.integers(min_value=1, max_value=5),
    fraction=st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False),
)
def test_fuzzed_polynomial_value_is_inside_certified_enclosure(a: int, b: int, c: int, low: int, width: int, fraction: float) -> None:
    high = low + width
    point = low + fraction * width
    session = Session()
    expression = f"{a}*x**2 + {b}*x + {c}"
    result = analyze(session, "rigorous_bounds", expression, box={"x": [low, high]})
    assert result["verdict"] == "proved"
    lower = float(sympy.N(sympy.sympify(result["enclosure"]["lower"]), 17))
    upper = float(sympy.N(sympy.sympify(result["enclosure"]["upper"]), 17))
    actual = a * point * point + b * point + c
    assert math.isfinite(actual)
    assert lower - 1e-12 <= actual <= upper + 1e-12
