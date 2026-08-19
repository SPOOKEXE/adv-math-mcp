"""Numeric ops: precision that is real, and empirical answers labelled empirical."""

from __future__ import annotations

import pytest

from math_mcp.cas.numeric import evaluate
from math_mcp.cas.session import MathError, Session


@pytest.fixture()
def session() -> Session:
    return Session()


class TestEvalf:
    def test_digits_are_delivered(self, session: Session) -> None:
        result = evaluate(session, "evalf", "pi", digits=40)
        assert result["value"].startswith("3.14159265358979323846")

    def test_unbound_symbols_are_named_not_zeroed(self, session: Session) -> None:
        with pytest.raises(MathError, match="x.*unbound|unbound.*x"):
            evaluate(session, "evalf", "x + 1")


class TestRoot:
    def test_a_root_comes_with_its_residual(self, session: Session) -> None:
        result = evaluate(session, "root", "cos(x) - x", start={"x": "1"})
        assert result["root"]["x"].startswith("0.739085")
        assert float(result["residual"]) == pytest.approx(0.0, abs=1e-10)

    def test_equations_are_accepted(self, session: Session) -> None:
        # `x**2 = 2` and `x**2 - 2` are the same question; requiring the caller to move the
        # right side over is busywork.
        from math_mcp.cas.algebra import as_relation

        handle = session.store(as_relation(session, "x**2 = 2"))
        result = evaluate(session, "root", handle, start={"x": "1"})
        assert result["root"]["x"].startswith("1.41421")

    def test_a_start_is_required(self, session: Session) -> None:
        with pytest.raises(MathError, match="start"):
            evaluate(session, "root", "cos(x) - x")


class TestBounds:
    def test_extremes_of_a_monotone_function_live_at_the_corners(self, session: Session) -> None:
        # Random interior sampling misses corner extremes by a margin that looks like a real
        # bound; the corner sweep is what makes this usable on monotone expressions.
        result = evaluate(session, "bounds", "x + y", box={"x": [-1, 1], "y": [-1, 1]}, samples=10)
        assert result["empirical_min"]["value"] == -2.0
        assert result["empirical_max"]["value"] == 2.0

    def test_the_answer_says_it_is_empirical(self, session: Session) -> None:
        result = evaluate(session, "bounds", "x**2", box={"x": [-1, 1]}, samples=10)
        assert "not a proof" in result["detail"]

    def test_an_uncovered_variable_is_named(self, session: Session) -> None:
        with pytest.raises(MathError, match="y"):
            evaluate(session, "bounds", "x + y", box={"x": [0, 1]})


class TestConvexity:
    def test_a_constant_hessian_settles_it_exactly(self, session: Session) -> None:
        result = evaluate(session, "convexity", "x**2 + y**2", wrt=["x", "y"])
        assert result["convex"] == "proved"
        assert result["concave"] == "disproved"

    def test_a_symbolic_second_derivative_can_prove_it(self, session: Session) -> None:
        result = evaluate(session, "convexity", "exp(x)", wrt=["x"])
        assert result["convex"] == "proved"

    def test_refutation_carries_a_witness_point(self, session: Session) -> None:
        result = evaluate(session, "convexity", "x**3", wrt=["x"])
        assert result["convex"] == "disproved"
        assert result["concave"] == "disproved"
        assert "convexity_witness" in result

    def test_absence_of_a_witness_is_never_proof(self, session: Session) -> None:
        # x**4 is convex, but its Hessian is not constant and sympy cannot sign it here;
        # the honest answer is "not-refuted", never "proved" from samples.
        session.declare("a", real=True)
        result = evaluate(session, "convexity", "a**4 + sin(a)/1000", wrt=["a"])
        assert result["convex"].startswith("not-refuted") or result["convex"] == "disproved"
        assert "not a proof" in result["detail"] or "symbolically" in result["detail"]


class TestMoreRefusals:
    def test_an_unknown_op_is_refused_by_name(self, session: Session) -> None:
        with pytest.raises(MathError, match="minimise"):
            evaluate(session, "minimise", "x")

    def test_bounds_without_a_box_is_refused_with_an_example(self, session: Session) -> None:
        with pytest.raises(MathError, match=r"\[-1, 1\]"):
            evaluate(session, "bounds", "x**2")

    def test_a_root_search_that_cannot_converge_says_so(self, session: Session) -> None:
        # x**2 + 1 has no real root; nsolve's failure surfaces as a refusal naming the start,
        # never as a complex number quietly presented as an answer.
        with pytest.raises(MathError, match="did not converge"):
            evaluate(session, "root", "x**2 + 1", start={"x": "1"})

    def test_convexity_needs_a_variable(self, session: Session) -> None:
        with pytest.raises(MathError, match="at least one variable"):
            evaluate(session, "convexity", "3")

    def test_a_convexity_box_must_cover_every_variable(self, session: Session) -> None:
        with pytest.raises(MathError, match="cover"):
            evaluate(session, "convexity", "x*y + y**4", wrt=["x", "y"], box={"x": [0, 1]})


class TestMoreBranches:
    def test_bounds_counts_points_where_the_expression_is_undefined(self, session: Session) -> None:
        # sqrt over a box straddling zero: negative samples are off the domain, and the count
        # travels with the answer instead of vanishing.
        result = evaluate(session, "bounds", "sqrt(x)", box={"x": [-1, 1]}, samples=50)
        assert result["points_undefined"] > 0
        assert result["empirical_min"]["value"] >= 0.0

    def test_a_box_entirely_off_the_domain_is_an_error(self, session: Session) -> None:
        with pytest.raises(MathError, match="undefined"):
            evaluate(session, "bounds", "sqrt(x)", box={"x": [-4, -2]}, samples=10)

    def test_concavity_can_be_proved_symbolically(self, session: Session) -> None:
        session.declare("x", positive=True)
        result = evaluate(session, "convexity", "log(x)", wrt=["x"])
        assert result["concave"] == "proved"
        assert result["convex"] == "unknown"

    def test_a_convexity_box_confines_the_witness_search(self, session: Session) -> None:
        # x**3 is convex for positive x: a box on [1, 2] must find a concavity witness inside
        # it and must not refute convexity with a point it was told not to visit.
        result = evaluate(session, "convexity", "x**3", wrt=["x"], box={"x": [1, 2]})
        assert result["concave"] == "disproved"
        assert 1.0 <= result["concavity_witness"]["x"] <= 2.0
        assert result["convex"].startswith("not-refuted")

    def test_evalf_substitutes_before_evaluating(self, session: Session) -> None:
        result = evaluate(session, "evalf", "x**2 + y", at={"x": "3", "y": "1/4"}, digits=10)
        assert result["value"] == "9.250000000"
