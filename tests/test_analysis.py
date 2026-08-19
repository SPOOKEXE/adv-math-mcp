"""Calculus ops: self-checking integrals, honest failures, error terms that travel."""

from __future__ import annotations

import pytest

from math_mcp.cas.analysis import calc
from math_mcp.cas.session import MathError, Session


@pytest.fixture()
def session() -> Session:
    return Session()


class TestIntegrate:
    def test_an_antiderivative_is_verified_by_differentiating_back(self, session: Session) -> None:
        result = calc(session, "integrate", "sin(x)", "x")
        assert result.pretty == "-cos(x)"
        assert result.verified == "proved"

    def test_a_definite_integral_has_no_verification_claim(self, session: Session) -> None:
        result = calc(session, "integrate", "x**2", "x", bounds=["0", "1"])
        assert result.pretty == "1/3"
        assert result.verified == ""

    def test_an_integral_sympy_cannot_do_is_reported_not_returned(self, session: Session) -> None:
        # An unevaluated Integral(...) handed back as an answer is how a wrong "closed form"
        # ends up in someone's derivation.
        result = calc(session, "integrate", "x**x", "x")
        assert result.expr_id == ""
        assert any("could not evaluate" in warning for warning in result.warnings)


class TestLimit:
    def test_a_two_sided_limit(self, session: Session) -> None:
        assert calc(session, "limit", "sin(x)/x", "x", point="0").pretty == "1"

    def test_disagreeing_sides_report_both_rather_than_picking_one(self, session: Session) -> None:
        result = calc(session, "limit", "1/x", "x", point="0")
        assert result.expr_id == ""
        assert any("disagree" in warning for warning in result.warnings)

    def test_one_sided_when_asked(self, session: Session) -> None:
        assert calc(session, "limit", "1/x", "x", point="0", direction="+").pretty == "oo"


class TestSeries:
    def test_the_error_term_is_a_separate_field(self, session: Session) -> None:
        result = calc(session, "series", "exp(x)", "x", order=4)
        assert result.pretty == "x**3/6 + x**2/2 + x + 1"
        # The same shape the contract layer's `error_term` field expects.
        assert result.error_term == "O(x**4)"


class TestSumProduct:
    def test_a_symbolic_upper_bound_works(self, session: Session) -> None:
        result = calc(session, "sum", "k", "k", bounds=["1", "n"])
        assert result.pretty == "n**2/2 + n/2"

    def test_bounds_are_required(self, session: Session) -> None:
        with pytest.raises(MathError, match="bounds"):
            calc(session, "sum", "k", "k")

    def test_a_product_evaluates(self, session: Session) -> None:
        assert calc(session, "product", "k", "k", bounds=["1", "5"]).pretty == "120"


class TestDispatch:
    def test_unknown_ops_are_refused_by_name(self, session: Session) -> None:
        with pytest.raises(MathError, match="differentiate"):
            calc(session, "differentiate", "x", "x")


class TestRefusals:
    def test_limit_without_a_point_is_refused(self, session: Session) -> None:
        with pytest.raises(MathError, match="needs a point"):
            calc(session, "limit", "1/x", "x")

    def test_a_bad_direction_is_refused_with_the_real_ones(self, session: Session) -> None:
        with pytest.raises(MathError, match="`both`"):
            calc(session, "limit", "1/x", "x", point="0", direction="sideways")


class TestHonestTimeouts:
    def test_a_hopeless_integral_reports_the_deadline_not_a_hang(self, session: Session) -> None:
        # This integrand stalls sympy for minutes; the 50ms deadline fires with a wide margin.
        # A partial result with the work that finished beats a call that never returns.
        result = calc(session, "integrate", "sqrt(1 + x**4)/(1 + x**7)", "x", timeout=0.05)
        assert result.expr_id == ""
        assert any("exceeded" in warning for warning in result.warnings)


class TestMoreBranches:
    def test_series_expands_at_a_nonzero_point(self, session: Session) -> None:
        result = calc(session, "series", "log(x)", "x", point="1", order=3)
        assert result.error_term == "O((x - 1)**3, (x, 1))"

    def test_a_product_with_no_closed_form_warns(self, session: Session) -> None:
        result = calc(session, "product", "1 + 1/k**k", "k", bounds=["1", "n"])
        assert result.expr_id == ""
        assert any("could not evaluate" in warning for warning in result.warnings)

    def test_render_is_opt_in(self, session: Session) -> None:
        assert calc(session, "integrate", "cos(x)", "x").render is None
        rendered = calc(session, "integrate", "cos(x)", "x", render=True).render
        assert rendered == {"latex": r"\sin{\left(x \right)}", "text": "sin(x)"}


class TestWireFormat:
    def test_to_dict_carries_only_the_fields_that_are_set(self, session: Session) -> None:
        clean = calc(session, "integrate", "x**2", "x", bounds=["0", "1"]).to_dict()
        assert "warnings" not in clean and "steps" not in clean and "render" not in clean

    def test_warnings_and_steps_appear_when_earned(self, session: Session) -> None:
        warned = calc(session, "integrate", "x**x", "x", steps=True).to_dict()
        assert "warnings" in warned and "steps" in warned
