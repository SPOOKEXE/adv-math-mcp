"""Probability: symbolic answers over named distributions, refusals that teach."""

from __future__ import annotations

import pytest

from math_mcp.cas.prob import prob
from math_mcp.cas.session import MathError, Session


@pytest.fixture()
def session() -> Session:
    session = Session()
    # Declared up front: `sigma > 0` is what lets sympy.stats accept a symbolic std.
    session.declare("sigma", positive=True)
    return session


NORMAL = {"mean": "mu", "std": "sigma"}


class TestMoments:
    def test_expectation_is_symbolic_not_sampled(self, session: Session) -> None:
        result = prob(session, "expectation", "normal", NORMAL)
        assert result["pretty"] == "mu"

    def test_variance_of_a_symbolic_normal(self, session: Session) -> None:
        assert prob(session, "variance", "normal", NORMAL)["pretty"] == "sigma**2"

    def test_expectation_of_an_expression_in_the_variable(self, session: Session) -> None:
        result = prob(session, "expectation", "normal", {"mean": "0", "std": "1"}, expr="X**2")
        assert result["pretty"] == "1"


class TestDensitiesAndProbability:
    def test_pdf_at_a_point(self, session: Session) -> None:
        result = prob(session, "pdf", "normal", {"mean": "0", "std": "1"}, at="0")
        assert result["pretty"] == "sqrt(2)/(2*sqrt(pi))"

    def test_probability_of_a_condition(self, session: Session) -> None:
        result = prob(session, "probability", "normal", {"mean": "0", "std": "1"}, condition="X > 0")
        assert result["pretty"] == "1/2"

    def test_a_non_condition_is_refused_with_an_example(self, session: Session) -> None:
        with pytest.raises(MathError, match="X > 1"):
            prob(session, "probability", "normal", {"mean": "0", "std": "1"}, condition="X + 1")


class TestKL:
    def test_gaussian_kl_reaches_the_closed_form(self, session: Session) -> None:
        # KL(N(0,1) ‖ N(1,1)) = 1/2 exactly; anything erf-shaped means simplification failed.
        result = prob(
            session,
            "kl",
            "normal",
            {"mean": "0", "std": "1"},
            other={"family": "normal", "params": {"mean": "1", "std": "1"}},
        )
        assert result["pretty"] == "1/2"


class TestRefusals:
    def test_unknown_families_list_the_known_ones(self, session: Session) -> None:
        with pytest.raises(MathError, match="normal"):
            prob(session, "expectation", "cauchy-ish", {})

    def test_missing_parameters_are_named(self, session: Session) -> None:
        with pytest.raises(MathError, match="missing std"):
            prob(session, "expectation", "normal", {"mean": "0"})

    def test_invalid_parameters_surface_sympys_reason(self, session: Session) -> None:
        with pytest.raises(MathError, match="invalid parameters"):
            prob(session, "expectation", "normal", {"mean": "0", "std": "-1"})


class TestMoreBranches:
    def test_cdf_of_a_normal_at_its_mean_is_a_half(self, session: Session) -> None:
        result = prob(session, "cdf", "normal", {"mean": "0", "std": "1"}, at="0")
        assert result["pretty"] == "1/2"

    def test_a_discrete_pdf_reads_from_the_density_table(self, session: Session) -> None:
        # Discrete densities come back from sympy.stats as a mapping, not a callable; the
        # lookup path is different code from the continuous one and earns its own pin.
        result = prob(session, "pdf", "bernoulli", {"p": "3/10"}, at="1")
        assert result["pretty"] == "3/10"

    def test_render_is_opt_in(self, session: Session) -> None:
        plain = prob(session, "expectation", "normal", {"mean": "0", "std": "1"})
        assert "render" not in plain
        rendered = prob(session, "variance", "normal", NORMAL, render=True)
        assert rendered["render"]["latex"] == r"\sigma^{2}"

    def test_kl_over_discrete_families_sums_over_the_support(self, session: Session) -> None:
        # E_p[log p/q] sums rather than integrates for a discrete variable, and
        # KL(Bern(1/2) ‖ Bern(1/4)) = (1/2)log(1/2 / 1/4) + (1/2)log(1/2 / 3/4) = log 2 - (log 3)/2.
        result = prob(
            session,
            "kl",
            "bernoulli",
            {"p": "1/2"},
            other={"family": "bernoulli", "params": {"p": "1/4"}},
        )
        assert result["pretty"] in ("-log(3)/2 + log(2)", "log(2) - log(3)/2")


class TestMoreRefusals:
    def test_an_unknown_op_is_refused_by_name(self, session: Session) -> None:
        with pytest.raises(MathError, match="entropy"):
            prob(session, "entropy", "normal", NORMAL)

    def test_probability_without_a_condition_is_refused(self, session: Session) -> None:
        with pytest.raises(MathError, match="condition"):
            prob(session, "probability", "normal", NORMAL)

    def test_kl_without_the_second_distribution_is_refused(self, session: Session) -> None:
        with pytest.raises(MathError, match="other"):
            prob(session, "kl", "normal", NORMAL)
