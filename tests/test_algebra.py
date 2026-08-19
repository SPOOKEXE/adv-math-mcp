"""Solving and simplification: verified roots, honest failures, sound candidates."""

from __future__ import annotations

import pytest
import sympy

from math_mcp.cas.algebra import as_relation, simplify, solve
from math_mcp.cas.session import MathError, Session, canonical_form


@pytest.fixture()
def session() -> Session:
    return Session()


class TestRelationParsing:
    def test_single_equals_becomes_an_equation(self, session: Session) -> None:
        relation = as_relation(session, "x**2 = 4")
        assert isinstance(relation, sympy.Eq)

    def test_double_equals_is_not_swallowed_by_python(self, session: Session) -> None:
        # `a == b` on sympy objects evaluates to a bare bool, which silently discards the
        # equation; folding `==` to `=` first is what keeps it an Eq.
        relation = as_relation(session, "x**2 == 4")
        assert isinstance(relation, sympy.Eq)

    def test_inequalities_stay_relational(self, session: Session) -> None:
        relation = as_relation(session, "x**2 <= 4")
        assert isinstance(relation, sympy.core.relational.Relational)


class TestSolve:
    def test_roots_come_back_verified(self, session: Session) -> None:
        result = solve(session, "equation", ["x**2 = 4"], ["x"])
        assert {entry["x"] for entry in result.solutions} == {"-2", "2"}
        assert result.verified == ["proved", "proved"]

    def test_a_system_solves_and_verifies(self, session: Session) -> None:
        result = solve(session, "system", ["x + y = 3", "x - y = 1"], ["x", "y"])
        assert result.solutions == [{"x": "2", "y": "1"}]
        assert result.verified == ["proved"]

    def test_no_solutions_is_reported_not_invented(self, session: Session) -> None:
        result = solve(session, "equation", ["exp(x) = 0"], ["x"])
        assert result.solutions == []
        assert result.warnings

    def test_an_inequality_returns_a_condition_not_points(self, session: Session) -> None:
        result = solve(session, "inequality", ["x**2 < 4"], ["x"])
        assert "-2 < x" in result.condition and "x < 2" in result.condition

    def test_a_recurrence_with_initial_conditions(self, session: Session) -> None:
        result = solve(session, "recurrence", ["a(n+1) = 2*a(n)"], ["a", "n"], given={"a(0)": "1"})
        assert result.solutions == [{"a(n)": "2**n"}]

    def test_an_ode_with_initial_conditions(self, session: Session) -> None:
        result = solve(session, "ode", ["Derivative(f(x), x) = f(x)"], ["f", "x"], given={"f(0)": "1"})
        assert result.solutions == [{"f(x)": "exp(x)"}]

    def test_functional_kinds_demand_their_wrt_shape(self, session: Session) -> None:
        with pytest.raises(MathError, match=r"\[function, variable\]"):
            solve(session, "recurrence", ["a(n+1) = 2*a(n)"], ["a"])

    def test_diophantine_solutions_are_parametric(self, session: Session) -> None:
        result = solve(session, "diophantine", ["2*x + 3*y = 1"])
        assert len(result.solutions) == 1
        # A parametric family, not a single point: the parameter symbol appears in the answer.
        assert any("t_0" in value for value in result.solutions[0].values())

    def test_steps_are_off_by_default(self, session: Session) -> None:
        assert solve(session, "equation", ["x = 1"], ["x"]).steps == []
        assert solve(session, "equation", ["x = 1"], ["x"], steps=True).steps

    def test_render_is_off_by_default(self, session: Session) -> None:
        assert solve(session, "equation", ["x = 1"], ["x"]).render is None
        rendered = solve(session, "equation", ["x = 1"], ["x"], render=True).render
        assert rendered and "latex" in rendered[0]


class TestSimplify:
    def test_candidates_are_ranked_by_operation_count(self, session: Session) -> None:
        result = simplify(session, "x**2 + 2*x + 1")
        assert result.best["strategy"] == "factor"
        assert result.best["pretty"] == "(x + 1)**2"
        counts = [entry["ops"] for entry in result.candidates]
        assert counts == sorted(counts)

    def test_every_candidate_is_the_same_expression(self, session: Session) -> None:
        result = simplify(session, "(x**2 - 1)/(x - 1) + sin(x)**2 + cos(x)**2")
        original = session.get(result.candidates[0]["expr_id"])
        for entry in result.candidates[1:]:
            candidate = session.get(entry["expr_id"])
            assert sympy.simplify(canonical_form(original - candidate)) == 0

    def test_the_input_is_always_a_candidate(self, session: Session) -> None:
        # Worst case is "nothing improved it", reported as exactly that, never an empty result.
        result = simplify(session, "x + 1")
        assert any(entry["strategy"] == "input" for entry in result.candidates)

    def test_partfrac_needs_a_variable_and_skips_without_one(self, session: Session) -> None:
        # Two symbols, no `wrt`: there is no right guess, so the strategy is skipped, not guessed.
        result = simplify(session, "1/(x*y + y)", strategies=["partfrac"])
        assert [entry["strategy"] for entry in result.candidates] == ["input"]
        with_wrt = simplify(session, "1/(x**2 - 1)", strategies=["partfrac"], wrt="x")
        assert any(entry["strategy"] == "partfrac" for entry in with_wrt.candidates)

    def test_unknown_strategies_are_named(self, session: Session) -> None:
        with pytest.raises(MathError, match="magic"):
            simplify(session, "x", strategies=["magic"])


class TestSolveRefusals:
    def test_an_unknown_kind_is_refused_by_name(self, session: Session) -> None:
        with pytest.raises(MathError, match="wish"):
            solve(session, "wish", ["x = 1"])

    def test_nothing_to_solve_is_an_error_not_an_empty_success(self, session: Session) -> None:
        with pytest.raises(MathError, match="nothing to solve"):
            solve(session, "equation", [])

    def test_an_unsolvable_diophantine_warns_rather_than_stalls(self, session: Session) -> None:
        # gcd(4, 6) does not divide 1, so there are no integer solutions; the honest output is
        # an empty list that says so, never a silent empty list.
        result = solve(session, "diophantine", ["4*x + 6*y = 1"])
        assert result.solutions == []
        assert any("no integer solutions" in warning for warning in result.warnings)


class TestRenderPaths:
    def test_an_inequality_condition_renders_to_latex(self, session: Session) -> None:
        result = solve(session, "inequality", ["x**2 < 4"], ["x"], render=True)
        assert result.render and "latex" in result.render[0]

    def test_an_ode_solution_renders_to_latex(self, session: Session) -> None:
        result = solve(session, "ode", ["Derivative(f(x), x) = f(x)"], ["f", "x"], given={"f(0)": "1"}, render=True)
        assert result.render == [{"latex": "e^{x}", "text": "exp(x)"}]

    def test_functional_kinds_record_steps_when_asked(self, session: Session) -> None:
        result = solve(session, "recurrence", ["a(n+1) = 2*a(n)"], ["a", "n"], given={"a(0)": "1"}, steps=True)
        assert [step["stage"] for step in result.steps] == ["parsed", "solved"]


class TestSolverLimits:
    def test_a_two_variable_inequality_surfaces_sympys_refusal(self, session: Session) -> None:
        # reduce_inequalities handles one symbol of interest; the refusal crosses the boundary
        # as a MathError with sympy's reason, never as a bare NotImplementedError.
        with pytest.raises(MathError, match="could not reduce"):
            solve(session, "inequality", ["x*y > 1"], ["x", "y"])

    def test_a_transcendental_equation_beyond_the_solver_is_named(self, session: Session) -> None:
        with pytest.raises(MathError, match="solver: could not solve"):
            solve(session, "equation", ["sin(x) = x"], ["x"])

    def test_a_nonlinear_recurrence_is_refused_with_sympys_reason(self, session: Session) -> None:
        with pytest.raises(MathError, match="recurrence solver:"):
            solve(session, "recurrence", ["a(n+1) = a(n)**2"], ["a", "n"])


class TestWireFormat:
    def test_to_dict_carries_only_the_fields_that_are_set(self, session: Session) -> None:
        # The MCP payload: absent is absent, so an agent never reads an empty `condition` as
        # a finding or pays context for fields with nothing in them.
        equation = solve(session, "equation", ["x = 1"], ["x"]).to_dict()
        assert "condition" not in equation and "warnings" not in equation and "render" not in equation

        inequality = solve(session, "inequality", ["x**2 < 4"], ["x"], render=True).to_dict()
        assert "condition" in inequality and "render" in inequality

        unsolved = solve(session, "diophantine", ["4*x + 6*y = 1"]).to_dict()
        assert "warnings" in unsolved
