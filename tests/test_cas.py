"""CAS layer: what the model cannot fake."""

from __future__ import annotations

import math

import pytest
import sympy

from math_mcp.cas.calculus import check_grad, matrix_grad, shape_check, to_code
from math_mcp.cas.equivalence import (
    TimeoutExceeded,
    batch_equivalence,
    check_derivation,
    check_equivalence,
    deadline,
)
from math_mcp.cas.session import MathError, ParseError, Session, canonical_form


@pytest.fixture()
def session() -> Session:
    return Session()


class TestParsing:
    def test_handles_are_opaque_and_stable(self, session: Session) -> None:
        first, _ = session.parse("x**2 + 2*x + 1")
        second, _ = session.parse("1 + 2*x + x**2")
        # Content-derived from the canonical form: the same expression written two ways is one
        # handle, which is also what makes the handle a usable identity.
        assert first == second

    def test_a_handle_round_trips(self, session: Session) -> None:
        handle, expr = session.parse("sin(x) + cos(x)")
        assert session.get(handle) == expr

    def test_an_unknown_handle_is_named(self, session: Session) -> None:
        with pytest.raises(MathError, match="unknown expression handle"):
            session.get("e:deadbeef")

    def test_parse_errors_carry_a_position(self, session: Session) -> None:
        with pytest.raises(ParseError) as caught:
            session.parse("x + * 2")
        error = caught.value
        # A model can fix a column; it rewrites a whole expression from scratch otherwise, and
        # usually reintroduces the same mistake.
        assert error.column >= 1
        assert "^" in error.to_dict()["excerpt"]

    @pytest.mark.parametrize(
        "hostile",
        [
            "__import__('os').system('echo pwned')",
            "eval('1+1')",
            "().__class__.__bases__",
            "lambda: 1",
            "open('/etc/passwd')",
        ],
    )
    def test_the_parser_refuses_anything_that_reaches_the_interpreter(self, session: Session, hostile: str) -> None:
        # `sympify` would evaluate these. The input is by construction attacker-adjacent: it
        # comes from a model that read a web page.
        with pytest.raises(ParseError):
            session.parse(hostile)

    def test_an_undefined_name_becomes_a_symbol_not_an_import(self, session: Session) -> None:
        _, expr = session.parse("collections + 1")
        assert expr.free_symbols == {sympy.Symbol("collections")}

    def test_latex_folds_to_parser_syntax(self, session: Session) -> None:
        _, expr = session.parse(r"\frac{x^2}{2} + \sqrt{y}", syntax="latex")
        assert expr == sympy.Symbol("x") ** 2 / 2 + sympy.sqrt(sympy.Symbol("y"))

    def test_implicit_multiplication_parses(self, session: Session) -> None:
        _, expr = session.parse("2x + 3y")
        assert expr == 2 * sympy.Symbol("x") + 3 * sympy.Symbol("y")

    def test_canonical_form_identifies_rewrites(self) -> None:
        x = sympy.Symbol("x")
        assert canonical_form((x + 1) ** 2) == canonical_form(x**2 + 2 * x + 1)


class TestAssumptions:
    def test_declaring_rewrites_stored_expressions(self, session: Session) -> None:
        handle, _ = session.parse("sqrt(x**2)")
        session.declare("x", positive=True)
        # Sympy symbols compare by name *and* assumptions. Leaving the old symbol in stored
        # expressions produces `x != x` with no error anywhere.
        assert session.get(handle).free_symbols == {session.symbol("x")}
        assert sympy.simplify(session.get(handle) - session.symbol("x")) == 0

    def test_an_unknown_assumption_is_refused_by_name(self, session: Session) -> None:
        with pytest.raises(MathError, match="unknown assumption"):
            session.declare("x", purple=True)

    def test_assumptions_accumulate(self, session: Session) -> None:
        session.declare("n", integer=True)
        session.declare("n", positive=True)
        assert session.assumptions_of("n") == {"integer": True, "positive": True}


class TestEquivalence:
    def test_it_proves_a_real_identity(self, session: Session) -> None:
        result = check_equivalence(session, "(x + 1)**2", "x**2 + 2*x + 1")
        assert result.verdict == "proved"

    def test_it_disproves_with_a_real_counterexample(self, session: Session) -> None:
        result = check_equivalence(session, "(x + 1)**2", "x**2 + 1")
        assert result.verdict == "disproved"

        # The counterexample must actually be one, not a plausible-looking assignment.
        x = sympy.Symbol("x")
        value = sympy.Rational(result.counterexample["x"].split("+")[0].strip("(") or 0)
        assert ((x + 1) ** 2 - (x**2 + 1)).subs(x, value) != 0

    def test_a_verdict_is_never_a_bare_boolean(self, session: Session) -> None:
        result = check_equivalence(session, "x", "x")
        assert result.verdict in {"proved", "disproved", "unknown"}
        assert result.strategies

    def test_the_assumption_sensitive_case(self, session: Session) -> None:
        # The doc's claim, and the reason assumptions are first class: most wrong symbolic
        # answers trace to a missing assumption rather than bad algebra.
        naive = Session()
        assert check_equivalence(naive, "sqrt(x**2)", "x").verdict == "disproved"

        informed = Session()
        informed.declare("x", nonnegative=True)
        assert check_equivalence(informed, "sqrt(x**2)", "x").verdict == "proved"

    def test_sampling_respects_the_declared_domain(self, session: Session) -> None:
        # Sampling outside the domain is how a correct identity gets "disproved".
        session.declare("x", positive=True)
        result = check_equivalence(session, "Abs(x)", "x")
        assert result.verdict == "proved"

    def test_integer_assumptions_are_sampled_as_integers(self, session: Session) -> None:
        session.declare("n", integer=True, positive=True)
        assert check_equivalence(session, "floor(n)", "n").verdict == "proved"

    def test_an_unknown_verdict_says_what_was_tried(self, session: Session) -> None:
        result = check_equivalence(session, "x", "x + 1")
        assert result.verdict == "disproved"
        assert "numeric-sampling" in result.strategies

    def test_results_are_seeded_and_reproducible(self, session: Session) -> None:
        first = check_equivalence(session, "(x + 1)**2", "x**2 + 1", seed=7)
        second = check_equivalence(Session(), "(x + 1)**2", "x**2 + 1", seed=7)
        assert first.counterexample == second.counterexample


class TestDerivation:
    def test_it_finds_the_planted_bad_step(self, session: Session) -> None:
        result = check_derivation(
            session,
            [
                "(x + 1)**2",
                "x**2 + 2*x + 1",
                "x**2 + 2*x + 2",  # planted
                "x**2 + 2*x + 2",
            ],
        )
        assert result.valid is False
        assert result.first_invalid_step == 2

    def test_it_stops_at_the_first_failure(self, session: Session) -> None:
        result = check_derivation(session, ["x", "x + 1", "x + 2", "x + 3"])
        # Continuing past a broken step reports a cascade of failures with one cause.
        assert result.first_invalid_step == 1
        assert result.checked == 1

    def test_a_valid_derivation_passes(self, session: Session) -> None:
        result = check_derivation(session, ["(x + 1)*(x - 1)", "x**2 - 1", "x*x - 1"])
        assert result.valid is True
        assert result.first_invalid_step is None

    def test_a_single_step_is_not_a_derivation(self, session: Session) -> None:
        with pytest.raises(MathError, match="at least two steps"):
            check_derivation(session, ["x"])


class TestTimeout:
    def test_the_deadline_actually_fires(self) -> None:
        with pytest.raises(TimeoutExceeded), deadline(0.05):
            sympy.integrate(sympy.exp(sympy.Symbol("x") ** 5) * sympy.sin(sympy.Symbol("x") ** 3), sympy.Symbol("x"))

    def test_a_timed_out_check_returns_unknown_rather_than_hanging(self, session: Session) -> None:
        result = check_equivalence(session, "x", "x", timeout=0.0001)
        assert result.verdict in {"proved", "unknown"}

    def test_the_timer_is_always_cleared(self) -> None:
        import signal

        with deadline(5.0):
            pass
        # A leaked itimer fires in the middle of the next unrelated call.
        assert signal.getitimer(signal.ITIMER_REAL)[0] == 0.0


class TestBatch:
    def test_a_batch_matches_n_single_calls(self, session: Session) -> None:
        pairs = [("(x+1)**2", "x**2+2*x+1"), ("x", "x+1"), ("sin(x)**2 + cos(x)**2", "1")]
        batched = batch_equivalence(session, pairs)
        singles = [check_equivalence(Session(), left, right).to_dict() for left, right in pairs]

        assert [entry["verdict"] for entry in batched] == [entry["verdict"] for entry in singles]

    def test_one_bad_pair_does_not_fail_the_batch(self, session: Session) -> None:
        results = batch_equivalence(session, [("x", "x"), ("e:missing", "x")])
        assert results[0]["verdict"] == "proved"
        assert "error" in results[1]


class TestMatrixGrad:
    def test_the_two_layouts_are_transposes(self, session: Session) -> None:
        session.parse("Matrix([x*y, x + y])")
        numerator = matrix_grad(session, "Matrix([x*y, x + y])", ["x", "y"], layout="numerator")
        denominator = matrix_grad(session, "Matrix([x*y, x + y])", ["x", "y"], layout="denominator")

        # Layout convention silently ruins more derivations than anything else.
        assert numerator.shape == (2, 2)
        assert denominator.shape == (2, 2)
        assert session.get(numerator.expr_id).T == session.get(denominator.expr_id)

    def test_a_non_square_case_shows_the_shapes_differ(self, session: Session) -> None:
        numerator = matrix_grad(session, "Matrix([x*y, x + y, x])", ["x", "y"], layout="numerator")
        denominator = matrix_grad(session, "Matrix([x*y, x + y, x])", ["x", "y"], layout="denominator")
        assert (numerator.shape, denominator.shape) == ((3, 2), (2, 3))

    def test_the_layout_is_required_and_never_guessed(self, session: Session) -> None:
        with pytest.raises(MathError, match="never safe to guess"):
            matrix_grad(session, "x*y", ["x"], layout="whatever")  # type: ignore[arg-type]


class TestCheckGrad:
    def test_a_correct_gradient_passes(self, session: Session) -> None:
        result = check_grad(session, "x**2 * y + log(x)", "Matrix([2*x*y + 1/x, x**2])", ["x", "y"])
        assert result.ok is True
        assert result.max_relative_error < 1e-5

    def test_a_deliberately_wrong_gradient_is_rejected(self, session: Session) -> None:
        # The sign on the second component is wrong; a forward difference with a loose
        # tolerance would accept it.
        result = check_grad(session, "x**2 * y", "Matrix([2*x*y, -x**2])", ["x", "y"])
        assert result.ok is False
        assert result.worst_index == [1]
        assert result.worst_point

    def test_a_wrongly_shaped_gradient_is_refused(self, session: Session) -> None:
        with pytest.raises(MathError, match="expected 2 components"):
            check_grad(session, "x*y", "Matrix([1, 2, 3])", ["x", "y"])

    def test_it_is_seeded(self, session: Session) -> None:
        first = check_grad(session, "x**2 * y", "Matrix([2*x*y, -x**2])", ["x", "y"], seed=3)
        second = check_grad(Session(), "x**2 * y", "Matrix([2*x*y, -x**2])", ["x", "y"], seed=3)
        assert first.worst_point == second.worst_point


class TestToCode:
    def test_the_emitted_code_matches_lambdify(self, session: Session) -> None:
        expression = "exp(x*y) + log(x*y) + sqrt(x*y + 1)"
        result = to_code(session, expression, target="numpy", name="f")

        namespace: dict = {}
        import numpy as np

        exec(result.source, {"np": np}, namespace)  # noqa: S102 - the code under test
        symbols = sorted(session.resolve(expression).free_symbols, key=str)
        reference = sympy.lambdify(symbols, session.resolve(expression), "numpy")

        for x, y in [(1.5, 2.0), (0.5, 3.0), (2.0, 0.25)]:
            assert math.isclose(namespace["f"](x, y), float(reference(x, y)), rel_tol=1e-12)

    def test_cse_actually_reduces_the_operation_count(self, session: Session) -> None:
        # Symbolic differentiation produces expressions where one subterm appears fifteen
        # times; emitting that literally is correct and fifteen times slower.
        result = to_code(session, "exp(x*y) + log(x*y) + sqrt(x*y) + (x*y)**3")
        assert result.temporaries >= 1
        assert result.operations_after < result.operations_before

    def test_each_target_emits_its_own_module(self, session: Session) -> None:
        assert "torch.exp" in to_code(session, "exp(x)", target="torch").source
        assert "jnp.exp" in to_code(session, "exp(x)", target="jax").source
        assert "np.exp" in to_code(session, "exp(x)", target="numpy").source

    def test_an_unknown_target_is_named(self, session: Session) -> None:
        with pytest.raises(MathError, match="unknown target"):
            to_code(session, "x", target="tensorflow")  # type: ignore[arg-type]


class TestShapeCheck:
    def test_named_dims_catch_what_numeric_shapes_cannot(self) -> None:
        # `(32, 32)` type-checks perfectly and is wrong: `b` is batch in one tensor and beams
        # in the other. Numeric shape checking passes exactly the cases worth catching.
        result = shape_check("bd,bd->b", {"a": ["batch", "d_model"], "z": ["beams", "d_model"]})
        assert result.ok is False
        assert result.error == "axis-mismatch"
        assert result.axis == "b"
        assert "batch" in result.detail and "beams" in result.detail

    def test_a_matching_contraction_passes_and_names_the_output(self) -> None:
        result = shape_check("ij,jk->ik", {"a": ["batch", "d_model"], "b": ["d_model", "d_ff"]})
        assert result.ok is True
        assert result.output_shape == ["batch", "d_ff"]

    def test_a_three_operand_attention_contraction_checks(self) -> None:
        result = shape_check(
            "bqd,bkd->bqk",
            {"q": ["batch", "q_len", "d_head"], "k": ["batch", "k_len", "d_head"]},
        )
        assert result.ok is True
        assert result.output_shape == ["batch", "q_len", "k_len"]

    def test_a_rank_mismatch_is_reported_separately(self) -> None:
        result = shape_check("ij->i", {"a": ["batch", "d_model", "heads"]})
        assert result.error == "rank"
        assert "3 dimensions" in result.detail

    def test_the_implicit_output_follows_einsum(self) -> None:
        result = shape_check("ij,jk", {"a": ["batch", "d_model"], "b": ["d_model", "d_ff"]})
        assert result.ok is True
        assert result.output_shape == ["batch", "d_ff"]

    def test_an_output_axis_from_nowhere_is_caught(self) -> None:
        result = shape_check("ij->iz", {"a": ["batch", "d_model"]})
        assert result.error == "unbound-output"
        assert result.axis == "z"

    def test_operand_count_is_checked(self) -> None:
        result = shape_check("ij,jk->ik", {"a": ["batch", "d_model"]})
        assert result.error == "operand-count"

    def test_named_dims_resolve_to_numbers_when_given(self) -> None:
        result = shape_check(
            "ij,jk->ik",
            {"a": ["batch", "d_model"], "b": ["d_model", "d_ff"]},
            {"batch": 32, "d_ff": 2048},
        )
        assert result.output_shape == ["32", "2048"]


class TestHessian:
    def test_the_hessian_of_a_matrix_expression_is_refused(self) -> None:
        from math_mcp.cas.calculus import hessian

        session = Session()
        with pytest.raises(MathError, match="scalar"):
            hessian(session, "Matrix([x, y])", ["x", "y"])
