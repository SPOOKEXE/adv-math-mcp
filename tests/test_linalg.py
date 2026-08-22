"""Matrix operations: cheap checks taken, refusals that say why."""

from __future__ import annotations

import pytest

from math_mcp.cas.linalg import linalg
from math_mcp.cas.session import MathError, Session


@pytest.fixture()
def session() -> Session:
    return Session()


class TestShapesAndRefusals:
    def test_a_ragged_matrix_is_refused_with_the_row(self, session: Session) -> None:
        with pytest.raises(MathError, match="row 1"):
            linalg(session, "det", [["1", "2"], ["3"]])

    def test_det_of_a_non_square_matrix_names_the_shape(self, session: Session) -> None:
        with pytest.raises(MathError, match=r"\(1, 2\)"):
            linalg(session, "det", [["1", "2"]])

    def test_a_singular_inverse_is_an_error_not_a_matrix(self, session: Session) -> None:
        with pytest.raises(MathError, match="singular"):
            linalg(session, "inverse", [["1", "2"], ["2", "4"]])

    def test_cholesky_of_an_indefinite_matrix_names_the_requirement(self, session: Session) -> None:
        # The failure is the finding: "not positive-definite" is an answer about the matrix.
        with pytest.raises(MathError, match="positive-definite"):
            linalg(session, "decompose", [["0", "1"], ["1", "0"]], method="cholesky")


class TestVerifiedResults:
    def test_an_inverse_is_multiplied_back(self, session: Session) -> None:
        result = linalg(session, "inverse", [["1", "2"], ["3", "4"]])
        assert result["verified"] == "proved"

    def test_a_linear_solve_checks_its_residual(self, session: Session) -> None:
        result = linalg(session, "solve", [["1", "2"], ["3", "4"]], rhs=["5", "6"])
        assert result["verified"] == "proved"
        assert result["solution"]["shape"] == [2, 1]


class TestSymbolicEntries:
    def test_entries_go_through_the_session_parser(self, session: Session) -> None:
        session.declare("t", real=True)
        result = linalg(session, "det", [["cos(t)", "-sin(t)"], ["sin(t)", "cos(t)"]])
        # A rotation matrix: the determinant simplifies to 1 only if the entries really parsed.
        assert result["det"] == "1"

    def test_eigenvalues_carry_multiplicity(self, session: Session) -> None:
        result = linalg(session, "eigen", [["2", "1"], ["0", "2"]])
        assert result["eigenvalues"] == [{"value": "2", "multiplicity": 2}]

    def test_nullspace_of_a_rank_deficient_matrix(self, session: Session) -> None:
        result = linalg(session, "nullspace", [["1", "2"], ["2", "4"]])
        assert result["nullity"] == 1

    def test_lu_round_trips(self, session: Session) -> None:
        result = linalg(session, "decompose", [["4", "3"], ["6", "3"]], method="LU")
        lower = session.get(result["L"]["expr_id"])
        upper = session.get(result["U"]["expr_id"])
        product = lower * upper
        assert product == session.resolve("Matrix([[4, 3], [6, 3]])")


class TestMoreRefusals:
    def test_an_empty_matrix_is_refused(self, session: Session) -> None:
        with pytest.raises(MathError, match="empty"):
            linalg(session, "det", [])

    def test_an_unknown_op_is_refused_by_name(self, session: Session) -> None:
        with pytest.raises(MathError, match="transmogrify"):
            linalg(session, "transmogrify", [["1"]])

    def test_an_unknown_decomposition_lists_the_real_ones(self, session: Session) -> None:
        with pytest.raises(MathError, match="LU, QR or cholesky"):
            linalg(session, "decompose", [["1"]], method="SVD")

    def test_eigen_and_inverse_demand_square_matrices(self, session: Session) -> None:
        with pytest.raises(MathError, match="square"):
            linalg(session, "eigen", [["1", "2"]])
        with pytest.raises(MathError, match="square"):
            linalg(session, "inverse", [["1", "2"]])

    def test_solve_without_rhs_is_refused(self, session: Session) -> None:
        with pytest.raises(MathError, match="rhs"):
            linalg(session, "solve", [["1", "0"], ["0", "1"]])


class TestMoreOps:
    def test_rank_of_a_deficient_matrix(self, session: Session) -> None:
        assert linalg(session, "rank", [["1", "2"], ["2", "4"]])["rank"] == 1

    def test_eigenvectors_are_opt_in(self, session: Session) -> None:
        plain = linalg(session, "eigen", [["2", "0"], ["0", "3"]])
        assert "eigenvectors" not in plain
        result = linalg(session, "eigen", [["2", "0"], ["0", "3"]], vectors=True)
        assert {entry["value"] for entry in result["eigenvectors"]} == {"2", "3"}

    def test_qr_round_trips(self, session: Session) -> None:
        result = linalg(session, "decompose", [["1", "1"], ["0", "1"]], method="QR")
        q = session.get(result["Q"]["expr_id"])
        r = session.get(result["R"]["expr_id"])
        assert q * r == session.resolve("Matrix([[1, 1], [0, 1]])")

    def test_render_is_opt_in_on_scalars_and_matrices(self, session: Session) -> None:
        assert "render" not in linalg(session, "det", [["1", "2"], ["3", "4"]])
        assert linalg(session, "det", [["1", "2"], ["3", "4"]], render=True)["render"]["latex"] == "-2"
        rendered = linalg(session, "inverse", [["2", "0"], ["0", "2"]], render=True)
        assert "matrix" in rendered["inverse"]["render"]["latex"]


class TestStructureAndConditioning:
    def test_structure_finds_diagonal_banded_symmetric_and_toeplitz(self, session: Session) -> None:
        result = linalg(session, "structure", [["2", "0", "0"], ["0", "2", "0"], ["0", "0", "2"]])
        assert result["diagonal"] == "proved"
        assert result["symmetric"] == "proved"
        assert result["toeplitz"] == "proved"
        assert result["bandwidth"] == {"verdict": "proved", "lower": 0, "upper": 0}
        assert "diagonal elementwise solve/inverse" in result["suggested_algorithms"]

    def test_structure_finds_an_exact_low_rank_matrix(self, session: Session) -> None:
        result = linalg(session, "structure", [["1", "2"], ["2", "4"]])
        assert result["exact_rank"] == 1
        assert result["low_rank"] == "proved"
        assert "exact low-rank factorization" in result["suggested_algorithms"]

    def test_structure_distinguishes_false_from_unknown(self, session: Session) -> None:
        result = linalg(session, "structure", [["1", "x"], ["0", "1"]])
        assert result["symmetric"] in {"disproved", "unknown"}
        assert result["unknown_zero_status_entries"] >= 1

    def test_svd_reconstructs_and_reports_numerical_rank(self, session: Session) -> None:
        result = linalg(session, "svd", [["3", "0"], ["0", "0"]], tolerance=1e-12)
        assert result["verified"] == "proved"
        assert result["numerical_rank"] == 1
        assert result["singular_values"] == ["3", "0"]

    def test_condition_number_comes_from_singular_values(self, session: Session) -> None:
        result = linalg(session, "condition", [["4", "0"], ["0", "2"]])
        assert result["condition"] == "2"
        singular = linalg(session, "condition", [["1", "0"], ["0", "0"]])
        assert singular["condition"] == "oo"
        assert singular["ill_conditioned"] == "proved"

    def test_negative_numerical_rank_tolerance_is_refused(self, session: Session) -> None:
        with pytest.raises(MathError, match="nonnegative"):
            linalg(session, "svd", [["1"]], tolerance=-1)
        with pytest.raises(MathError, match="finite"):
            linalg(session, "svd", [["1"]], tolerance="small")  # type: ignore[arg-type]


class TestHonestTimeouts:
    def test_a_hopeless_symbolic_inverse_reports_the_deadline(self, session: Session) -> None:
        # A 7x7 fully symbolic inverse stalls sympy for well over 15s; the 50ms deadline fires
        # with a wide margin and the result says so instead of never returning.
        matrix = [[f"a{i}{j}" for j in range(7)] for i in range(7)]
        result = linalg(session, "inverse", matrix, timeout=0.05)
        assert any("exceeded" in warning for warning in result["warnings"])


class TestRemainingBranches:
    def test_cholesky_of_a_positive_definite_matrix_round_trips(self, session: Session) -> None:
        result = linalg(session, "decompose", [["4", "2"], ["2", "3"]], method="cholesky")
        lower = session.get(result["L"]["expr_id"])
        assert lower * lower.T == session.resolve("Matrix([[4, 2], [2, 3]])")

    def test_a_singular_linear_solve_is_refused_not_guessed(self, session: Session) -> None:
        with pytest.raises(MathError, match="no unique solution"):
            linalg(session, "solve", [["1", "2"], ["2", "4"]], rhs=["1", "3"])
