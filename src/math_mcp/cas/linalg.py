"""Symbolic linear algebra over explicit matrices.

Entries are strings through the session parser, so a matrix can hold ``cos(t)`` next to ``1/2``
and inherits every declared assumption. Results that can be cheap to check are checked:
``inverse`` multiplies back to the identity and ``solve`` substitutes into the residual, both
reported as a ``verified`` verdict. Sympy is rarely wrong here, but "rarely" is exactly the
case worth catching, and the check costs one multiplication.

The result shape is op-dependent, so this module returns plain dicts rather than forcing seven
shapes through one dataclass.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import sympy

from .equivalence import TimeoutExceeded, deadline
from .session import MathError, Session, pretty, render_expr

LinalgOp = Literal["det", "rank", "eigen", "inverse", "nullspace", "solve", "decompose", "svd", "condition", "structure"]
Decomposition = Literal["LU", "QR", "cholesky"]


def _parse_matrix(session: Session, rows: list[list[str]]) -> sympy.Matrix:
    if not rows or not rows[0]:
        raise MathError("the matrix is empty")
    width = len(rows[0])
    for index, row in enumerate(rows):
        if len(row) != width:
            raise MathError(f"row {index} has {len(row)} entries but row 0 has {width}; a matrix is rectangular")
    return sympy.Matrix([[session.resolve(str(entry)) for entry in row] for row in rows])


def _matrix_payload(session: Session, name: str, matrix: sympy.Matrix, render: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {"expr_id": session.store(matrix), "shape": list(matrix.shape), "pretty": pretty(matrix, limit=400)}
    if render:
        payload["render"] = render_expr(matrix)
    return {name: payload}


def _truth(value: bool | None) -> str:
    if value is True:
        return "proved"
    if value is False:
        return "disproved"
    return "unknown"


def _zero_status(entry: sympy.Expr) -> bool | None:
    if entry == 0 or entry.is_zero is True:
        return True
    if entry.is_zero is False:
        return False
    simplified = sympy.simplify(entry)
    if simplified == 0:
        return True
    return simplified.is_zero


def _structure(value: sympy.Matrix) -> dict[str, Any]:
    """Exact structural facts, with unknown kept distinct from false."""
    zero_states = [[_zero_status(value[row, column]) for column in range(value.cols)] for row in range(value.rows)]
    known_nonzero = sum(state is False for row in zero_states for state in row)
    unknown = sum(state is None for row in zero_states for state in row)

    toeplitz: bool | None = True
    for row in range(1, value.rows):
        for column in range(1, value.cols):
            status = _zero_status(value[row, column] - value[row - 1, column - 1])
            if status is False:
                toeplitz = False
                break
            if status is None:
                toeplitz = None
        if toeplitz is False:
            break

    lower_band = 0
    upper_band = 0
    band_known = unknown == 0
    if band_known:
        for row in range(value.rows):
            for column in range(value.cols):
                if zero_states[row][column] is False:
                    lower_band = max(lower_band, row - column)
                    upper_band = max(upper_band, column - row)

    diagonal = (
        all(zero_states[row][column] is True for row in range(value.rows) for column in range(value.cols) if row != column)
        if unknown == 0
        else value.is_diagonal()
    )
    symmetric = value.is_symmetric() if value.is_square else False
    exact_rank = int(value.rank())
    low_rank = exact_rank < min(value.shape)
    algorithms: list[str] = []
    if diagonal is True:
        algorithms.append("diagonal elementwise solve/inverse")
    if toeplitz is True:
        algorithms.append("Toeplitz-aware matvec/solve")
    if band_known and max(lower_band, upper_band) < max(value.shape) - 1:
        algorithms.append("banded storage and banded factorization")
    if value.is_lower is True or value.is_upper is True:
        algorithms.append("triangular solve")
    if low_rank:
        algorithms.append("exact low-rank factorization")

    return {
        "diagonal": _truth(diagonal),
        "symmetric": _truth(symmetric),
        "lower_triangular": _truth(value.is_lower),
        "upper_triangular": _truth(value.is_upper),
        "toeplitz": _truth(toeplitz),
        "exact_rank": exact_rank,
        "low_rank": "proved" if low_rank else "disproved",
        "known_nonzero_entries": known_nonzero,
        "unknown_zero_status_entries": unknown,
        "sparsity": {
            "verdict": "proved" if unknown == 0 else "unknown",
            "known_zero_fraction": (value.rows * value.cols - known_nonzero - unknown) / (value.rows * value.cols),
        },
        "bandwidth": ({"verdict": "proved", "lower": lower_band, "upper": upper_band} if band_known else {"verdict": "unknown"}),
        "suggested_algorithms": algorithms,
    }


def linalg(
    session: Session,
    op: LinalgOp,
    matrix: list[list[str]],
    *,
    rhs: list[str] | None = None,
    method: Decomposition = "LU",
    vectors: bool = False,
    tolerance: float = 1e-10,
    timeout: float = 10.0,
    render: bool = False,
) -> dict[str, Any]:
    """One verb over the matrix operations; ``op`` picks the machinery."""
    if op not in ("det", "rank", "eigen", "inverse", "nullspace", "solve", "decompose", "svd", "condition", "structure"):
        raise MathError(f"unknown linalg op `{op}`")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance):
        raise MathError("tolerance must be a finite nonnegative number")
    if tolerance < 0:
        raise MathError("tolerance must be nonnegative")

    value = _parse_matrix(session, matrix)
    result: dict[str, Any] = {"op": op, "shape": list(value.shape)}

    try:
        with deadline(timeout):
            if op == "det":
                if not value.is_square:
                    raise MathError(f"determinant needs a square matrix, not {value.shape}")
                answer = sympy.simplify(value.det())
                result["det"] = str(answer)
                if render:
                    result["render"] = render_expr(answer)

            elif op == "rank":
                result["rank"] = int(value.rank())

            elif op == "eigen":
                if not value.is_square:
                    raise MathError(f"eigenvalues need a square matrix, not {value.shape}")
                result["eigenvalues"] = [
                    {"value": str(eigenvalue), "multiplicity": int(multiplicity)}
                    for eigenvalue, multiplicity in sorted(value.eigenvals().items(), key=lambda kv: str(kv[0]))
                ]
                if vectors:
                    result["eigenvectors"] = [
                        {"value": str(eigenvalue), "vectors": [[str(entry) for entry in vector] for vector in basis]}
                        for eigenvalue, _multiplicity, basis in value.eigenvects()
                    ]

            elif op == "inverse":
                if not value.is_square:
                    raise MathError(f"inversion needs a square matrix, not {value.shape}")
                try:
                    inverse = value.inv()
                except sympy.matrices.exceptions.NonInvertibleMatrixError as error:
                    raise MathError(f"the matrix is singular: {error}") from error
                result.update(_matrix_payload(session, "inverse", inverse, render))
                residual = sympy.simplify(value * inverse - sympy.eye(value.rows))
                result["verified"] = "proved" if residual == sympy.zeros(value.rows) else "unknown"

            elif op == "nullspace":
                basis = value.nullspace()
                result["nullspace"] = [[str(entry) for entry in vector] for vector in basis]
                result["nullity"] = len(basis)

            elif op == "solve":
                if rhs is None:
                    raise MathError("`solve` needs rhs, the right-hand-side vector")
                vector = sympy.Matrix([session.resolve(str(entry)) for entry in rhs])
                try:
                    solution = value.solve(vector)
                except (ValueError, sympy.matrices.exceptions.NonInvertibleMatrixError) as error:
                    raise MathError(f"no unique solution: {error}") from error
                result.update(_matrix_payload(session, "solution", solution, render))
                residual = sympy.simplify(value * solution - vector)
                result["verified"] = "proved" if residual == sympy.zeros(*vector.shape) else "unknown"

            elif op == "svd":
                left, singular, right = value.singular_value_decomposition()
                result.update(_matrix_payload(session, "U", left, render=False))
                result.update(_matrix_payload(session, "S", singular, render=False))
                result.update(_matrix_payload(session, "V", right, render=False))
                residual = sympy.simplify(left * singular * right.H - value)
                result["verified"] = "proved" if residual == sympy.zeros(*value.shape) else "unknown"
                singular_values = list(value.singular_values())
                result["singular_values"] = [str(item) for item in singular_values]
                numeric = []
                for item in singular_values:
                    try:
                        numeric.append(abs(float(sympy.N(item))))
                    except (TypeError, ValueError):
                        numeric = []
                        break
                if numeric:
                    result["numerical_rank"] = sum(item > tolerance for item in numeric)
                    result["tolerance"] = tolerance

            elif op == "condition":
                singular_values = list(value.singular_values())
                if not singular_values:
                    raise MathError("condition number needs a non-empty matrix")
                largest = sympy.Max(*singular_values)
                smallest = sympy.Min(*singular_values)
                condition = sympy.oo if smallest == 0 or smallest.is_zero is True else sympy.simplify(largest / smallest)
                result.update(
                    {
                        "norm": "2",
                        "condition": str(condition),
                        "singular_values": [str(item) for item in singular_values],
                        "ill_conditioned": ("proved" if condition is sympy.oo else "unknown"),
                    }
                )

            elif op == "structure":
                result.update(_structure(value))

            else:  # decompose
                if method == "LU":
                    lower, upper, perm = value.LUdecomposition()
                    result.update(_matrix_payload(session, "L", lower, render=False))
                    result.update(_matrix_payload(session, "U", upper, render=False))
                    result["row_swaps"] = [[int(a), int(b)] for a, b in perm]
                elif method == "QR":
                    q, r = value.QRdecomposition()
                    result.update(_matrix_payload(session, "Q", q, render=False))
                    result.update(_matrix_payload(session, "R", r, render=False))
                elif method == "cholesky":
                    try:
                        lower = value.cholesky()
                    except (ValueError, sympy.matrices.exceptions.NonSquareMatrixError) as error:
                        # Cholesky is a claim about the matrix, and its failure is the finding.
                        raise MathError(f"cholesky needs symmetric positive-definite: {error}") from error
                    result.update(_matrix_payload(session, "L", lower, render=False))
                else:
                    raise MathError(f"unknown decomposition `{method}`; use LU, QR or cholesky")
    except TimeoutExceeded:
        result["warnings"] = [f"{op} exceeded {timeout}s"]
        return result

    return result
