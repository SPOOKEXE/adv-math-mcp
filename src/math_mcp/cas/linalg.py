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

from typing import Any, Literal

import sympy

from .equivalence import TimeoutExceeded, deadline
from .session import MathError, Session, pretty, render_expr

LinalgOp = Literal["det", "rank", "eigen", "inverse", "nullspace", "solve", "decompose"]
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
    payload: dict[str, Any] = {
        "expr_id": session.store(matrix),
        "shape": list(matrix.shape),
        "pretty": pretty(matrix, limit=400),
    }
    if render:
        payload["render"] = render_expr(matrix)
    return {name: payload}


def linalg(
    session: Session,
    op: LinalgOp,
    matrix: list[list[str]],
    *,
    rhs: list[str] | None = None,
    method: Decomposition = "LU",
    vectors: bool = False,
    timeout: float = 10.0,
    render: bool = False,
) -> dict[str, Any]:
    """One verb over the matrix operations; ``op`` picks the machinery."""
    if op not in ("det", "rank", "eigen", "inverse", "nullspace", "solve", "decompose"):
        raise MathError(f"unknown linalg op `{op}`")

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
                        {
                            "value": str(eigenvalue),
                            "vectors": [[str(entry) for entry in vector] for vector in basis],
                        }
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
