"""Numeric evaluation: arbitrary precision, root finding, empirical bounds, convexity.

The empirical ops are labelled as such in their results. ``bounds`` reports the extremes it
*found*, over the points it *tried*. An empirical range presented as a proven one is exactly
the confident wrongness this server exists to catch, so the sample count travels with the
answer. ``convexity`` is three-way like everything else: ``proved`` when the symbolic second
derivative settles it, ``disproved`` with a witness point when a sample refutes it, and
``not-refuted``, never "yes", when sampling finds nothing.
"""

from __future__ import annotations

import random
from typing import Any, Literal

import sympy

from .equivalence import TimeoutExceeded, deadline, _sample_domain
from .session import MathError, Session, pretty

NumericOp = Literal["evalf", "root", "bounds", "convexity"]

#: Below this, an eigenvalue is treated as zero rather than as a sign.
EIGEN_TOLERANCE = 1e-9


def _substitutions(session: Session, at: dict[str, Any] | None) -> dict[sympy.Symbol, sympy.Expr]:
    return {session.symbol(name): session.resolve(str(value)) for name, value in (at or {}).items()}


def evaluate(
    session: Session,
    op: NumericOp,
    expr: str,
    *,
    at: dict[str, Any] | None = None,
    digits: int = 30,
    start: dict[str, Any] | None = None,
    box: dict[str, list[float]] | None = None,
    wrt: list[str] | None = None,
    samples: int = 200,
    seed: int = 20260810,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """One verb over the numeric operations; ``op`` picks the machinery."""
    if op not in ("evalf", "root", "bounds", "convexity"):
        raise MathError(f"unknown numeric op `{op}`")

    value = session.resolve(expr)
    result: dict[str, Any] = {"op": op}

    if op == "evalf":
        substituted = value.subs(_substitutions(session, at))
        try:
            with deadline(timeout):
                numeric = sympy.N(substituted, digits)
        except TimeoutExceeded:
            raise MathError(f"evaluation exceeded {timeout}s") from None
        if numeric.free_symbols:
            raise MathError(
                f"still symbolic after substitution: {', '.join(sorted(str(s) for s in numeric.free_symbols))} unbound"
            )
        result.update({"value": str(numeric), "digits": digits})
        return result

    if op == "root":
        if isinstance(value, sympy.Eq):
            value = value.lhs - value.rhs
        if not start:
            raise MathError("`root` needs start, an initial guess per variable, e.g. {\"x\": 1.0}")
        symbols = [session.symbol(name) for name in start]
        guesses = [sympy.Float(str(start[str(symbol)])) for symbol in symbols]
        try:
            with deadline(timeout):
                found = sympy.nsolve(value, symbols[0] if len(symbols) == 1 else symbols, guesses[0] if len(guesses) == 1 else guesses)
        except TimeoutExceeded:
            raise MathError(f"root finding exceeded {timeout}s") from None
        except (ValueError, ZeroDivisionError) as error:
            raise MathError(f"nsolve did not converge from that start: {error}") from error
        roots = list(found) if isinstance(found, sympy.MatrixBase) else [found]
        assignment = {str(symbol): str(root) for symbol, root in zip(symbols, roots)}
        residual = value.subs({symbol: root for symbol, root in zip(symbols, roots)})
        result.update({"root": assignment, "residual": str(sympy.N(residual, 6))})
        return result

    if op == "bounds":
        if not box:
            raise MathError("`bounds` needs box, an interval per variable, e.g. {\"x\": [-1, 1]}")
        return _bounds(value, session, box, samples, seed, result)

    return _convexity(value, session, wrt, box, samples, seed, timeout, result)


def _bounds(
    value: sympy.Expr,
    session: Session,
    box: dict[str, list[float]],
    samples: int,
    seed: int,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Empirical extremes over a box: random interior points plus every corner.

    Corners on purpose: for monotone expressions the extremes live there, and a uniform sample
    of the interior can miss them by a margin that looks like a real bound.
    """
    symbols = [session.symbol(name) for name in box]
    unbound = value.free_symbols - set(symbols)
    if unbound:
        raise MathError(f"box does not cover: {', '.join(sorted(str(s) for s in unbound))}")

    rng = random.Random(seed)
    points: list[dict[sympy.Symbol, float]] = []
    for _ in range(samples):
        points.append({symbol: rng.uniform(*box[str(symbol)]) for symbol in symbols})
    corner_count = 2 ** len(symbols)
    if corner_count <= 64:
        for mask in range(corner_count):
            points.append(
                {symbol: box[str(symbol)][(mask >> index) & 1] for index, symbol in enumerate(symbols)}
            )

    lowest: tuple[float, dict[str, float]] | None = None
    highest: tuple[float, dict[str, float]] | None = None
    failures = 0
    for point in points:
        try:
            numeric = float(value.subs(point))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            failures += 1
            continue
        named = {str(symbol): float(coordinate) for symbol, coordinate in point.items()}
        if lowest is None or numeric < lowest[0]:
            lowest = (numeric, named)
        if highest is None or numeric > highest[0]:
            highest = (numeric, named)

    if lowest is None or highest is None:
        raise MathError("the expression evaluated at no point in the box; it may be undefined there")

    result.update(
        {
            "empirical_min": {"value": lowest[0], "at": lowest[1]},
            "empirical_max": {"value": highest[0], "at": highest[1]},
            "points_tried": len(points),
            "points_undefined": failures,
            "detail": "empirical extremes over sampled points, not a proof",
        }
    )
    return result


def _convexity(
    value: sympy.Expr,
    session: Session,
    wrt: list[str] | None,
    box: dict[str, list[float]] | None,
    samples: int,
    seed: int,
    timeout: float,
    result: dict[str, Any],
) -> dict[str, Any]:
    symbols = [session.symbol(name) for name in wrt] if wrt else sorted(value.free_symbols, key=str)
    if not symbols:
        raise MathError("`convexity` needs at least one variable")

    hessian = sympy.hessian(value, symbols)

    # Symbolic first: a constant Hessian settles it outright, and in one dimension sympy's
    # assumption engine can often sign the second derivative from the declarations alone.
    try:
        with deadline(timeout):
            if not hessian.free_symbols:
                eigenvalues = [sympy.re(sympy.N(v)) for v in hessian.eigenvals()]
                convex = all(v >= -EIGEN_TOLERANCE for v in eigenvalues)
                concave = all(v <= EIGEN_TOLERANCE for v in eigenvalues)
                result.update(
                    {
                        "convex": "proved" if convex else "disproved",
                        "concave": "proved" if concave else "disproved",
                        "detail": "constant Hessian; eigenvalues computed exactly",
                    }
                )
                return result
            if len(symbols) == 1:
                second = sympy.simplify(hessian[0, 0])
                # Convexity is a question over the reals by definition, so undeclared symbols
                # may soundly be treated as real here: `exp(x).is_nonnegative` is None for a
                # possibly-complex `x` and True for a real one.
                second = second.subs(
                    {
                        symbol: sympy.Symbol(str(symbol), real=True)
                        for symbol in second.free_symbols
                        if symbol.is_real is None
                    }
                )
                if second.is_nonnegative:
                    result.update({"convex": "proved", "concave": "unknown", "detail": f"d²/d{symbols[0]}² = {second} ≥ 0 symbolically"})
                    return result
                if second.is_nonpositive:
                    result.update({"convex": "unknown", "concave": "proved", "detail": f"d²/d{symbols[0]}² = {second} ≤ 0 symbolically"})
                    return result
    except TimeoutExceeded:
        pass

    # Sampling: a witness disproves; absence of one is only ever "not-refuted".
    rng = random.Random(seed)
    convex_witness: dict[str, float] | None = None
    concave_witness: dict[str, float] | None = None
    tried = 0
    for _ in range(min(samples, 64)):
        if box:
            point = {symbol: sympy.Float(rng.uniform(*box[str(symbol)])) for symbol in symbols if str(symbol) in box}
            if len(point) != len(symbols):
                raise MathError("box does not cover every variable")
        else:
            point = {symbol: _sample_domain(symbol, rng) for symbol in symbols}
        try:
            numeric = hessian.subs(point).evalf()
            eigenvalues = [complex(v).real for v in numeric.eigenvals()]
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        tried += 1
        named = {str(symbol): float(coordinate) for symbol, coordinate in point.items()}
        if convex_witness is None and any(v < -EIGEN_TOLERANCE for v in eigenvalues):
            convex_witness = named
        if concave_witness is None and any(v > EIGEN_TOLERANCE for v in eigenvalues):
            concave_witness = named
        if convex_witness and concave_witness:
            break

    result["convex"] = "disproved" if convex_witness else f"not-refuted by {tried} samples"
    result["concave"] = "disproved" if concave_witness else f"not-refuted by {tried} samples"
    if convex_witness:
        result["convexity_witness"] = convex_witness
    if concave_witness:
        result["concavity_witness"] = concave_witness
    result.setdefault("detail", "Hessian eigenvalue signs at sampled points; not a proof either way")
    return result
