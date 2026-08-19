"""Calculus beyond gradients: integration, limits, series, sums and products.

**An antiderivative is checked by differentiating it back.** Integration is where a CAS is most
likely to return something subtly wrong or simply give up, and differentiation is cheap and
reliable, so every indefinite integral carries a ``verified`` verdict from the round trip.
An integral sympy could not evaluate is reported as exactly that, never returned as if the
unevaluated ``Integral(...)`` were an answer.

**Series results carry their error term** as a separate field, in the same shape the contract
layer's ``error_term`` expects: a truncation whose error is stated is an approximation; one
whose error is dropped is a lie with extra steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import sympy

from .equivalence import TimeoutExceeded, deadline
from .session import MathError, Session, canonical_form, pretty, render_expr

CalcOp = Literal["integrate", "limit", "series", "sum", "product"]


@dataclass
class CalcResult:
    op: CalcOp
    expr_id: str = ""
    pretty: str = ""
    #: Indefinite integrals: verdict from differentiating the result back.
    verified: str = ""
    #: Series: the dropped tail, e.g. ``O(x**6)``.
    error_term: str = ""
    steps: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    render: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"op": self.op, "expr_id": self.expr_id, "pretty": self.pretty}
        for key in ("verified", "error_term"):
            if getattr(self, key):
                payload[key] = getattr(self, key)
        if self.steps:
            payload["steps"] = self.steps
        if self.warnings:
            payload["warnings"] = self.warnings
        if self.render is not None:
            payload["render"] = self.render
        return payload


def calc(
    session: Session,
    op: CalcOp,
    expr: str,
    wrt: str,
    *,
    bounds: list[str] | None = None,
    point: str | None = None,
    direction: str = "both",
    order: int = 6,
    timeout: float = 10.0,
    steps: bool = False,
    render: bool = False,
) -> CalcResult:
    """One verb over the calculus operations; ``op`` picks the machinery.

    ``bounds`` makes ``integrate`` definite and is required for ``sum``/``product`` (either end
    may be symbolic, so ``["1", "n"]`` works). ``point`` is where ``limit`` and ``series``
    expand; ``direction`` is ``+`` | ``-`` | ``both`` and ``both`` that disagrees is reported
    with both one-sided values rather than picking one.
    """
    if op not in ("integrate", "limit", "series", "sum", "product"):
        raise MathError(f"unknown calc op `{op}`")

    value = session.resolve(expr)
    variable = session.symbol(wrt)
    result = CalcResult(op)
    record = result.steps.append if steps else (lambda item: None)
    record({"stage": "parsed", "expr": pretty(value)})

    def finish(answer: sympy.Expr) -> CalcResult:
        result.expr_id = session.store(answer)
        result.pretty = pretty(answer)
        record({"stage": "result", "expr": sympy.sstr(answer)})
        if render:
            result.render = render_expr(answer)
        return result

    if op == "integrate":
        try:
            with deadline(timeout):
                if bounds:
                    low, high = (session.resolve(bound) for bound in bounds)
                    answer = sympy.integrate(value, (variable, low, high))
                else:
                    answer = sympy.integrate(value, variable)
        except TimeoutExceeded:
            result.warnings.append(f"integration exceeded {timeout}s")
            return result

        if answer.has(sympy.Integral):
            # Not an answer: an unevaluated integral handed back as one is how a wrong
            # "closed form" ends up in someone's derivation.
            result.warnings.append("sympy could not evaluate this integral; no closed form returned")
            return result

        if not bounds:
            try:
                with deadline(timeout):
                    residual = canonical_form(sympy.diff(answer, variable) - value)
                    result.verified = "proved" if residual == 0 or sympy.simplify(residual) == 0 else "unknown"
            except TimeoutExceeded:
                result.verified = "unknown"
            record({"stage": "verified", "expr": result.verified})
        return finish(answer)

    if op == "limit":
        if point is None:
            raise MathError("`limit` needs a point")
        at = session.resolve(point)
        try:
            with deadline(timeout):
                if direction == "both":
                    left = sympy.limit(value, variable, at, "-")
                    right = sympy.limit(value, variable, at, "+")
                    if sympy.simplify(left - right) != 0:
                        result.warnings.append(
                            f"one-sided limits disagree: {sympy.sstr(left)} from below, {sympy.sstr(right)} from above"
                        )
                        return result
                    answer = right
                elif direction in ("+", "-"):
                    answer = sympy.limit(value, variable, at, direction)
                else:
                    raise MathError("direction must be `+`, `-` or `both`")
        except TimeoutExceeded:
            result.warnings.append(f"limit exceeded {timeout}s")
            return result
        return finish(answer)

    if op == "series":
        at = session.resolve(point) if point is not None else sympy.Integer(0)
        try:
            with deadline(timeout):
                expansion = value.series(variable, at, order)
        except TimeoutExceeded:
            result.warnings.append(f"series expansion exceeded {timeout}s")
            return result
        except (NotImplementedError, ValueError) as error:
            raise MathError(f"series: {error}") from error
        tail = expansion.getO()
        result.error_term = sympy.sstr(tail) if tail is not None else ""
        return finish(expansion.removeO())

    # sum and product
    if not bounds or len(bounds) != 2:
        raise MathError(f"`{op}` needs bounds = [low, high]; either end may be symbolic")
    low, high = (session.resolve(bound) for bound in bounds)
    try:
        with deadline(timeout):
            if op == "sum":
                answer = sympy.summation(value, (variable, low, high))
            else:
                answer = sympy.product(value, (variable, low, high))
    except TimeoutExceeded:
        result.warnings.append(f"{op} exceeded {timeout}s")
        return result
    if answer.has(sympy.Sum) or answer.has(sympy.Product):
        result.warnings.append(f"sympy could not evaluate this {op}; no closed form returned")
        return result
    return finish(answer)
