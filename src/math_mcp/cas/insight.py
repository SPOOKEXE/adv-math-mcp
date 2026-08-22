"""Numerical insight tools: stability, certified enclosures, complexity and error budgets.

These operations share one public verb because they answer one question about an expression:
"what can safely be inferred before evaluating it at production scale?"  Proven interval results
use a deliberately small, auditable interval evaluator. Unsupported functions return ``unknown``
instead of falling back to sampling and calling it a proof.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import sympy
from sympy.calculus.util import function_range

from .equivalence import TimeoutExceeded, deadline
from .session import MathError, Session, pretty, render_expr

ANALYSIS_OPS = frozenset({"stability", "rigorous_bounds", "complexity", "error_budget", "optimize"})


class IntervalUnknown(Exception):
    """The small certified interval evaluator does not support this expression."""


class DomainFailure(Exception):
    """The expression is not defined everywhere on the requested box."""


@dataclass(frozen=True, slots=True)
class Bound:
    low: sympy.Expr
    high: sympy.Expr

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": sympy.sstr(self.low),
            "upper": sympy.sstr(self.high),
            "lower_numeric": str(sympy.N(self.low, 16)),
            "upper_numeric": str(sympy.N(self.high, 16)),
        }


def _minimum(values: list[sympy.Expr]) -> sympy.Expr:
    return sympy.Min(*values)


def _maximum(values: list[sympy.Expr]) -> sympy.Expr:
    return sympy.Max(*values)


def _contains_zero(bound: Bound) -> bool:
    below = sympy.Le(bound.low, 0)
    above = sympy.Ge(bound.high, 0)
    # Unknown means conservatively yes: a false negative here would certify division by an
    # interval that may contain zero.
    return below is not sympy.false and above is not sympy.false


def _mul(left: Bound, right: Bound) -> Bound:
    products = [left.low * right.low, left.low * right.high, left.high * right.low, left.high * right.high]
    return Bound(_minimum(products), _maximum(products))


def _integer_power(base: Bound, exponent: int) -> Bound:
    if exponent == 0:
        return Bound(sympy.Integer(1), sympy.Integer(1))
    if exponent < 0:
        if _contains_zero(base):
            raise DomainFailure("a denominator can be zero inside the box")
        positive = _integer_power(base, -exponent)
        reciprocals = [sympy.Integer(1) / positive.low, sympy.Integer(1) / positive.high]
        return Bound(_minimum(reciprocals), _maximum(reciprocals))
    endpoints = [base.low**exponent, base.high**exponent]
    if exponent % 2 == 0 and _contains_zero(base):
        return Bound(sympy.Integer(0), _maximum(endpoints))
    return Bound(_minimum(endpoints), _maximum(endpoints))


def _interval(expr: sympy.Expr, env: dict[sympy.Symbol, Bound]) -> Bound:
    if expr.is_number:
        return Bound(expr, expr)
    if isinstance(expr, sympy.Symbol):
        if expr not in env:
            raise IntervalUnknown(f"no interval was supplied for `{expr}`")
        return env[expr]
    if isinstance(expr, sympy.Add):
        parts = [_interval(arg, env) for arg in expr.args]
        return Bound(sum((part.low for part in parts), sympy.Integer(0)), sum((part.high for part in parts), sympy.Integer(0)))
    if isinstance(expr, sympy.Mul):
        result = Bound(sympy.Integer(1), sympy.Integer(1))
        for arg in expr.args:
            result = _mul(result, _interval(arg, env))
        return result
    if isinstance(expr, sympy.Pow):
        base = _interval(expr.base, env)
        exponent = expr.exp
        if exponent.is_Integer:
            return _integer_power(base, int(exponent))
        if exponent == sympy.Rational(1, 2):
            if sympy.Lt(base.low, 0) is not sympy.false:
                raise DomainFailure("sqrt is not real everywhere inside the box")
            return Bound(sympy.sqrt(base.low), sympy.sqrt(base.high))
        raise IntervalUnknown(f"non-integer power `{expr}` is not in the certified evaluator")
    if expr.func is sympy.Abs:
        source = _interval(expr.args[0], env)
        if _contains_zero(source):
            return Bound(sympy.Integer(0), _maximum([sympy.Abs(source.low), sympy.Abs(source.high)]))
        values = [sympy.Abs(source.low), sympy.Abs(source.high)]
        return Bound(_minimum(values), _maximum(values))
    if expr.func is sympy.exp:
        source = _interval(expr.args[0], env)
        return Bound(sympy.exp(source.low), sympy.exp(source.high))
    if expr.func is sympy.log:
        source = _interval(expr.args[0], env)
        if sympy.Gt(source.low, 0) is not sympy.true:
            raise DomainFailure("log is not real and finite everywhere inside the box")
        return Bound(sympy.log(source.low), sympy.log(source.high))
    if expr.func is sympy.sin or expr.func is sympy.cos:
        # Deliberately broad and certain. Tight periodic range reduction can be added without
        # changing the contract; returning [-1, 1] cannot under-enclose.
        _interval(expr.args[0], env)
        return Bound(sympy.Integer(-1), sympy.Integer(1))
    if expr.func is sympy.Min:
        parts = [_interval(arg, env) for arg in expr.args]
        return Bound(_minimum([part.low for part in parts]), _minimum([part.high for part in parts]))
    if expr.func is sympy.Max:
        parts = [_interval(arg, env) for arg in expr.args]
        return Bound(_maximum([part.low for part in parts]), _maximum([part.high for part in parts]))
    raise IntervalUnknown(f"function `{expr.func}` is not in the certified interval evaluator")


def _box_env(session: Session, box: dict[str, list[Any]] | None, symbols: set[sympy.Symbol]) -> dict[sympy.Symbol, Bound]:
    if not box:
        raise MathError('this operation needs `box`, e.g. {"x": [-1, 1]}')
    missing = {str(symbol) for symbol in symbols} - set(box)
    if missing:
        raise MathError(f"box does not cover: {', '.join(sorted(missing))}")
    env: dict[sympy.Symbol, Bound] = {}
    for name, interval in box.items():
        if not isinstance(interval, list) or len(interval) != 2:
            raise MathError(f"box interval for `{name}` must be [low, high]")
        try:
            low = sympy.Rational(str(interval[0]))
            high = sympy.Rational(str(interval[1]))
        except (TypeError, ValueError) as error:
            raise MathError(f"box interval for `{name}` must contain finite real numbers") from error
        if low > high:
            raise MathError(f"box interval for `{name}` has low > high")
        env[session.symbol(name)] = Bound(low, high)
    return env


def _rigorous_bounds(session: Session, value: sympy.Expr, box: dict[str, list[Any]] | None) -> dict[str, Any]:
    env = _box_env(session, box, set(value.free_symbols))
    try:
        enclosure = _interval(value, env)
    except DomainFailure as error:
        return {"op": "rigorous_bounds", "verdict": "unknown", "detail": str(error), "box_covered": False}
    except IntervalUnknown as error:
        return {"op": "rigorous_bounds", "verdict": "unknown", "detail": str(error), "box_covered": True}
    return {
        "op": "rigorous_bounds",
        "verdict": "proved",
        "enclosure": enclosure.to_dict(),
        "method": "symbolic natural interval extension",
        "box_covered": True,
        "dependency_caveat": "repeated variables are enclosed independently, so the result can be loose",
    }


def _operation_counts(value: sympy.Expr) -> dict[str, int]:
    counts = {"add": 0, "multiply": 0, "power": 0, "function": 0}
    for node in sympy.preorder_traversal(value):
        if isinstance(node, sympy.Add):
            counts["add"] += max(0, len(node.args) - 1)
        elif isinstance(node, sympy.Mul):
            counts["multiply"] += max(0, len(node.args) - 1)
        elif isinstance(node, sympy.Pow):
            counts["power"] += 1
        elif isinstance(node, sympy.Function):
            counts["function"] += 1
    return counts


def _complexity(session: Session, value: sympy.Expr) -> dict[str, Any]:
    replacements, reduced = sympy.cse(value, optimizations="basic")
    before = int(sympy.count_ops(value))
    after = sum(int(sympy.count_ops(expr)) for _, expr in replacements) + sum(int(sympy.count_ops(expr)) for expr in reduced)
    return {
        "op": "complexity",
        "verdict": "proved",
        "method": "exact expression-tree count",
        "operations": _operation_counts(value),
        "operations_before_cse": before,
        "operations_after_cse": after,
        "common_subexpressions": len(replacements),
        "tree_nodes": sum(1 for _ in sympy.preorder_traversal(value)),
        "caveat": "symbolic scalar operation counts are not hardware instruction counts",
    }


def _stability(session: Session, value: sympy.Expr, wrt: list[str] | None, at: dict[str, Any] | None) -> dict[str, Any]:
    findings: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, detail: str, suggestion: str) -> None:
        key = (kind, detail)
        if key in seen:
            return
        seen.add(key)
        findings.append({"kind": kind, "detail": detail, "suggestion": suggestion})

    for node in sympy.preorder_traversal(value):
        if isinstance(node, sympy.Add) and any(arg.could_extract_minus_sign() for arg in node.args):
            add("cancellation", pretty(node), "compare term magnitudes; use a factored or dedicated stable form")
        if node.func is sympy.log and isinstance(node.args[0], sympy.Add) and sympy.Integer(1) in node.args[0].args:
            add("small-increment-log", pretty(node), "use log1p for numerical evaluation")
        if isinstance(node, sympy.Pow) and node.exp.is_negative:
            add("division", pretty(node), "bound the denominator away from zero")
        if node.func is sympy.exp:
            add("range", pretty(node), "check overflow/underflow in the target dtype")

    symbols = [session.symbol(name) for name in wrt] if wrt else sorted(value.free_symbols, key=str)
    substitutions = {session.symbol(name): session.resolve(str(raw)) for name, raw in (at or {}).items()}
    conditions: list[dict[str, Any]] = []
    for symbol in symbols:
        condition = sympy.Abs(symbol * sympy.diff(value, symbol) / value)
        record: dict[str, Any] = {"wrt": str(symbol), "relative_condition_expr": sympy.sstr(condition), "expr_id": session.store(condition)}
        if substitutions and not condition.subs(substitutions).free_symbols:
            record["at_point"] = str(sympy.N(condition.subs(substitutions), 16))
        conditions.append(record)
    return {
        "op": "stability",
        "verdict": "unknown",
        "findings": findings,
        "relative_condition": conditions,
        "detail": "structural hazards and relative condition expressions; absence of a finding is not a proof of stability",
    }


def _error_budget(
    session: Session, value: sympy.Expr, errors: dict[str, Any] | None, at: dict[str, Any] | None, box: dict[str, list[Any]] | None
) -> dict[str, Any]:
    if not errors:
        raise MathError('`error_budget` needs absolute input errors, e.g. {"x": "1e-6"}')
    terms: list[sympy.Expr] = []
    contributions: list[dict[str, str]] = []
    for name, raw_error in errors.items():
        symbol = session.symbol(name)
        error = session.resolve(str(raw_error))
        if error.is_negative is True:
            raise MathError(f"error for `{name}` must be nonnegative")
        term = sympy.Abs(sympy.diff(value, symbol)) * error
        terms.append(term)
        contributions.append({"input": name, "term": sympy.sstr(term)})
    first_order = sum(terms, sympy.Integer(0))
    result: dict[str, Any] = {
        "op": "error_budget",
        "verdict": "unknown",
        "model": "first-order absolute error propagation",
        "bound_expr": sympy.sstr(first_order),
        "expr_id": session.store(first_order),
        "contributions": contributions,
        "caveat": "higher-order remainder and correlated errors are not bounded by this model",
    }
    if at:
        substitutions = {session.symbol(name): session.resolve(str(raw)) for name, raw in at.items()}
        evaluated = first_order.subs(substitutions)
        if not evaluated.free_symbols:
            result["at_point"] = str(sympy.N(evaluated, 16))
    if box:
        env = _box_env(session, box, set(first_order.free_symbols))
        try:
            enclosure = _interval(first_order, env)
            result["coefficient_enclosure"] = enclosure.to_dict()
        except (IntervalUnknown, DomainFailure) as error:
            result["enclosure_warning"] = str(error)
    return result


def _optimize(
    session: Session, value: sympy.Expr, wrt: list[str] | None, box: dict[str, list[Any]] | None, goal: str, timeout: float
) -> dict[str, Any]:
    if goal not in {"min", "max", "both"}:
        raise MathError("optimization goal must be `min`, `max` or `both`")
    symbols = [session.symbol(name) for name in wrt] if wrt else sorted(value.free_symbols, key=str)
    env = _box_env(session, box, set(symbols))
    if len(symbols) != 1:
        enclosure_result = _rigorous_bounds(session, value, box)
        return {
            "op": "optimize",
            "verdict": "unknown",
            "detail": "global symbolic optimization is currently exact only for one variable",
            "enclosure": enclosure_result.get("enclosure"),
        }
    symbol = symbols[0]
    domain = sympy.Interval(env[symbol].low, env[symbol].high)
    try:
        with deadline(timeout):
            image = function_range(value, symbol, domain)
    except TimeoutExceeded:
        return {"op": "optimize", "verdict": "unknown", "detail": f"function range exceeded {timeout}s"}
    except (NotImplementedError, ValueError) as error:
        return {"op": "optimize", "verdict": "unknown", "detail": f"symbolic range did not close: {error}"}
    if not isinstance(image, sympy.Interval):
        return {"op": "optimize", "verdict": "unknown", "range": sympy.sstr(image), "detail": "range is not one interval"}
    result: dict[str, Any] = {
        "op": "optimize",
        "verdict": "proved",
        "method": "symbolic function range over a closed interval",
        "range": {"lower": sympy.sstr(image.inf), "upper": sympy.sstr(image.sup)},
    }
    if goal in {"min", "both"}:
        result["minimum"] = sympy.sstr(image.inf)
    if goal in {"max", "both"}:
        result["maximum"] = sympy.sstr(image.sup)
    return result


def analyze(
    session: Session,
    op: str,
    expr: str,
    *,
    wrt: list[str] | None = None,
    box: dict[str, list[Any]] | None = None,
    at: dict[str, Any] | None = None,
    errors: dict[str, Any] | None = None,
    goal: str = "both",
    timeout: float = 10.0,
    render: bool = False,
) -> dict[str, Any]:
    """Dispatch one of the compact expression-analysis operations."""
    if op not in ANALYSIS_OPS:
        raise MathError(f"unknown analysis op `{op}`; use {', '.join(sorted(ANALYSIS_OPS))}")
    try:
        timeout = float(timeout)
    except (TypeError, ValueError) as error:
        raise MathError("`timeout` must be a positive finite number") from error
    if not math.isfinite(timeout) or timeout <= 0:
        raise MathError("`timeout` must be a positive finite number")
    value = session.resolve(expr)
    if isinstance(value, sympy.MatrixBase):
        raise MathError(f"`{op}` expects a scalar expression; use `linalg` for matrices")

    if op == "stability":
        result = _stability(session, value, wrt, at)
    elif op == "rigorous_bounds":
        result = _rigorous_bounds(session, value, box)
    elif op == "complexity":
        result = _complexity(session, value)
    elif op == "error_budget":
        result = _error_budget(session, value, errors, at, box)
    else:
        result = _optimize(session, value, wrt, box, goal, timeout)
    if render:
        result["render"] = render_expr(value)
    return result
