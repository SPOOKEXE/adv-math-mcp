"""Solving and simplification.

**Solutions come verified.** ``solve`` substitutes each solution back and reports a per-solution
verdict, because sympy occasionally returns spurious roots (radical equations, piecewise edges)
and a caller acting on an unchecked root is acting on a guess. The verdict vocabulary is the
same as ``check_equivalence``: ``proved`` | ``unknown``, never a bare boolean.

**Simplification returns candidates, not a winner.** "Simplest" depends on what the caller is
about to do (``factor`` for roots, ``expand`` for coefficient reading, ``partfrac`` for
integration), so every strategy's result comes back with an operation count and the caller
picks. Every candidate is a sound rewrite of the input, so no equivalence check is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import sympy

from .equivalence import TimeoutExceeded, deadline
from .session import MathError, Session, canonical_form, pretty, render_expr

SolveKind = Literal["equation", "inequality", "system", "diophantine", "recurrence", "ode"]

#: Strategy → rewriter. Each entry is a sound transformation: it may fail or stall (hence the
#: deadline per strategy), but it never changes the value, which is why the results need no
#: equivalence check.
STRATEGIES: dict[str, Any] = {
    "auto": sympy.simplify,
    "factor": sympy.factor,
    "expand": sympy.expand,
    "cancel": sympy.cancel,
    "together": sympy.together,
    "trig": sympy.trigsimp,
    "radical": sympy.radsimp,
    "partfrac": None,  # needs a variable; handled in `simplify`
}


def as_relation(session: Session, text: str, *, functions: tuple[str, ...] = ()) -> sympy.Basic:
    """Parse ``lhs = rhs`` into an ``Eq``, and comparisons as themselves.

    ``==`` is folded to ``=`` first: Python evaluates ``a == b`` on sympy objects to a plain
    bool, which silently discards the equation. ``<`` and friends build relationals and pass
    through the parser unharmed.
    """
    text = text.replace("==", "=")
    if "=" in text and "<=" not in text and ">=" not in text and "!=" not in text:
        left, right = text.split("=", 1)
        return sympy.Eq(
            session.parse(left, functions=functions)[1],
            session.parse(right, functions=functions)[1],
        )
    return session.parse(text, functions=functions)[1]


@dataclass
class SolveResult:
    kind: SolveKind
    solutions: list[dict[str, str]] = field(default_factory=list)
    #: Inequalities: the solution set as a condition rather than points.
    condition: str = ""
    #: Per-solution verdict, aligned with `solutions`.
    verified: list[str] = field(default_factory=list)
    steps: list[dict[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    render: list[dict[str, str]] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"kind": self.kind, "solutions": self.solutions}
        if self.condition:
            payload["condition"] = self.condition
        if self.verified:
            payload["verified"] = self.verified
        if self.steps:
            payload["steps"] = self.steps
        if self.warnings:
            payload["warnings"] = self.warnings
        if self.render is not None:
            payload["render"] = self.render
        return payload


def _verify_solution(equations: list[sympy.Basic], solution: dict[sympy.Symbol, Any], timeout: float) -> str:
    """Substitute a solution back. ``proved`` if every residual is zero, ``unknown`` otherwise."""
    try:
        with deadline(timeout):
            for equation in equations:
                residual = equation.lhs - equation.rhs if isinstance(equation, sympy.Eq) else equation
                value = canonical_form(residual.subs(solution))
                if value != 0 and sympy.simplify(value) != 0:
                    return "unknown"
    except (TimeoutExceeded, TypeError, ValueError):
        return "unknown"
    return "proved"


def solve(
    session: Session,
    kind: SolveKind,
    exprs: list[str],
    wrt: list[str] | None = None,
    *,
    given: dict[str, Any] | None = None,
    timeout: float = 5.0,
    steps: bool = False,
    render: bool = False,
) -> SolveResult:
    """One verb over the solvable kinds; ``kind`` picks the machinery.

    ``wrt`` is the unknowns; for ``recurrence`` and ``ode`` it is ``[function, variable]``.
    ``given`` carries initial conditions (``{"a(0)": "1"}``) for those two kinds.
    """
    if kind not in ("equation", "inequality", "system", "diophantine", "recurrence", "ode"):
        raise MathError(f"unknown solve kind `{kind}`")
    if not exprs:
        raise MathError("nothing to solve")

    result = SolveResult(kind)
    record = result.steps.append if steps else (lambda item: None)

    if kind in ("recurrence", "ode"):
        return _solve_functional(session, kind, exprs[0], wrt or [], given or {}, result, record, timeout, render)

    if kind == "inequality":
        relations = [as_relation(session, text) for text in exprs]
        symbols = [session.symbol(name) for name in wrt] if wrt else sorted(
            {symbol for relation in relations for symbol in relation.free_symbols}, key=str
        )
        record({"stage": "parsed", "expr": "; ".join(sympy.sstr(r) for r in relations)})
        try:
            with deadline(timeout):
                solved = sympy.reduce_inequalities(relations, symbols)
        except TimeoutExceeded:
            result.warnings.append(f"inequality reduction exceeded {timeout}s")
            return result
        except (NotImplementedError, TypeError, ValueError) as error:
            raise MathError(f"could not reduce: {error}") from error
        result.condition = sympy.sstr(solved)
        if render:
            result.render = [render_expr(solved)]
        return result

    if kind == "diophantine":
        relation = as_relation(session, exprs[0])
        residual = relation.lhs - relation.rhs if isinstance(relation, sympy.Eq) else relation
        record({"stage": "parsed", "expr": sympy.sstr(residual)})
        try:
            with deadline(timeout):
                found = sympy.diophantine(residual)
        except TimeoutExceeded:
            result.warnings.append(f"diophantine search exceeded {timeout}s")
            return result
        except (NotImplementedError, TypeError, ValueError) as error:
            raise MathError(f"diophantine solver: {error}") from error
        names = sorted(residual.free_symbols, key=str)
        for entry in sorted(found, key=str):
            values = entry if isinstance(entry, tuple) else (entry,)
            result.solutions.append({str(name): str(value) for name, value in zip(names, values)})
        if not result.solutions:
            result.warnings.append("no integer solutions found")
        return result

    # equation and system share one path: parse, solve, verify each root by substitution.
    relations = [as_relation(session, text) for text in exprs]
    symbols = [session.symbol(name) for name in wrt] if wrt else sorted(
        {symbol for relation in relations for symbol in relation.free_symbols}, key=str
    )
    record({"stage": "parsed", "expr": "; ".join(sympy.sstr(r) for r in relations)})

    try:
        with deadline(timeout):
            found = sympy.solve(relations, symbols, dict=True)
    except TimeoutExceeded:
        result.warnings.append(f"solve exceeded {timeout}s")
        return result
    except (NotImplementedError, TypeError, ValueError) as error:
        raise MathError(f"solver: {error}") from error

    rendered: list[dict[str, str]] = []
    for solution in found:
        result.solutions.append({str(symbol): str(value) for symbol, value in solution.items()})
        result.verified.append(_verify_solution(relations, solution, timeout))
        if render:
            rendered.extend(render_expr(value) for value in solution.values())
        record({"stage": "solved", "expr": ", ".join(f"{k} = {v}" for k, v in sorted(solution.items(), key=lambda kv: str(kv[0])))})
    if not found:
        result.warnings.append("no solutions found; the system may be inconsistent or beyond the solver")
    if render:
        result.render = rendered
    return result


def _solve_functional(
    session: Session,
    kind: SolveKind,
    text: str,
    wrt: list[str],
    given: dict[str, Any],
    result: SolveResult,
    record: Any,
    timeout: float,
    render: bool,
) -> SolveResult:
    """Recurrences and ODEs: the unknown is a function, so parsing needs its name up front."""
    if len(wrt) != 2:
        raise MathError(f"`{kind}` needs wrt = [function, variable], e.g. ['a', 'n']")
    function_name, variable_name = wrt
    function = sympy.Function(function_name)
    variable = session.symbol(variable_name)

    relation = as_relation(session, text, functions=(function_name,))
    equation = relation if isinstance(relation, sympy.Eq) else sympy.Eq(relation, 0)
    record({"stage": "parsed", "expr": sympy.sstr(equation)})

    conditions = {
        session.parse(key, functions=(function_name,))[1]: session.parse(str(value))[1]
        for key, value in given.items()
    }

    try:
        with deadline(timeout):
            if kind == "recurrence":
                solved = sympy.rsolve(equation, function(variable), conditions or None)
            else:
                solved = sympy.dsolve(equation, function(variable), ics=conditions or None)
    except TimeoutExceeded:
        result.warnings.append(f"{kind} solve exceeded {timeout}s")
        return result
    except (NotImplementedError, TypeError, ValueError) as error:
        raise MathError(f"{kind} solver: {error}") from error

    if solved is None:
        result.warnings.append(f"the {kind} has no solution in the forms rsolve knows")
        return result

    if isinstance(solved, sympy.Eq):
        result.solutions.append({sympy.sstr(solved.lhs): str(solved.rhs)})
        body = solved.rhs
    else:
        result.solutions.append({f"{function_name}({variable_name})": str(solved)})
        body = solved
    record({"stage": "solved", "expr": sympy.sstr(body)})
    if render:
        result.render = [render_expr(body)]
    return result


@dataclass
class SimplifyResult:
    best: dict[str, Any]
    candidates: list[dict[str, Any]] = field(default_factory=list)
    render: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"best": self.best, "candidates": self.candidates}
        if self.render is not None:
            payload["render"] = self.render
        return payload


def simplify(
    session: Session,
    expr: str,
    strategies: list[str] | None = None,
    wrt: str | None = None,
    *,
    timeout: float = 5.0,
    render: bool = False,
) -> SimplifyResult:
    """Run the named strategies (all of them by default) and rank by operation count.

    A strategy that fails or stalls is dropped, not fatal: the input itself is always a
    candidate, so the worst case is "nothing improved it", reported as exactly that.
    """
    value = session.resolve(expr)
    chosen = strategies or list(STRATEGIES)
    unknown = set(chosen) - set(STRATEGIES)
    if unknown:
        raise MathError(f"unknown strategy(s): {', '.join(sorted(unknown))}; expected {', '.join(sorted(STRATEGIES))}")

    variable = session.symbol(wrt) if wrt else None
    if variable is None and len(value.free_symbols) == 1:
        variable = next(iter(value.free_symbols))

    # The rewritten expression rides beside each candidate dict. Handles are content-addressed
    # by *canonical* form, so every candidate here shares one handle and `session.get` returns
    # whichever form was stored last, which is useless for a tool whose entire output is the
    # written form. Anything derived from the written form must come from the local object.
    candidates: list[tuple[dict[str, Any], sympy.Expr]] = [
        ({"strategy": "input", "expr_id": session.store(value), "pretty": pretty(value), "ops": sympy.count_ops(value)}, value)
    ]
    for name in chosen:
        rewriter = STRATEGIES[name]
        try:
            with deadline(timeout):
                if name == "partfrac":
                    if variable is None:
                        continue  # apart needs a variable; with several and no `wrt` there is no right guess
                    rewritten = sympy.apart(value, variable)
                else:
                    rewritten = rewriter(value)
        except (TimeoutExceeded, NotImplementedError, sympy.PolynomialError, TypeError, ValueError):
            continue
        candidates.append(
            ({"strategy": name, "expr_id": session.store(rewritten), "pretty": pretty(rewritten), "ops": sympy.count_ops(rewritten)}, rewritten)
        )

    candidates.sort(key=lambda entry: (entry[0]["ops"], entry[0]["strategy"]))
    best, best_expr = candidates[0]
    result = SimplifyResult(best, [entry for entry, _ in candidates])
    if render:
        result.render = render_expr(best_expr)
    return result
