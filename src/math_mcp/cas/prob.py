"""Probability: named distributions, moments, densities and divergences.

Parameters go through the session parser, so they can be symbolic: ``Normal(mu, sigma)`` with
declared ``sigma > 0`` gives ``E[X] = mu`` exactly, which is the point: a model checking its own
derivation needs the symbolic answer, not a Monte Carlo estimate of it.

Expectations and divergences integrate, and integration hangs, so every op sits under the same
deadline the equivalence checker uses, and a timeout is an honest ``unknown``, never a partial
number.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import sympy
from sympy import stats

from .equivalence import TimeoutExceeded, deadline
from .session import MathError, Session, pretty, render_expr

ProbOp = Literal["expectation", "variance", "pdf", "cdf", "probability", "kl"]

#: family → (constructor, parameter names in order). The parameter names are the API: a caller
#: passing `{"mean": ..., "std": ...}` for a normal is matched by name, never by position.
FAMILIES: dict[str, tuple[Callable[..., Any], tuple[str, ...]]] = {
    "normal": (stats.Normal, ("mean", "std")),
    "uniform": (stats.Uniform, ("low", "high")),
    "bernoulli": (stats.Bernoulli, ("p",)),
    "binomial": (stats.Binomial, ("n", "p")),
    "poisson": (stats.Poisson, ("rate",)),
    "exponential": (stats.Exponential, ("rate",)),
    "beta": (stats.Beta, ("alpha", "beta")),
    "gamma": (stats.Gamma, ("k", "theta")),
}


def _build(session: Session, family: str, params: dict[str, Any], name: str) -> Any:
    if family not in FAMILIES:
        raise MathError(f"unknown family `{family}`; expected one of {', '.join(sorted(FAMILIES))}")
    constructor, expected = FAMILIES[family]
    missing = set(expected) - set(params)
    if missing:
        raise MathError(f"`{family}` needs parameters {', '.join(expected)}; missing {', '.join(sorted(missing))}")
    arguments = [session.resolve(str(params[key])) for key in expected]
    try:
        return constructor(name, *arguments)
    except ValueError as error:
        # sympy.stats validates (std > 0, 0 <= p <= 1, ...); its refusal is the finding.
        raise MathError(f"invalid parameters for `{family}`: {error}") from error


def prob(
    session: Session,
    op: ProbOp,
    family: str,
    params: dict[str, Any],
    *,
    expr: str | None = None,
    at: str | None = None,
    condition: str | None = None,
    other: dict[str, Any] | None = None,
    name: str = "X",
    timeout: float = 10.0,
    render: bool = False,
) -> dict[str, Any]:
    """One verb over the probability operations; ``op`` picks the machinery.

    ``expr`` and ``condition`` are written in terms of ``name`` (default ``X``): expectation of
    ``X**2``, probability of ``X > 1``. ``kl`` takes ``other = {"family": ..., "params": ...}``
    and computes KL(this ‖ other) against this distribution's support.
    """
    if op not in ("expectation", "variance", "pdf", "cdf", "probability", "kl"):
        raise MathError(f"unknown prob op `{op}`")

    variable = _build(session, family, params, name)
    placeholder = session.symbol(name)
    result: dict[str, Any] = {"op": op, "family": family}

    def finish(answer: sympy.Expr) -> dict[str, Any]:
        answer = sympy.simplify(answer)
        result["expr_id"] = session.store(answer)
        result["pretty"] = pretty(answer)
        if render:
            result["render"] = render_expr(answer)
        return result

    try:
        with deadline(timeout):
            if op == "expectation":
                target = session.resolve(expr).subs(placeholder, variable) if expr else variable
                return finish(stats.E(target))

            if op == "variance":
                target = session.resolve(expr).subs(placeholder, variable) if expr else variable
                return finish(stats.variance(target))

            if op in ("pdf", "cdf"):
                point = session.resolve(at) if at is not None else sympy.Symbol("z")
                mapping = stats.density(variable) if op == "pdf" else stats.cdf(variable)
                value = mapping(point) if callable(mapping) else mapping.get(point, sympy.Integer(0))
                return finish(value)

            if op == "probability":
                if condition is None:
                    raise MathError("`probability` needs a condition, e.g. `X > 1`")
                relation = session.parse(condition)[1]
                if not isinstance(relation, sympy.core.relational.Relational):
                    raise MathError(f"`{condition}` is not a condition; write a comparison like `{name} > 1`")
                return finish(stats.P(relation.subs(placeholder, variable)))

            # kl
            if other is None:
                raise MathError("`kl` needs other = {family, params} for the second distribution")
            second = _build(session, str(other.get("family", "")), dict(other.get("params", {})), f"{name}_q")
            p_density = stats.density(variable)
            q_density = stats.density(second)
            if not callable(p_density) or not callable(q_density):
                raise MathError("`kl` needs densities sympy can evaluate for both families")
            # E_p[log p/q]: expectation against `variable` integrates (or sums) over p's own
            # support, which is what makes this correct for one-sided families like
            # exponential and for discrete families alike.
            ratio = sympy.log(p_density(variable) / q_density(variable))
            # `rewrite(erf)` folds erfc into erf so `erf + erfc = 1` cancels; without it the
            # Gaussian KL comes back as three erf terms that are 1/2 in disguise.
            return finish(stats.E(ratio).rewrite(sympy.erf))
    except TimeoutExceeded:
        result["verdict"] = "unknown"
        result["detail"] = f"{op} exceeded {timeout}s"
        return result
    except (NotImplementedError, TypeError) as error:
        raise MathError(f"{op}: {error}") from error
