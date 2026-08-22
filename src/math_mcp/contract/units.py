"""Dimensional analysis over the ``units`` strings variables already carry.

Free-form unit algebra rather than a fixed SI table, because the units that matter here are as
often ``tokens/step`` and ``params`` as ``m/s^2``. The check is internal consistency, not
conformance to a catalogue. A unit string parses to a monomial over unit symbols; a formula is
checked by substituting each variable with its unit monomial and asking two questions:

* every top-level added term must carry the same units: ``m + m/s`` is the finding;
* every function argument must be dimensionless: ``exp(3 joules)`` is not a quantity.

A formula mentioning any variable with no recorded units is skipped, not guessed at: an
inferred unit that happens to make the check pass is worse than no check.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import sympy
from sympy.parsing.sympy_parser import parse_expr

from ..cas.session import _FORBIDDEN, RESTRICTED_GLOBALS, TRANSFORMATIONS, MathError

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, types only
    from .model import Scope

#: Spellings of "no units". Distinct from an *empty* units field, which means "not recorded".
DIMENSIONLESS = frozenset({"1", "dimensionless", "none", "-"})


def parse_units(text: str) -> sympy.Expr:
    """A unit string as a monomial over prefixed symbols.

    Prefixed (``m`` → ``u_m``) so a unit named ``m`` can never collide with a variable named
    ``m`` when both end up in the same substituted expression.
    """
    text = text.strip()
    if not text or text.lower() in DIMENSIONLESS:
        return sympy.Integer(1)
    if _FORBIDDEN.search(text):
        raise MathError(f"`{text}` is not a unit expression")
    try:
        parsed = parse_expr(
            text,
            local_dict={},
            transformations=TRANSFORMATIONS,
            global_dict=dict(RESTRICTED_GLOBALS),
            evaluate=True,
        )
    except (SyntaxError, TypeError, ValueError, AttributeError, NameError) as error:
        raise MathError(f"`{text}` does not parse as units: {error}") from error
    return parsed.subs({symbol: sympy.Symbol(f"u_{symbol}", positive=True) for symbol in parsed.free_symbols})


def unit_witnesses(scope: Scope) -> list[dict[str, Any]]:
    """Dimensional findings across a scope's formulas, as plain dicts for ``audit`` to wrap.

    Each entry: ``{"formulas": [id], "values": {...}, "detail": str}``.
    """
    witnesses: list[dict[str, Any]] = []

    units_of: dict[str, sympy.Expr] = {}
    for name, variable in scope.variables.items():
        if not variable.units:
            continue
        try:
            units_of[name] = parse_units(variable.units)
        except MathError as error:
            witnesses.append({"formulas": [], "values": {name: variable.units}, "detail": str(error)})

    if not units_of:
        return witnesses

    for formula_id, formula in sorted(scope.formulas.items()):
        try:
            expression = scope._parse(formula.expression)
        except MathError:
            continue  # unparseable formulas are audit's orphan problem, not a units problem

        mentioned = {scope.canonical(str(symbol)): symbol for symbol in expression.free_symbols}
        if not mentioned or any(name not in units_of for name in mentioned):
            continue

        # `x` becomes `x * units(x)`, keeping the variable in place: substituting the bare
        # monomial lets same-unit terms cancel numerically (`E - m*c**2` goes to zero), which
        # hides the mismatched term standing next to them.
        substituted = expression.subs({symbol: symbol * units_of[name] for name, symbol in mentioned.items()})
        unit_symbols = substituted.free_symbols - set(mentioned.values())

        # Function arguments must be dimensionless: sin, exp and log of a quantity with units
        # is a category error no cancellation later can repair.
        for application in substituted.atoms(sympy.Function):
            for argument in application.args:
                if sympy.simplify(argument).free_symbols & unit_symbols:
                    witnesses.append(
                        {
                            "formulas": [formula_id],
                            "values": {str(application.func): sympy.sstr(argument)},
                            "detail": (
                                f"`{formula_id}` passes a dimensionful argument to "
                                f"{application.func}; transcendental arguments must be dimensionless"
                            ),
                        }
                    )
                    break

        terms = sympy.Add.make_args(sympy.expand(substituted))
        units_seen: dict[str, sympy.Expr] = {}
        for term in terms:
            if term.is_number:
                # A bare numeral's units are unknowable, not dimensionless: `x = 16` is a value
                # assignment, and flagging every such assignment makes the report unreadable.
                continue
            _coefficient, unit_part = term.as_independent(*unit_symbols)
            key = sympy.sstr(sympy.powsimp(unit_part))
            units_seen.setdefault(key, unit_part)
        if len(units_seen) > 1:
            # Two added terms with different units. The ratio test below trims false alarms
            # from unsimplified but equal monomials.
            distinct = list(units_seen.values())
            genuinely_different = any(
                sympy.simplify(distinct[0] / candidate).free_symbols for candidate in distinct[1:]
            )
            if genuinely_different:
                witnesses.append(
                    {
                        "formulas": [formula_id],
                        "values": {f"term_{index}": sympy.sstr(part).replace("u_", "") for index, part in enumerate(distinct)},
                        "detail": f"`{formula_id}` adds terms with different units",
                    }
                )

    return witnesses
