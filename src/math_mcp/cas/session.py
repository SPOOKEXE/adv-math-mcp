"""Parsing, assumptions and the expression workspace.

Two decisions here shape everything above them.

**Handles, not strings.** Every expression that crosses the boundary is an opaque ``expr_id``.
Passing LaTeX blobs through every call burns context on text the model already wrote, and it
re-introduces a parse error at each hop — the same expression parsed four times can fail on the
fourth. A handle parses once.

**Never ``sympify`` on a raw string.** ``sympify`` evals. Given ``__import__('os').system(...)``
it does exactly what it says, and the input here is by construction attacker-adjacent: it comes
from a model that read a web page. ``parse_expr`` with an explicit local namespace and a
restricted transformation set is the only entry point, and anything not in the namespace is
refused by name.

**Assumptions are first class**, because most wrong symbolic answers trace to a missing
assumption rather than bad algebra. ``sqrt(x**2) == x`` is false, and true for ``x >= 0``, and a
system that cannot say which is being asked answers the wrong question confidently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_application,
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

# `implicit_multiplication` lets `2x` parse and `implicit_application` lets `sin x` parse;
# `convert_xor` makes `^` mean exponentiation, which is what everyone writing maths by hand
# means by it.
#
# **`implicit_multiplication_application` is deliberately not used.** It bundles `split_symbols`,
# which rewrites any multi-letter name into a product of one-letter symbols — `collections`
# becomes `c*o*l*l*e*c*t*i*o*n*s`, and `batch` becomes five symbols. In a system whose entire
# premise is that names carry meaning, that is not a parse, it is silent destruction of the
# input, and it produces expressions that are syntactically fine and refer to nothing.
TRANSFORMATIONS = (*standard_transformations, implicit_multiplication, implicit_application, convert_xor)

#: Functions a caller may name. The allowlist *is* the security boundary — a name absent from
#: it is refused rather than resolved against the sympy namespace.
ALLOWED_FUNCTIONS: dict[str, Any] = {
    name: getattr(sympy, name)
    for name in (
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
        "exp", "log", "sqrt", "Abs", "sign", "floor", "ceiling",
        "gamma", "factorial", "binomial", "erf",
        "Sum", "Product", "Integral", "Derivative", "Matrix", "Piecewise",
        "Min", "Max", "Rational", "pi", "E", "I", "oo", "zoo", "nan",
    )
}

#: The globals `parse_expr` needs to evaluate its own transformed output.
#:
#: It cannot be empty. `parse_expr` rewrites `x**2 + 1` into `Symbol('x')**Integer(2) +
#: Integer(1)` and then evaluates *that*, so `Symbol` and the numeric constructors have to be
#: reachable — an empty global dict fails on every input. Listing exactly what the transformer
#: emits keeps the boundary intact: the ambient sympy namespace, and with it everything
#: importable, stays out of reach.
RESTRICTED_GLOBALS: dict[str, Any] = {
    "Symbol": sympy.Symbol,
    "Integer": sympy.Integer,
    "Float": sympy.Float,
    "Rational": sympy.Rational,
}

ASSUMPTION_KEYS = frozenset(
    {
        "real", "integer", "rational", "complex", "positive", "negative",
        "nonnegative", "nonpositive", "nonzero", "even", "odd", "prime", "finite",
    }
)

# Anything that could reach the interpreter. Checked before parsing so the error names the
# construct rather than surfacing as an obscure sympy failure.
_FORBIDDEN = re.compile(r"__|\bimport\b|\bexec\b|\beval\b|\bopen\b|\bcompile\b|\bglobals\b|\blambda\b")


class MathError(Exception):
    """Base for every error this package raises with a message meant for a model."""


@dataclass
class ParseError(MathError):
    """A parse failure that says *where*.

    Column positions are the difference between a model fixing its own expression and it
    rewriting the whole thing from scratch, which usually reintroduces the same mistake.
    """

    message: str
    line: int
    column: int
    source: str

    def __str__(self) -> str:  # pragma: no cover - formatting only
        return f"{self.message} at line {self.line}, column {self.column}"

    def to_dict(self) -> dict[str, Any]:
        caret = " " * max(0, self.column - 1) + "^"
        return {
            "error": "parse",
            "message": self.message,
            "line": self.line,
            "column": self.column,
            "excerpt": f"{self.source.splitlines()[self.line - 1] if self.source else ''}\n{caret}",
        }


@dataclass(frozen=True)
class Assumptions:
    """What is known about a symbol, as sympy assumption flags."""

    flags: dict[str, bool] = field(default_factory=dict)

    def merged(self, other: dict[str, bool]) -> Assumptions:
        return Assumptions({**self.flags, **other})


def _position_of(source: str, offset: int) -> tuple[int, int]:
    prefix = source[:offset]
    line = prefix.count("\n") + 1
    column = offset - (prefix.rfind("\n") + 1) + 1
    return line, column


def _latex_to_ascii(text: str) -> str:
    """Fold the LaTeX a model actually writes into parser syntax.

    Not a LaTeX implementation — a normalisation of the handful of constructs that appear in
    every derivation. Anything beyond it fails at the parser with a position, which is a better
    outcome than a half-parse that silently drops a term.
    """
    out = text
    out = re.sub(r"\\left|\\right", "", out)
    out = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", out)
    out = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", out)
    out = re.sub(r"\\(sin|cos|tan|exp|log|ln|sinh|cosh|tanh|min|max)\b", r"\1", out)
    out = out.replace("ln(", "log(")
    out = re.sub(r"\\cdot|\\times", "*", out)
    out = re.sub(r"\\(alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|phi|psi|omega)\b", r"\1", out)
    out = out.replace("{", "(").replace("}", ")")
    out = out.replace("\\", "")
    return out


class Session:
    """The shared expression workspace.

    One per MCP session. Holds the symbol table with its assumptions and the handle map, so
    ``check_equivalence(a, b)`` is two ids rather than two strings that have to agree about what
    ``x`` means.
    """

    def __init__(self) -> None:
        self._symbols: dict[str, sympy.Symbol] = {}
        self._assumptions: dict[str, Assumptions] = {}
        self._exprs: dict[str, sympy.Expr] = {}

    # -- symbols ---------------------------------------------------------------

    @property
    def symbols(self) -> dict[str, sympy.Symbol]:
        return dict(self._symbols)

    def declare(self, name: str, **flags: bool) -> sympy.Symbol:
        """Declare or refine a symbol's assumptions.

        Refining rebuilds the symbol, and **every stored expression is rewritten** to use the
        new one. Sympy's symbols compare by name *and* assumptions, so leaving old expressions
        holding the old symbol produces the worst possible failure mode: ``x`` that is not equal
        to ``x``, with no error anywhere.
        """
        unknown = set(flags) - ASSUMPTION_KEYS
        if unknown:
            raise MathError(f"unknown assumption(s): {', '.join(sorted(unknown))}")

        merged = self._assumptions.get(name, Assumptions()).merged({k: bool(v) for k, v in flags.items()})
        symbol = sympy.Symbol(name, **merged.flags)

        old = self._symbols.get(name)
        self._symbols[name] = symbol
        self._assumptions[name] = merged

        if old is not None and old != symbol:
            self._exprs = {key: value.subs(old, symbol) for key, value in self._exprs.items()}
        return symbol

    def symbol(self, name: str) -> sympy.Symbol:
        if name not in self._symbols:
            self._symbols[name] = sympy.Symbol(name)
            self._assumptions[name] = Assumptions()
        return self._symbols[name]

    def assumptions_of(self, name: str) -> dict[str, bool]:
        return dict(self._assumptions.get(name, Assumptions()).flags)

    def declarations(self) -> dict[str, dict[str, bool]]:
        """Every declared symbol and its flags, for saving.

        Only the flags. The symbols themselves are SymPy objects rebuilt by ``declare``, and the
        stored expression handles are deliberately left out: an ``expr_id`` means something to
        one running session and nothing to the next, so persisting them would restore names for
        expressions nobody can reach.
        """
        return {name: dict(assumptions.flags) for name, assumptions in sorted(self._assumptions.items())}

    def restore(self, declarations: dict[str, dict[str, bool]]) -> None:
        """Re-declare a saved symbol table.

        Through ``declare`` rather than by writing the dictionaries, so the rebuilt symbols are
        the ones SymPy would have made — a symbol assembled by hand compares unequal to the same
        symbol built by `declare`, and the failure is `x != x` with no error anywhere.
        """
        for name, flags in declarations.items():
            self.declare(name, **{key: bool(value) for key, value in flags.items()})

    # -- parsing ---------------------------------------------------------------

    def parse(
        self, text: str, *, syntax: str = "auto", functions: tuple[str, ...] = ()
    ) -> tuple[str, sympy.Expr]:
        """Parse to a handle. ``syntax`` is ``auto`` | ``latex`` | ``ascii``.

        ``functions`` names undefined functions (``a`` in a recurrence ``a(n+1) = 2*a(n)``,
        ``f`` in an ODE) that should parse as applications rather than be refused. An explicit
        allowlist per call, not an open namespace: the caller says which names are functions,
        and everything else stays a symbol or a refusal.
        """
        source = text
        if syntax == "latex" or (syntax == "auto" and "\\" in text):
            text = _latex_to_ascii(text)

        forbidden = _FORBIDDEN.search(text)
        if forbidden is not None:
            line, column = _position_of(source, min(forbidden.start(), len(source) - 1) if source else 0)
            raise ParseError(f"`{forbidden.group()}` is not allowed in an expression", line, column, source)

        namespace: dict[str, Any] = dict(ALLOWED_FUNCTIONS)
        namespace.update(self._symbols)
        namespace.update({name: sympy.Function(name) for name in functions})

        try:
            expr = parse_expr(
                text,
                local_dict=namespace,
                transformations=TRANSFORMATIONS,
                # The parser must not reach the ambient namespace: an undefined name has to
                # become a symbol, never resolve to something importable.
                global_dict=dict(RESTRICTED_GLOBALS),
                evaluate=True,
            )
        except SyntaxError as error:
            offset = getattr(error, "offset", None) or 1
            line, column = _position_of(source, min(max(offset - 1, 0), max(len(source) - 1, 0)))
            raise ParseError(str(error.msg or "invalid syntax"), line, column, source) from error
        except (TypeError, ValueError, AttributeError, NameError) as error:
            raise ParseError(str(error), 1, 1, source) from error

        # Free symbols the caller never declared become plain symbols, and are adopted so a
        # later `declare` refines them rather than creating a second `x`.
        for symbol in expr.free_symbols:
            name = str(symbol)
            if name not in self._symbols:
                self._symbols[name] = symbol
                self._assumptions[name] = Assumptions()

        expr = self._apply_assumptions(expr)
        return self.store(expr), expr

    def _apply_assumptions(self, expr: sympy.Expr) -> sympy.Expr:
        substitutions = {
            symbol: self._symbols[str(symbol)]
            for symbol in expr.free_symbols
            if str(symbol) in self._symbols and self._symbols[str(symbol)] != symbol
        }
        return expr.subs(substitutions) if substitutions else expr

    # -- handles ---------------------------------------------------------------

    def store(self, expr: sympy.Expr) -> str:
        """Store an expression under a content-derived handle.

        Derived from the canonical form, so the same expression written two ways gets one
        handle — which is also what makes ``canonical_form`` a usable equality test for the
        cheap structural cases.
        """
        handle = f"e:{canonical_hash(expr)}"
        self._exprs[handle] = expr
        return handle

    def get(self, handle: str) -> sympy.Expr:
        if handle not in self._exprs:
            raise MathError(f"unknown expression handle `{handle}`")
        return self._exprs[handle]

    def resolve(self, value: str) -> sympy.Expr:
        """A handle if it looks like one, otherwise parse it. The convenience path."""
        if value.startswith("e:"):
            return self.get(value)
        return self.parse(value)[1]

    def __len__(self) -> int:
        return len(self._exprs)


def canonical_form(expr: sympy.Expr) -> sympy.Expr:
    """A normal form, so structurally identical expressions compare equal.

    ``simplify`` is deliberately not used: it is slow, and it is not canonical — it can return
    different results for equal inputs depending on their written form, which is precisely what
    a canonical form must not do.
    """
    return sympy.expand(sympy.together(sympy.powsimp(expr, force=False)))


def canonical_hash(expr: sympy.Expr) -> str:
    import hashlib

    return hashlib.sha256(sympy.srepr(canonical_form(expr)).encode()).hexdigest()[:16]


def render_expr(expr: sympy.Expr) -> dict[str, str]:
    """The human face of a result: LaTeX for documents, plain text for terminals.

    Off by default at every call site, because agent turns should not pay for typesetting they
    will never show anyone. Produced here in one place so every tool renders the same way.
    """
    return {"latex": sympy.latex(expr), "text": sympy.sstr(expr)}


def pretty(expr: sympy.Expr, limit: int = 120) -> str:
    """A short human-readable form to accompany a handle.

    Truncated, because the point of the handle is that the full expression does not have to
    travel; returning it in the preview would give back the cost the handle just saved.
    """
    text = sympy.sstr(expr)
    return text if len(text) <= limit else f"{text[: limit - 1]}…"
