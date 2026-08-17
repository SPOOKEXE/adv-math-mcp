"""Matrix gradients, gradient checking, code emission and shape checking.

The four tools here exist because each catches a class of error a model produces confidently and
cannot detect by rereading its own work.

**Layout convention silently ruins more derivations than anything else.** ``∂y/∂x`` for vector
``y`` and vector ``x`` is one matrix under numerator layout and its transpose under denominator
layout, both are standard, papers routinely do not say which, and the result is a derivation
that is correct except for being transposed. The flag is required rather than defaulted.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Literal

import sympy

from .session import MathError, Session, pretty

Layout = Literal["numerator", "denominator"]
CodeTarget = Literal["numpy", "torch", "jax"]


@dataclass
class GradResult:
    layout: Layout
    shape: tuple[int, ...]
    expr_id: str
    pretty: str

    def to_dict(self) -> dict[str, Any]:
        return {"layout": self.layout, "shape": list(self.shape), "expr_id": self.expr_id, "pretty": self.pretty}


def _as_matrix(expr: sympy.Expr | sympy.Matrix) -> sympy.Matrix:
    if isinstance(expr, sympy.MatrixBase):
        return expr
    return sympy.Matrix([expr])


def matrix_grad(session: Session, expr: str, wrt: list[str], *, layout: Layout) -> GradResult:
    """Differentiate with an explicit layout.

    Numerator layout ("Jacobian"): rows index the output, columns the variable — shape
    ``(m, n)``. Denominator layout ("gradient"): the transpose. Sympy's ``jacobian`` is
    numerator layout, so denominator is that transposed, and stating it here is the whole point.
    """
    if layout not in ("numerator", "denominator"):
        raise MathError("layout must be `numerator` or `denominator`; it is never safe to guess")

    target = _as_matrix(session.resolve(expr))
    variables = sympy.Matrix([session.symbol(name) for name in wrt])

    jacobian = target.jacobian(variables)
    result = jacobian if layout == "numerator" else jacobian.T

    return GradResult(layout, tuple(result.shape), session.store(result), pretty(result))


@dataclass
class GradCheckResult:
    ok: bool
    max_relative_error: float
    worst_point: dict[str, float] = field(default_factory=dict)
    worst_index: list[int] = field(default_factory=list)
    samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "max_relative_error": self.max_relative_error,
            "worst_point": self.worst_point,
            "worst_index": self.worst_index,
            "samples": self.samples,
        }


def check_grad(
    session: Session,
    expr: str,
    claimed: str,
    wrt: list[str],
    *,
    samples: int = 8,
    seed: int = 20260810,
    step: float = 1e-6,
    tolerance: float = 1e-5,
) -> GradCheckResult:
    """Validate a claimed gradient against finite differences.

    A **central** difference, because the forward one has first-order error and would need a
    tolerance loose enough to accept a genuinely wrong gradient. Relative error, because an
    absolute threshold either rejects large-magnitude gradients or accepts wrong small ones.
    """
    function = session.resolve(expr)
    gradient = _as_matrix(session.resolve(claimed))
    symbols = [session.symbol(name) for name in wrt]

    if gradient.shape not in {(len(symbols), 1), (1, len(symbols))}:
        raise MathError(f"claimed gradient has shape {gradient.shape}, expected {len(symbols)} components")

    flat = list(gradient)
    rng = random.Random(seed)
    worst = 0.0
    worst_point: dict[str, float] = {}
    worst_index: list[int] = []

    for _ in range(samples):
        point = {symbol: rng.uniform(0.5, 2.0) for symbol in symbols}

        for index, symbol in enumerate(symbols):
            forward = dict(point)
            backward = dict(point)
            forward[symbol] = point[symbol] + step
            backward[symbol] = point[symbol] - step

            try:
                numeric = (float(function.subs(forward)) - float(function.subs(backward))) / (2 * step)
                analytic = float(flat[index].subs(point))
            except (TypeError, ValueError, ZeroDivisionError):
                continue

            error = abs(numeric - analytic) / max(1.0, abs(numeric), abs(analytic))
            if error > worst:
                worst = error
                worst_point = {str(symbol): value for symbol, value in point.items()}
                worst_index = [index]

    return GradCheckResult(worst <= tolerance, worst, worst_point, worst_index, samples)


@dataclass
class CodeResult:
    target: CodeTarget
    source: str
    operations_before: int
    operations_after: int
    temporaries: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "source": self.source,
            "operations_before": self.operations_before,
            "operations_after": self.operations_after,
            "temporaries": self.temporaries,
        }


_MODULE = {"numpy": "np", "torch": "torch", "jax": "jnp"}
_FUNCTIONS: dict[str, dict[str, str]] = {
    "numpy": {"exp": "np.exp", "log": "np.log", "sqrt": "np.sqrt", "sin": "np.sin", "cos": "np.cos", "tanh": "np.tanh", "Abs": "np.abs"},
    "torch": {"exp": "torch.exp", "log": "torch.log", "sqrt": "torch.sqrt", "sin": "torch.sin", "cos": "torch.cos", "tanh": "torch.tanh", "Abs": "torch.abs"},
    "jax": {"exp": "jnp.exp", "log": "jnp.log", "sqrt": "jnp.sqrt", "sin": "jnp.sin", "cos": "jnp.cos", "tanh": "jnp.tanh", "Abs": "jnp.abs"},
}


def _count_ops(expr: sympy.Expr) -> int:
    return sum(1 for _ in sympy.preorder_traversal(expr) if isinstance(_, (sympy.Add, sympy.Mul, sympy.Pow)))


def to_code(session: Session, expr: str, *, target: CodeTarget = "numpy", name: str = "f") -> CodeResult:
    """Emit array code, with common subexpressions extracted.

    CSE is the reason this is not a string template. Symbolic differentiation produces
    expressions where one subterm appears fifteen times, and emitting that literally is code
    that is correct and fifteen times slower than it needs to be — which nobody notices, because
    it gives the right answer.
    """
    if target not in _MODULE:
        raise MathError(f"unknown target `{target}`; expected one of {', '.join(sorted(_MODULE))}")

    value = session.resolve(expr)
    before = _count_ops(value)

    replacements, reduced = sympy.cse(value, optimizations="basic")
    body = reduced[0]

    printer = {"numpy": sympy.printing.pycode, "torch": sympy.printing.pycode, "jax": sympy.printing.pycode}[target]

    def render(item: sympy.Expr) -> str:
        text = printer(item)
        text = text.replace("math.", f"{_MODULE[target]}.")
        for source, replacement in _FUNCTIONS[target].items():
            text = text.replace(f"{_MODULE[target]}.{source}", replacement)
        return text

    arguments = ", ".join(sorted(str(symbol) for symbol in value.free_symbols))
    lines = [f"def {name}({arguments}):"]
    for symbol, replacement in replacements:
        lines.append(f"    {symbol} = {render(replacement)}")
    lines.append(f"    return {render(body)}")

    after = _count_ops(body) + sum(_count_ops(replacement) for _, replacement in replacements)
    return CodeResult(target, "\n".join(lines), before, after, len(replacements))


@dataclass
class ShapeResult:
    ok: bool
    output_shape: list[str] = field(default_factory=list)
    error: str = ""
    axis: str = ""
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": self.ok, "output_shape": self.output_shape}
        if not self.ok:
            payload.update({"error": self.error, "axis": self.axis, "detail": self.detail})
        return payload


def shape_check(spec: str, shapes: dict[str, list[str]], dims: dict[str, int] | None = None) -> ShapeResult:
    """Check an einsum expression over **named** dimensions.

    Named rather than numeric, because that is where the error lives: ``b`` meaning batch in one
    tensor and beams in another type-checks perfectly as ``(32, 32)`` and is wrong. Numeric shape
    checking passes exactly the cases worth catching.

    The failure names the **axis**, not just the fact of a mismatch — "operand 1 axis `d` is 64,
    bound to 512" is a fix; "shape mismatch" is a search.
    """
    dims = dims or {}

    if "->" in spec:
        inputs_text, output_text = spec.split("->", 1)
        output_axes: list[str] | None = list(output_text.strip())
    else:
        inputs_text, output_axes = spec, None

    operands = [part.strip() for part in inputs_text.split(",")]
    # Insertion order, never sorted: the spec's operands are positional, and pairing `bqd` with
    # whichever tensor happens to sort first silently checks a different contraction than the
    # one that was written — which is exactly the class of error this tool exists to catch.
    names = list(shapes)

    if len(operands) != len(names):
        return ShapeResult(
            False,
            error="operand-count",
            detail=f"the spec has {len(operands)} operands but {len(names)} tensors were given",
        )

    bound: dict[str, tuple[str, str]] = {}  # axis -> (size, where it was bound)

    for position, (operand, tensor) in enumerate(zip(operands, names)):
        axes = list(operand)
        sizes = shapes[tensor]

        if len(axes) != len(sizes):
            return ShapeResult(
                False,
                error="rank",
                axis=operand,
                detail=f"`{tensor}` is declared with {len(sizes)} dimensions but the spec gives it {len(axes)}",
            )

        for axis, size in zip(axes, sizes):
            existing = bound.get(axis)
            if existing is None:
                bound[axis] = (size, tensor)
                continue
            if existing[0] != size:
                return ShapeResult(
                    False,
                    error="axis-mismatch",
                    axis=axis,
                    detail=(
                        f"axis `{axis}` is `{existing[0]}` in `{existing[1]}` "
                        f"but `{size}` in `{tensor}` (operand {position})"
                    ),
                )

    if output_axes is None:
        # einsum's implicit output: axes appearing once, in alphabetical order.
        counts: dict[str, int] = {}
        for operand in operands:
            for axis in operand:
                counts[axis] = counts.get(axis, 0) + 1
        output_axes = sorted(axis for axis, count in counts.items() if count == 1)

    unknown = [axis for axis in output_axes if axis not in bound]
    if unknown:
        return ShapeResult(
            False,
            error="unbound-output",
            axis=unknown[0],
            detail=f"output axis `{unknown[0]}` does not appear in any operand",
        )

    resolved = [bound[axis][0] for axis in output_axes]
    if dims:
        resolved = [str(dims.get(size, size)) for size in resolved]
    return ShapeResult(True, resolved)
