"""Equivalence, derivation checking and the timeout that makes them safe to call.

**Never a bare boolean.** Symbolic equivalence is undecidable in general, and a system that
answers ``False`` when it means "I could not show it" is worse than one that says nothing —
the caller acts on the negative. Three verdicts: ``proved``, ``disproved`` with a
counterexample, ``unknown`` with the strategies that were tried.

**Every call has a hard timeout.** ``integrate`` will hang forever given the chance, and so will
``simplify`` on a large enough expression. A partial result with the work that finished beats a
call that never returns.
"""

from __future__ import annotations

import random
import signal
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

import sympy

from .session import MathError, Session, canonical_form

Verdict = Literal["proved", "disproved", "unknown"]


class TimeoutExceeded(MathError):
    """Raised inside a guarded block when the deadline passes."""


@contextmanager
def deadline(seconds: float) -> Iterator[None]:
    """A hard wall-clock limit around a block of symbolic work.

    Implemented with ``SIGALRM``: sympy holds the GIL through long C-level loops, so a watchdog
    thread cannot interrupt it, and a subprocess per call would cost more than most calls take.
    On a platform without ``SIGALRM`` this degrades to no limit rather than pretending — an
    unenforced timeout that reports success is the failure this exists to prevent.
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _fire(_signum: int, _frame: Any) -> None:
        raise TimeoutExceeded(f"exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class EquivalenceResult:
    verdict: Verdict
    strategies: list[str] = field(default_factory=list)
    counterexample: dict[str, str] | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"verdict": self.verdict, "strategies": self.strategies}
        if self.counterexample is not None:
            payload["counterexample"] = self.counterexample
        if self.detail:
            payload["detail"] = self.detail
        return payload


def _sample_domain(symbol: sympy.Symbol, rng: random.Random) -> sympy.Rational | sympy.Integer:
    """Draw a value that respects the symbol's assumptions.

    Sampling outside the declared domain is how a correct identity gets "disproved": test
    ``sqrt(x**2) == x`` at ``x = -1`` after the caller declared ``x`` positive and the
    counterexample is not a counterexample, it is a bug in the checker.
    """
    if symbol.is_integer:
        low, high = (1, 40) if symbol.is_positive else (-40, 40)
        value = rng.randint(low, high)
        if symbol.is_nonzero and value == 0:
            value = 1
        if symbol.is_even:
            value = value * 2 if value != 0 else 2
        elif symbol.is_odd:
            value = value * 2 + 1
        return sympy.Integer(value)

    magnitude = sympy.Rational(rng.randint(1, 4000), 1000)
    if symbol.is_positive or symbol.is_nonnegative:
        return magnitude
    if symbol.is_negative or symbol.is_nonpositive:
        return -magnitude
    return magnitude if rng.random() < 0.5 else -magnitude


def check_equivalence(
    session: Session,
    left: str,
    right: str,
    *,
    samples: int = 24,
    seed: int = 20260810,
    timeout: float = 5.0,
) -> EquivalenceResult:
    """Decide whether two expressions are equal, over the declared domain.

    Strategies in cost order. The cheap structural test settles most real cases; the numeric
    probe is what turns "I could not simplify it to zero" into an actual counterexample.
    """
    a = session.resolve(left)
    b = session.resolve(right)
    tried: list[str] = []

    difference = a - b

    tried.append("canonical-form")
    try:
        with deadline(timeout):
            if canonical_form(difference) == 0:
                return EquivalenceResult("proved", tried)
    except TimeoutExceeded:
        return EquivalenceResult("unknown", tried, detail="canonical form timed out")

    # The numeric probe runs *before* `simplify` on purpose: it is fast, and finding a
    # counterexample makes the expensive symbolic attempt unnecessary.
    tried.append("numeric-sampling")
    rng = random.Random(seed)
    free = sorted(difference.free_symbols, key=str)
    numeric_failures = 0

    for _ in range(samples if free else 1):
        assignment = {symbol: _sample_domain(symbol, rng) for symbol in free}
        try:
            with deadline(timeout):
                value = complex(sympy.N(difference.subs(assignment), 30))
        except (TypeError, ValueError, ZeroDivisionError, TimeoutExceeded):
            # An undefined point is not a counterexample; it is a point off the domain.
            numeric_failures += 1
            continue

        if abs(value) > 1e-9:
            return EquivalenceResult(
                "disproved",
                tried,
                counterexample={str(symbol): str(value) for symbol, value in assignment.items()},
                detail=f"difference evaluates to {value:.6g}",
            )

    tried.append("simplify")
    try:
        with deadline(timeout):
            simplified = sympy.simplify(difference)
    except TimeoutExceeded:
        return EquivalenceResult("unknown", tried, detail="simplify timed out; sampling found no counterexample")

    if simplified == 0:
        return EquivalenceResult("proved", tried)

    # Sampling agreed everywhere but the difference will not reduce: the honest answer.
    return EquivalenceResult(
        "unknown",
        tried,
        detail=(
            f"no counterexample in {samples} samples, and the difference did not reduce to zero "
            f"(residual {sympy.sstr(simplified)[:80]})"
        ),
    )


@dataclass
class DerivationResult:
    valid: bool
    first_invalid_step: int | None
    checked: int
    steps: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "first_invalid_step": self.first_invalid_step,
            "checked": self.checked,
            "steps": self.steps,
        }


def check_derivation(
    session: Session,
    steps: list[str],
    *,
    samples: int = 16,
    seed: int = 20260810,
    timeout: float = 5.0,
) -> DerivationResult:
    """Check a chain of expressions, each claimed equal to the one before it.

    Returns the **index of the first invalid step**, and stops there. Continuing past a broken
    step checks consequences of something already known to be wrong, and reports a cascade of
    failures with one cause — the index is the actionable output.

    An ``unknown`` step does not fail the derivation; it is recorded and the chain continues.
    Treating "could not prove" as "wrong" would reject correct derivations for being hard.
    """
    if len(steps) < 2:
        raise MathError("a derivation needs at least two steps")

    records: list[dict[str, Any]] = []

    for index in range(1, len(steps)):
        result = check_equivalence(
            session, steps[index - 1], steps[index], samples=samples, seed=seed, timeout=timeout
        )
        records.append({"step": index, "verdict": result.verdict, "detail": result.detail})

        if result.verdict == "disproved":
            records[-1]["counterexample"] = result.counterexample
            return DerivationResult(False, index, index, records)

    return DerivationResult(True, None, len(steps) - 1, records)


def batch_equivalence(
    session: Session,
    pairs: list[tuple[str, str]],
    *,
    samples: int = 24,
    seed: int = 20260810,
    timeout: float = 5.0,
) -> list[dict[str, Any]]:
    """Check many pairs in one round trip.

    The round trip is the cost being avoided: twenty single calls is twenty turns, and the
    verdicts are independent, so there is nothing to gain by serialising them through the model.
    A failure in one pair is reported in place rather than failing the batch.
    """
    results: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(pairs):
        try:
            results.append({"index": index, **check_equivalence(
                session, left, right, samples=samples, seed=seed, timeout=timeout
            ).to_dict()})
        except MathError as error:
            results.append({"index": index, "verdict": "unknown", "error": str(error)})
    return results
