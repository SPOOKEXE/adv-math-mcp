"""Integer arithmetic: factoring, gcd, primality, modular inverses, congruences.

Everything here is decidable, so unlike the symbolic modules the answers are definite. But the
house rule still holds where a claim can carry evidence: ``is_prime`` on a composite returns a
factor, and an impossible modular inverse names the offending gcd, because "no" with a witness
is checkable and "no" alone is a shrug.

Values arrive as expression strings through the session parser, so ``2**61 - 1`` works and
``__import__`` does not.
"""

from __future__ import annotations

from typing import Any, Literal

import sympy

from .equivalence import TimeoutExceeded, deadline
from .session import MathError, Session

NumTheoryOp = Literal["factorint", "gcd", "lcm", "is_prime", "totient", "mod_inverse", "crt"]


def _as_int(session: Session, text: str) -> int:
    value = session.resolve(str(text))
    if not value.is_Integer:
        value = sympy.nsimplify(value)
    if not value.is_Integer:
        raise MathError(f"`{text}` is not an integer; number theory works on integers")
    return int(value)


def numtheory(
    session: Session,
    op: NumTheoryOp,
    values: list[str] | None = None,
    *,
    modulus: str | None = None,
    pairs: list[list[str]] | None = None,
    timeout: float = 10.0,
) -> dict[str, Any]:
    """One verb over the integer operations; ``op`` picks the machinery.

    ``crt`` takes ``pairs`` of ``[residue, modulus]`` and solves the simultaneous congruences;
    moduli need not be coprime, and an unsolvable system names the pair that clashes.
    """
    if op not in ("factorint", "gcd", "lcm", "is_prime", "totient", "mod_inverse", "crt"):
        raise MathError(f"unknown numtheory op `{op}`")

    result: dict[str, Any] = {"op": op}
    try:
        with deadline(timeout):
            if op == "crt":
                if not pairs:
                    raise MathError("`crt` needs pairs of [residue, modulus]")
                congruences = [(_as_int(session, r), _as_int(session, m)) for r, m in pairs]
                solved = sympy.ntheory.modular.solve_congruence(*congruences)
                if solved is None:
                    clash = _first_clash(congruences)
                    result["solvable"] = False
                    result["detail"] = clash
                    return result
                value, modulus_value = solved
                result.update({"solvable": True, "value": str(value), "modulus": str(modulus_value)})
                return result

            numbers = [_as_int(session, item) for item in (values or [])]
            if not numbers:
                raise MathError(f"`{op}` needs values")

            if op == "factorint":
                factors = sympy.factorint(numbers[0])
                result["factors"] = {str(prime): int(exponent) for prime, exponent in sorted(factors.items())}
                result["distinct_primes"] = len(factors)

            elif op == "gcd":
                result["gcd"] = str(sympy.igcd(*numbers)) if len(numbers) > 1 else str(abs(numbers[0]))

            elif op == "lcm":
                result["lcm"] = str(sympy.ilcm(*numbers)) if len(numbers) > 1 else str(abs(numbers[0]))

            elif op == "is_prime":
                candidate = numbers[0]
                prime = sympy.isprime(candidate)
                result["is_prime"] = bool(prime)
                if not prime and candidate > 1:
                    # The witness: a factor the caller can multiply out to check the claim.
                    result["witness_factor"] = str(min(sympy.factorint(candidate)))

            elif op == "totient":
                result["totient"] = str(sympy.totient(numbers[0]))

            else:  # mod_inverse
                if modulus is None:
                    raise MathError("`mod_inverse` needs a modulus")
                m = _as_int(session, modulus)
                a = numbers[0]
                shared = sympy.igcd(a, m)
                if shared != 1:
                    result["invertible"] = False
                    result["detail"] = f"gcd({a}, {m}) = {shared}, so no inverse exists"
                    return result
                result.update({"invertible": True, "inverse": str(sympy.mod_inverse(a, m))})
    except TimeoutExceeded:
        result["warnings"] = [f"{op} exceeded {timeout}s"]
    return result


def _first_clash(congruences: list[tuple[int, int]]) -> str:
    """Name a pair of congruences that cannot hold together."""
    for i, (r1, m1) in enumerate(congruences):
        for r2, m2 in congruences[i + 1 :]:
            shared = sympy.igcd(m1, m2)
            if (r1 - r2) % shared != 0:
                return (
                    f"x = {r1} (mod {m1}) and x = {r2} (mod {m2}) clash: "
                    f"gcd({m1}, {m2}) = {shared} does not divide {r1} - {r2}"
                )
    return "the congruences have no simultaneous solution"
