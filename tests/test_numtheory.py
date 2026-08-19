"""Integer operations: definite answers, and witnesses where a claim can carry one."""

from __future__ import annotations

import pytest

from math_mcp.cas.numtheory import numtheory
from math_mcp.cas.session import MathError, Session


@pytest.fixture()
def session() -> Session:
    return Session()


class TestPrimality:
    def test_a_prime_is_a_prime(self, session: Session) -> None:
        assert numtheory(session, "is_prime", ["2**61 - 1"])["is_prime"] is True

    def test_a_composite_carries_a_factor_witness(self, session: Session) -> None:
        # "no" with a factor is checkable by one multiplication; "no" alone is a shrug.
        result = numtheory(session, "is_prime", ["91"])
        assert result["is_prime"] is False
        assert result["witness_factor"] == "7"


class TestFactoringAndGcd:
    def test_factorint_returns_the_factorisation(self, session: Session) -> None:
        assert numtheory(session, "factorint", ["360"])["factors"] == {"2": 3, "3": 2, "5": 1}

    def test_gcd_and_lcm_take_a_list(self, session: Session) -> None:
        assert numtheory(session, "gcd", ["12", "18", "24"])["gcd"] == "6"
        assert numtheory(session, "lcm", ["4", "6"])["lcm"] == "12"

    def test_values_may_be_expressions_but_must_be_integers(self, session: Session) -> None:
        assert numtheory(session, "totient", ["2**4"])["totient"] == "8"
        with pytest.raises(MathError, match="not an integer"):
            numtheory(session, "gcd", ["1/2", "3"])


class TestModular:
    def test_an_inverse_exists_when_coprime(self, session: Session) -> None:
        result = numtheory(session, "mod_inverse", ["3"], modulus="7")
        assert result == {"op": "mod_inverse", "invertible": True, "inverse": "5"}

    def test_no_inverse_names_the_gcd(self, session: Session) -> None:
        result = numtheory(session, "mod_inverse", ["4"], modulus="6")
        assert result["invertible"] is False
        assert "gcd(4, 6) = 2" in result["detail"]

    def test_crt_solves_compatible_congruences(self, session: Session) -> None:
        result = numtheory(session, "crt", pairs=[["2", "3"], ["3", "5"]])
        assert result["solvable"] is True
        assert (result["value"], result["modulus"]) == ("8", "15")

    def test_crt_names_the_clashing_pair(self, session: Session) -> None:
        result = numtheory(session, "crt", pairs=[["0", "4"], ["1", "6"]])
        assert result["solvable"] is False
        assert "gcd(4, 6)" in result["detail"]


class TestRefusals:
    def test_an_unknown_op_is_refused_by_name(self, session: Session) -> None:
        with pytest.raises(MathError, match="divine"):
            numtheory(session, "divine", ["7"])

    def test_crt_without_pairs_is_refused(self, session: Session) -> None:
        with pytest.raises(MathError, match="pairs"):
            numtheory(session, "crt")

    def test_ops_without_values_are_refused(self, session: Session) -> None:
        with pytest.raises(MathError, match="needs values"):
            numtheory(session, "factorint")

    def test_mod_inverse_without_a_modulus_is_refused(self, session: Session) -> None:
        with pytest.raises(MathError, match="modulus"):
            numtheory(session, "mod_inverse", ["3"])
