"""Dimensional analysis: the units field finally has teeth."""

from __future__ import annotations

import pytest
import sympy

from math_mcp.contract.model import Formula, Scope, Variable, audit, find_orphans
from math_mcp.contract.units import parse_units, unit_witnesses
from math_mcp.cas.session import MathError


def scope_with(*, variables: list[Variable], formulas: list[Formula]) -> Scope:
    scope = Scope("s")
    for variable in variables:
        scope.define_variable(variable)
    for formula in formulas:
        scope.define_formula(formula)
    return scope


MECHANICS = [
    Variable("E", "energy", units="kg*m**2/s**2", status="measured"),
    Variable("m", "mass", units="kg", status="measured"),
    Variable("c", "speed", units="m/s", status="measured"),
    Variable("p", "momentum", units="kg*m/s", status="measured"),
]


class TestParseUnits:
    def test_dimensionless_spellings_collapse_to_one(self) -> None:
        assert parse_units("1") == parse_units("dimensionless") == sympy.Integer(1)
        assert parse_units("") == sympy.Integer(1)

    def test_unit_symbols_are_prefixed_against_variable_collision(self) -> None:
        # A unit named `m` must never collide with a variable named `m` once both live in the
        # same substituted expression.
        parsed = parse_units("m/s")
        assert {str(symbol) for symbol in parsed.free_symbols} == {"u_m", "u_s"}

    def test_garbage_units_are_refused(self) -> None:
        with pytest.raises(MathError, match="does not parse"):
            parse_units("kg**")


class TestUnitWitnesses:
    def test_a_consistent_formula_produces_no_witness(self) -> None:
        scope = scope_with(variables=MECHANICS, formulas=[Formula("einstein", "definition", "E = m*c**2")])
        assert unit_witnesses(scope) == []

    def test_mismatched_terms_are_a_witness(self) -> None:
        scope = scope_with(variables=MECHANICS, formulas=[Formula("broken", "definition", "E = m*c**2 + p")])
        witnesses = unit_witnesses(scope)
        assert len(witnesses) == 1
        assert witnesses[0]["formulas"] == ["broken"]

    def test_matching_terms_do_not_cancel_into_silence(self) -> None:
        # Substituting bare unit monomials lets `E - m*c**2` cancel to zero, hiding the
        # mismatched term standing next to them. That is the regression this test pins.
        scope = scope_with(
            variables=MECHANICS,
            formulas=[Formula("broken", "definition", "E = m*c**2 + p"), Formula("fine", "definition", "E = m*c**2")],
        )
        assert [w["formulas"] for w in unit_witnesses(scope)] == [["broken"]]

    def test_a_dimensionful_transcendental_argument_is_a_witness(self) -> None:
        scope = scope_with(variables=MECHANICS, formulas=[Formula("cat", "definition", "E = exp(p)")])
        assert any("dimensionless" in w["detail"] for w in unit_witnesses(scope))

    def test_formulas_with_unrecorded_units_are_skipped_not_guessed(self) -> None:
        variables = MECHANICS + [Variable("x", "no units recorded", status="free")]
        scope = scope_with(variables=variables, formulas=[Formula("mystery", "definition", "E = m*x")])
        assert unit_witnesses(scope) == []

    def test_a_bare_numeral_is_unit_agnostic(self) -> None:
        # `x = 16` is a value assignment; flagging every constant assignment makes the report
        # unreadable, and a report nobody reads catches nothing.
        scope = scope_with(
            variables=[Variable("bpp", "bytes per param", units="bytes/parameters", status="measured")],
            formulas=[Formula("opt-bytes", "definition", "bpp = 16")],
        )
        assert unit_witnesses(scope) == []

    def test_unparseable_units_on_a_variable_are_reported(self) -> None:
        scope = scope_with(
            variables=[Variable("x", "broken units", units="kg**", status="measured")],
            formulas=[],
        )
        witnesses = unit_witnesses(scope)
        assert witnesses and "does not parse" in witnesses[0]["detail"]


class TestAuditIntegration:
    def test_units_findings_arrive_as_audit_witnesses(self) -> None:
        scope = scope_with(variables=MECHANICS, formulas=[Formula("broken", "definition", "E = m*c**2 + p")])
        report = audit(scope)
        assert any(witness.tier == "units" for witness in report.witnesses)

    def test_an_unbounded_approximation_is_an_orphan_class(self) -> None:
        scope = scope_with(
            variables=[],
            formulas=[
                Formula("bounded", "approximation", "a = b", error_term="O(b)"),
                Formula("unbounded", "approximation", "c = d"),
            ],
        )
        assert find_orphans(scope).unbounded == ["unbounded"]

    def test_a_malformed_big_o_error_term_is_a_witness(self) -> None:
        scope = scope_with(
            variables=[],
            formulas=[Formula("bad", "approximation", "a = b", error_term="O(b +)")],
        )
        report = audit(scope)
        assert any(witness.tier == "error-term" for witness in report.witnesses)

    def test_a_prose_error_term_is_legitimate(self) -> None:
        # "MFU varies 10-20% with parallelism" is a real error statement for an empirical fit;
        # a parser does not get a vote on prose.
        scope = scope_with(
            variables=[],
            formulas=[Formula("fit", "empirical-fit", "a = b", error_term="varies 10-20% with parallelism")],
        )
        report = audit(scope)
        assert not any(witness.tier == "error-term" for witness in report.witnesses)


class TestParseUnitsSecurity:
    def test_units_are_behind_the_same_forbidden_wall_as_expressions(self) -> None:
        # The units field is model-written text like everything else; `__` and friends are
        # refused before the parser ever sees them.
        with pytest.raises(MathError, match="not a unit expression"):
            parse_units("__import__('os')")
