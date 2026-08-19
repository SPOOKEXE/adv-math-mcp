"""The object model, and the six verbs that fall out of it.

The layer that catches what the CAS structurally cannot see: ``B`` meaning sequences in one
equation and tokens in another, a factor of 2 that only appears where two definitions meet, a
result still resting on an iid assumption dropped three iterations ago. None of those is a
symbolic error — each formula is individually correct.

Design commitments, each of which is a trap avoided rather than a feature added:

* **Six verbs, parameterised noun.** Tool count is a context cost. ``define`` · ``list`` ·
  ``audit`` · ``resolve`` · ``fork`` · ``impact``, not ``define-variable`` +
  ``define-formula`` + ``define-assumption`` + …
* **The registry rots if maintenance is tedious.** ``define`` on a formula auto-registers
  unknown symbols as *provisional* and infers what it can from usage. Demanding declaration
  first means the second formula never gets entered.
* **Relations are stored undirected.** Direction is computed at resolve time.
* **``audit`` returns witnesses, never verdicts** — not "eq 3 and eq 7 conflict" but "at n=10,
  d=64, eq 3 gives 0.5 and eq 7 gives 0.7". And never "consistent": only "no contradiction found
  by methods X, Y at depth N".
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Literal

import sympy

from ..cas.session import MathError, Session
from .graph import Block, dulmage_mendelsohn, hopcroft_karp, tarjan_blocks

VariableStatus = Literal["free", "measured", "derived", "hyperparameter", "provisional"]
FormulaKind = Literal["definition", "constraint", "assumption", "derived", "approximation", "empirical-fit"]
VerificationStatus = Literal["unverified", "verified", "stale", "refuted"]


@dataclass(frozen=True)
class Variable:
    name: str
    semantics: str = ""
    domain: str = "real"
    #: Shape as named dims, or a unit string. One field: a quantity has one or the other.
    shape: tuple[str, ...] = ()
    units: str = ""
    status: VariableStatus = "provisional"
    #: θ in the paper, `W` in the notes, `weight` in the code — all one variable.
    aliases: tuple[str, ...] = ()
    provenance: str = ""
    constraints: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "semantics": self.semantics,
            "domain": self.domain,
            "shape": list(self.shape),
            "units": self.units,
            "status": self.status,
            "aliases": list(self.aliases),
            "provenance": self.provenance,
            "constraints": list(self.constraints),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Variable:
        """The inverse of ``to_dict``, tolerant of a field the writer did not have.

        Tolerant on purpose: an environment saved by an older build is still worth loading, and
        the alternative — refusing it — turns every added field into a migration.
        """
        return Variable(
            name=str(data["name"]),
            semantics=str(data.get("semantics", "")),
            domain=str(data.get("domain", "real")),
            shape=tuple(str(part) for part in data.get("shape", ())),
            units=str(data.get("units", "")),
            status=data.get("status", "provisional"),
            aliases=tuple(str(alias) for alias in data.get("aliases", ())),
            provenance=str(data.get("provenance", "")),
            constraints=tuple(str(item) for item in data.get("constraints", ())),
        )


@dataclass(frozen=True)
class Formula:
    id: str
    kind: FormulaKind
    #: Stored as `lhs - rhs`, undirected. Direction is a property of the query.
    expression: str
    validity: tuple[str, ...] = ()
    provenance: str = ""
    status: VerificationStatus = "unverified"
    #: Assumption ids this rests on directly.
    assumes: tuple[str, ...] = ()
    #: For approximations: the error term, so a chain of them can be judged.
    error_term: str = ""

    @property
    def is_approximation(self) -> bool:
        return self.kind in ("approximation", "empirical-fit")

    @property
    def defines(self) -> str | None:
        """The quantity the author was defining, from the written orientation.

        Relations are *stored* undirected and solved in whichever direction is asked for — that
        does not change. But the side the author wrote on the left is real information about
        intent, and ``audit`` needs it: two relations that merely *mention* ``d_model`` are not
        in conflict when they disagree about it under a random assignment, because neither
        claims to determine it. Two relations that both **define** ``init_var`` and disagree
        are a genuine contradiction. Without this distinction the probe reports a witness for
        almost every pair, and a report that is always noisy is one nobody reads.
        """
        if "=" not in self.expression or "==" in self.expression:
            return None
        left = self.expression.split("=", 1)[0].strip()
        return left if left.replace("_", "").isalnum() and not left[0].isdigit() else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "expression": self.expression,
            "validity": list(self.validity),
            "provenance": self.provenance,
            "status": self.status,
            "assumes": list(self.assumes),
            "error_term": self.error_term,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Formula:
        return Formula(
            id=str(data["id"]),
            kind=data.get("kind", "definition"),
            expression=str(data["expression"]),
            validity=tuple(str(item) for item in data.get("validity", ())),
            provenance=str(data.get("provenance", "")),
            status=data.get("status", "unverified"),
            assumes=tuple(str(item) for item in data.get("assumes", ())),
            error_term=str(data.get("error_term", "")),
        )


@dataclass(frozen=True)
class Assumption:
    """A first-class node, not a comment.

    The reason it is a node: ``assumption_closure`` has to answer "what dies if I relax iid",
    and a comment cannot be traversed.
    """

    id: str
    statement: str
    provenance: str = ""
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "statement": self.statement, "provenance": self.provenance, "active": self.active}

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Assumption:
        return Assumption(
            id=str(data["id"]),
            statement=str(data.get("statement", "")),
            provenance=str(data.get("provenance", "")),
            active=bool(data.get("active", True)),
        )


class ContractError(MathError):
    pass


class Scope:
    """A forkable context. Mathematical iteration is branchy: paper's formulation, yours, v2."""

    def __init__(self, name: str, session: Session | None = None, parent: str | None = None) -> None:
        self.name = name
        self.parent = parent
        self.session = session or Session()
        self.variables: dict[str, Variable] = {}
        self.formulas: dict[str, Formula] = {}
        self.assumptions: dict[str, Assumption] = {}
        #: name or alias → canonical variable name
        self._alias_index: dict[str, str] = {}

    # -- define ---------------------------------------------------------------

    def canonical(self, name: str) -> str:
        return self._alias_index.get(name, name)

    def define_variable(self, variable: Variable) -> Variable:
        canonical = self.canonical(variable.name)
        existing = self.variables.get(canonical)

        if existing is not None and existing.status == "provisional" and variable.status != "provisional":
            # Promotion: the provisional record was a placeholder, and everything the caller now
            # states wins over what was inferred.
            variable = replace(variable, name=canonical, aliases=tuple(sorted(set(existing.aliases) | set(variable.aliases))))

        self.variables[variable.name] = variable
        self._alias_index[variable.name] = variable.name
        for alias in variable.aliases:
            self._alias_index[alias] = variable.name

        flags: dict[str, bool] = {}
        if variable.domain in ("real", "integer", "rational", "complex"):
            flags[variable.domain] = True
        for constraint in variable.constraints:
            text = constraint.replace(" ", "")
            if text in (f"{variable.name}>0", ">0"):
                flags["positive"] = True
            elif text in (f"{variable.name}>=0", ">=0"):
                flags["nonnegative"] = True
        if flags:
            self.session.declare(variable.name, **flags)
        return variable

    def define_formula(self, formula: Formula) -> tuple[Formula, list[str]]:
        """Register a formula, auto-registering the symbols it mentions.

        The provisional registration is what keeps the registry alive. A system that errors on
        an undeclared symbol gets one formula entered and then abandoned, and an empty registry
        catches nothing at all.
        """
        expression = self._parse(formula.expression)
        provisional: list[str] = []

        for symbol in sorted(expression.free_symbols, key=str):
            name = self.canonical(str(symbol))
            if name in self.variables:
                continue
            self.variables[name] = Variable(
                name=name,
                status="provisional",
                domain="real",
                provenance=f"inferred from {formula.id}",
            )
            self._alias_index[name] = name
            provisional.append(name)

        self.formulas[formula.id] = formula
        return formula, provisional

    def define_assumption(self, assumption: Assumption) -> Assumption:
        self.assumptions[assumption.id] = assumption
        return assumption

    def _parse(self, expression: str) -> sympy.Expr:
        text = expression
        if "=" in text and "==" not in text and ">=" not in text and "<=" not in text:
            left, right = text.split("=", 1)
            text = f"({left}) - ({right})"
        return self.session.parse(text)[1]

    # -- graph ----------------------------------------------------------------

    def relation_map(self, *, kinds: Iterable[FormulaKind] | None = None) -> dict[str, set[str]]:
        """Relation id → the canonical variables it mentions. Undirected, by construction."""
        allowed = set(kinds) if kinds is not None else None
        mapping: dict[str, set[str]] = {}
        for formula in self.formulas.values():
            if allowed is not None and formula.kind not in allowed:
                continue
            mapping[formula.id] = {
                self.canonical(str(symbol)) for symbol in self._parse(formula.expression).free_symbols
            }
        return mapping

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Everything needed to rebuild this scope, and nothing that can be recomputed.

        The alias index is rebuilt from the variables rather than stored: it is a projection of
        them, and a stored projection is a second copy that can disagree with the first.

        Symbol declarations live on the ``Session``, not here, and are saved beside this by the
        caller — a scope whose variables are `positive` in one session and unconstrained in the
        next is not the same scope, however identical its records look.
        """
        return {
            "name": self.name,
            "parent": self.parent,
            "variables": [variable.to_dict() for variable in self.variables.values()],
            "formulas": [formula.to_dict() for formula in self.formulas.values()],
            "assumptions": [assumption.to_dict() for assumption in self.assumptions.values()],
        }

    @staticmethod
    def from_dict(data: dict[str, Any], session: Session | None = None) -> Scope:
        """Rebuild a scope, re-deriving what ``to_dict`` left out.

        Formulas go back through ``define_formula`` so their expressions are re-parsed against
        the session that will actually evaluate them. Storing a parsed form would be storing a
        SymPy version, and a saved environment that will not load after an upgrade is worse than
        one that takes a moment longer.
        """
        scope = Scope(str(data.get("name", "default")), session, data.get("parent"))
        for record in data.get("variables", []):
            scope.define_variable(Variable.from_dict(record))
        for record in data.get("assumptions", []):
            scope.define_assumption(Assumption.from_dict(record))
        for record in data.get("formulas", []):
            scope.define_formula(Formula.from_dict(record))
        return scope

    def fork(self, name: str) -> Scope:
        """A deep copy under a new name.

        Deep because the whole point is isolation: a fork that shares its parent's dictionaries
        is a fork that corrupts the baseline the moment anyone edits it, which is the exact
        failure ``fork`` exists to prevent.
        """
        child = Scope(name, parent=self.name)
        child.variables = dict(self.variables)
        child.formulas = dict(self.formulas)
        child.assumptions = dict(self.assumptions)
        child._alias_index = dict(self._alias_index)
        for variable in child.variables.values():
            child.define_variable(variable)
        return child


@dataclass
class OrphanReport:
    """Seven classes, because "orphan" is seven different problems with seven different fixes."""

    undefined: list[str] = field(default_factory=list)
    unused: list[str] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)
    shadowed: list[dict[str, Any]] = field(default_factory=list)
    dangling: list[str] = field(default_factory=list)
    #: Approximations carrying no error term. Not wrong yet, just unbounded, which is how a
    #: chain of them becomes wrong without any single step being at fault.
    unbounded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "undefined": self.undefined,
            "unused": self.unused,
            "unreachable": self.unreachable,
            "unverified": self.unverified,
            "shadowed": self.shadowed,
            "dangling": self.dangling,
            "unbounded": self.unbounded,
        }


def find_orphans(scope: Scope, *, roots: Iterable[str] = ()) -> OrphanReport:
    """Split "orphan" into the six things it actually means."""
    relations = scope.relation_map()
    mentioned: set[str] = set()
    for variables in relations.values():
        mentioned |= variables

    report = OrphanReport()

    # Used but only provisionally registered — nobody ever said what it means.
    report.undefined = sorted(
        name for name in mentioned if scope.variables.get(name, Variable(name)).status == "provisional"
    )
    # Declared and never used. Real, and different: a definition nobody references.
    report.unused = sorted(name for name in scope.variables if name not in mentioned)
    report.unverified = sorted(
        formula.id for formula in scope.formulas.values() if formula.status in ("unverified", "stale")
    )
    report.unbounded = sorted(
        formula.id for formula in scope.formulas.values() if formula.is_approximation and not formula.error_term
    )
    # An assumption a formula claims that does not exist.
    report.dangling = sorted(
        {
            assumption
            for formula in scope.formulas.values()
            for assumption in formula.assumes
            if assumption not in scope.assumptions
        }
    )

    # One name, two meanings: the same string is an alias of one variable and the name of
    # another. This is the tokens-vs-sequences bug, and it is invisible to every symbolic check.
    shadowed: list[dict[str, Any]] = []
    for variable in scope.variables.values():
        for alias in variable.aliases:
            other = scope.variables.get(alias)
            if other is not None and other.name != variable.name:
                shadowed.append(
                    {
                        "name": alias,
                        "meanings": sorted([variable.semantics or variable.name, other.semantics or other.name]),
                        "variables": sorted([variable.name, other.name]),
                    }
                )
    report.shadowed = sorted(shadowed, key=lambda entry: entry["name"])

    # Unreachable: cannot be connected to any root by traversing relations.
    root_set = {scope.canonical(name) for name in roots}
    if root_set:
        reachable = set(root_set)
        changed = True
        while changed:
            changed = False
            for variables in relations.values():
                if variables & reachable and not variables <= reachable:
                    reachable |= variables
                    changed = True
        report.unreachable = sorted(name for name in mentioned if name not in reachable)

    return report


@dataclass
class ResolvePath:
    """The path, not just the value. A number with no derivation cannot be checked."""

    target: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    value: str | None = None
    approximation_chain: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    solvable: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "solvable": self.solvable,
            "reason": self.reason,
            "value": self.value,
            "steps": self.steps,
            "blocks": [block.to_dict() for block in self.blocks],
            "approximation_chain": self.approximation_chain,
            "warnings": self.warnings,
        }


APPROXIMATION_CHAIN_LIMIT = 3


def resolve(scope: Scope, target: str, given: dict[str, Any] | None = None) -> ResolvePath:
    """Solve for a target, and show the work.

    Structural analysis first: whether the system *can* determine the target is decidable before
    any algebra, and answering it first turns "sympy returned nothing" into "``n_heads`` is not
    constrained by anything you have given".
    """
    given = {scope.canonical(key): value for key, value in (given or {}).items()}
    target = scope.canonical(target)

    relations = scope.relation_map(kinds=("definition", "derived", "constraint", "approximation", "empirical-fit"))
    unknowns = {variable for variables in relations.values() for variable in variables} - set(given)

    if target not in unknowns and target not in given:
        return ResolvePath(target, solvable=False, reason=f"`{target}` does not appear in any relation")
    if target in given:
        return ResolvePath(target, value=str(given[target]), reason="given directly")

    # Relations that still have an unknown in them are the ones that can do work.
    active = {rid: (variables & unknowns) for rid, variables in relations.items()}
    active = {rid: variables for rid, variables in active.items() if variables}

    decomposition = dulmage_mendelsohn(unknowns, active)
    if target in decomposition.underdetermined_variables:
        return ResolvePath(
            target,
            solvable=False,
            reason=(
                f"`{target}` is under-determined: "
                f"{len(decomposition.underdetermined_variables)} variables "
                f"({', '.join(decomposition.underdetermined_variables)}) are constrained by "
                f"{len(decomposition.underdetermined_relations)} relations"
            ),
        )

    matching = hopcroft_karp(unknowns, active)
    blocks = tarjan_blocks(unknowns, active, matching)

    # Only the blocks the target actually depends on.
    #
    # The closure is walked target-first — `reversed(blocks)`, since Tarjan hands them back
    # dependencies-first — because a forward pass tests each block against a `needed` set that
    # does not yet contain anything downstream of it, and skips every dependency. The result is
    # a path with exactly one step whose "value" is still an expression in three unknowns.
    needed: set[str] = {target}
    for block in reversed(blocks):
        if set(block.variables) & needed:
            for relation in block.relations:
                needed |= active.get(relation, set())

    ordered: list[Block] = [block for block in blocks if set(block.variables) & needed]

    path = ResolvePath(target, blocks=ordered)
    # Keyed by symbol, not by name. `expr.subs({"d_model": 512})` silently does nothing, and
    # the failure surfaces as a "solved" path whose value is still an expression.
    values: dict[Any, Any] = {scope.session.symbol(name): value for name, value in given.items()}

    for block in ordered:
        formulas = [scope.formulas[rid] for rid in block.relations if rid in scope.formulas]
        for formula in formulas:
            if formula.is_approximation:
                path.approximation_chain.append(formula.id)

        step: dict[str, Any] = {
            "variables": block.variables,
            "relations": block.relations,
            "simultaneous": block.simultaneous,
        }

        try:
            equations = [scope._parse(scope.formulas[rid].expression) for rid in block.relations if rid in scope.formulas]
            substituted = [equation.subs(values) for equation in equations]
            symbols = [scope.session.symbol(name) for name in block.variables]
            solution = sympy.solve(substituted, symbols, dict=True)
            if solution:
                for symbol, value in solution[0].items():
                    values[symbol] = value
                step["solved"] = {str(symbol): str(value) for symbol, value in solution[0].items()}
            else:
                step["solved"] = None
                path.warnings.append(f"block {block.variables} did not solve symbolically")
        except (MathError, NotImplementedError, TypeError, ValueError) as error:
            step["solved"] = None
            path.warnings.append(f"block {block.variables}: {error}")

        path.steps.append(step)

    target_symbol = scope.session.symbol(target)
    if target_symbol in values:
        path.value = str(sympy.simplify(values[target_symbol]))

    if len(path.approximation_chain) >= APPROXIMATION_CHAIN_LIMIT:
        # Compounding approximation error silently is a classic way to be confidently wrong.
        path.warnings.append(
            f"this path chains {len(path.approximation_chain)} approximations "
            f"({', '.join(path.approximation_chain)}); the error terms compound"
        )

    return path


@dataclass
class Witness:
    """Evidence, not a verdict. A conflict nobody can reproduce is a conflict nobody will fix."""

    tier: str
    formulas: list[str]
    assignment: dict[str, str]
    values: dict[str, str]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "formulas": self.formulas,
            "assignment": self.assignment,
            "values": self.values,
            "detail": self.detail,
        }


@dataclass
class AuditReport:
    witnesses: list[Witness] = field(default_factory=list)
    orphans: OrphanReport = field(default_factory=OrphanReport)
    methods: list[str] = field(default_factory=list)
    depth: int = 0

    @property
    def summary(self) -> str:
        """Never "consistent"."""
        if self.witnesses:
            return f"{len(self.witnesses)} contradiction(s) found"
        return f"no contradiction found by {', '.join(self.methods)} at depth {self.depth}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "witnesses": [witness.to_dict() for witness in self.witnesses],
            "orphans": self.orphans.to_dict(),
            "methods": self.methods,
            "depth": self.depth,
        }


def audit(scope: Scope, *, samples: int = 12, seed: int = 20260810, roots: Iterable[str] = ()) -> AuditReport:
    """Tiered consistency checking, cheapest first.

    Shape and unit disagreement first — it costs nothing and catches an embarrassing fraction.
    Then the orphan classes. Then numeric probing, which is where a real contradiction between
    two individually-correct formulas actually shows up.
    """
    report = AuditReport(methods=["shape/units", "orphan-classes", "numeric-probe"], depth=samples)
    report.orphans = find_orphans(scope, roots=roots)

    # Tier 1: two formulas defining the same quantity with different units or shapes.
    for name, variable in sorted(scope.variables.items()):
        for other_name, other in sorted(scope.variables.items()):
            if name >= other_name:
                continue
            if name in other.aliases or other_name in variable.aliases:
                if variable.units and other.units and variable.units != other.units:
                    report.witnesses.append(
                        Witness(
                            "units",
                            [],
                            {},
                            {name: variable.units, other_name: other.units},
                            f"`{name}` and `{other_name}` are aliases but carry different units",
                        )
                    )

    # Tier 1b: dimensional analysis over the units the variables already carry, plus a parse
    # check on every claimed error term: an error term that does not parse bounds nothing.
    from .units import unit_witnesses

    for finding in unit_witnesses(scope):
        report.witnesses.append(Witness("units", finding["formulas"], {}, finding["values"], finding["detail"]))

    for formula_id, formula in sorted(scope.formulas.items()):
        if not (formula.is_approximation and formula.error_term):
            continue
        # Only O(...) is held to being parseable: that spelling claims a mathematical bound.
        # Anything else is prose ("MFU varies 10-20% with parallelism"), which is a legitimate
        # error statement for an empirical fit and not something a parser gets a vote on.
        text = formula.error_term.strip()
        if text.startswith("O(") and text.endswith(")"):
            try:
                scope._parse(text[2:-1])
            except MathError as error:
                report.witnesses.append(
                    Witness(
                        "error-term",
                        [formula_id],
                        {},
                        {formula_id: formula.error_term},
                        f"`{formula_id}` claims an O(...) error term that does not parse: {error}",
                    )
                )

    # Tier 3: probe. Solve each definition for its matched variable and compare where two
    # relations claim the same quantity.
    relations = scope.relation_map(kinds=("definition", "derived", "constraint"))
    unknowns = sorted({variable for variables in relations.values() for variable in variables})
    rng = random.Random(seed)

    # Grouped by what each relation *defines*, not by what it mentions.
    by_variable: dict[str, list[str]] = {}
    for rid in relations:
        defined = scope.formulas[rid].defines
        if defined is None:
            continue
        by_variable.setdefault(scope.canonical(defined), []).append(rid)

    for variable, rids in sorted(by_variable.items()):
        if len(rids) < 2:
            continue

        symbol = scope.session.symbol(variable)
        for _ in range(samples):
            assignment = {
                scope.session.symbol(name): sympy.Integer(rng.randint(1, 64))
                for name in unknowns
                if name != variable
            }
            solved: dict[str, Any] = {}

            for rid in sorted(rids):
                try:
                    equation = scope._parse(scope.formulas[rid].expression).subs(assignment)
                    roots_found = sympy.solve(equation, symbol)
                except (MathError, NotImplementedError, TypeError, ValueError):
                    continue
                if len(roots_found) == 1:
                    solved[rid] = sympy.nsimplify(roots_found[0])

            distinct = {str(value) for value in solved.values()}
            if len(distinct) > 1:
                report.witnesses.append(
                    Witness(
                        "numeric",
                        sorted(solved),
                        {str(k): str(v) for k, v in assignment.items()},
                        {rid: str(value) for rid, value in sorted(solved.items())},
                        f"`{variable}` takes {len(distinct)} different values under the same assignment",
                    )
                )
                break

    return report


@dataclass
class ImpactReport:
    """``make`` for mathematics: change eq 3, everything downstream goes stale."""

    symbol: str
    downstream_formulas: list[str] = field(default_factory=list)
    downstream_variables: list[str] = field(default_factory=list)
    newly_stale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "downstream_formulas": self.downstream_formulas,
            "downstream_variables": self.downstream_variables,
            "newly_stale": self.newly_stale,
        }


def impact(scope: Scope, symbol: str, *, mark_stale: bool = False) -> ImpactReport:
    """Downstream closure, transitively.

    Transitive because that is the whole failure mode: a one-hop report says the change touched
    two formulas, and the third one — which depends on those two — is the one that silently kept
    a verified stamp it no longer deserves.
    """
    name = scope.canonical(symbol)
    relations = scope.relation_map()

    reached_variables = {name}
    reached_formulas: set[str] = set()
    changed = True

    while changed:
        changed = False
        for rid, variables in relations.items():
            if rid in reached_formulas or not (variables & reached_variables):
                continue
            reached_formulas.add(rid)
            new = variables - reached_variables
            if new:
                reached_variables |= new
            changed = True

    newly_stale: list[str] = []
    if mark_stale:
        for rid in sorted(reached_formulas):
            formula = scope.formulas[rid]
            if formula.status == "verified":
                scope.formulas[rid] = replace(formula, status="stale")
                newly_stale.append(rid)

    return ImpactReport(
        name,
        sorted(reached_formulas),
        sorted(reached_variables - {name}),
        newly_stale,
    )


def assumption_closure(scope: Scope, formula_id: str) -> dict[str, Any]:
    """What a result rests on, and what dies if an assumption is relaxed."""
    if formula_id not in scope.formulas:
        raise ContractError(f"unknown formula `{formula_id}`")

    relations = scope.relation_map()
    seen: set[str] = set()
    frontier = [formula_id]
    assumptions: set[str] = set()

    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        formula = scope.formulas[current]
        assumptions |= set(formula.assumes)

        variables = relations.get(current, set())
        for rid, others in relations.items():
            if rid not in seen and others & variables:
                frontier.append(rid)

    return {
        "formula": formula_id,
        "rests_on": sorted(assumptions),
        "via_formulas": sorted(seen),
        "inactive": sorted(a for a in assumptions if a in scope.assumptions and not scope.assumptions[a].active),
    }


def relax(scope: Scope, assumption_id: str) -> dict[str, Any]:
    """"If I relax iid, what dies?" — the inverse query, which is the useful one."""
    if assumption_id not in scope.assumptions:
        raise ContractError(f"unknown assumption `{assumption_id}`")

    affected = sorted(f.id for f in scope.formulas.values() if assumption_id in f.assumes)
    downstream: set[str] = set(affected)
    for rid in affected:
        downstream |= set(assumption_closure(scope, rid)["via_formulas"])

    return {
        "assumption": assumption_id,
        "directly_invalidated": affected,
        "transitively_affected": sorted(downstream),
    }


def diff_contexts(left: Scope, right: Scope) -> dict[str, Any]:
    """Paper's formulation vs yours vs v2."""

    def changed(kind: str, a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for key in sorted(set(a) & set(b)):
            if a[key] != b[key]:
                entries.append({"kind": kind, "id": key, "left": a[key].to_dict(), "right": b[key].to_dict()})
        return entries

    return {
        "added": sorted(
            [f"variable:{k}" for k in set(right.variables) - set(left.variables)]
            + [f"formula:{k}" for k in set(right.formulas) - set(left.formulas)]
            + [f"assumption:{k}" for k in set(right.assumptions) - set(left.assumptions)]
        ),
        "removed": sorted(
            [f"variable:{k}" for k in set(left.variables) - set(right.variables)]
            + [f"formula:{k}" for k in set(left.formulas) - set(right.formulas)]
            + [f"assumption:{k}" for k in set(left.assumptions) - set(right.assumptions)]
        ),
        "changed": changed("variable", left.variables, right.variables)
        + changed("formula", left.formulas, right.formulas)
        + changed("assumption", left.assumptions, right.assumptions),
    }
