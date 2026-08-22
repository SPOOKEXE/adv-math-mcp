"""Structural analysis of an equation system.

Three classical algorithms, each answering a question the others cannot.

**Hopcroft–Karp** — maximum matching between variables and relations. A system is structurally
singular when no perfect matching exists, and that is a *structural* fact: it holds regardless
of the numbers, so it is detectable before any solving is attempted.

**Dulmage–Mendelsohn** — which subset is under- or over-determined. "The system is singular" is
a dead end; "these four variables are constrained by three equations" is the fix. This is the
difference between a solver that says no and one that says what to add.

**Tarjan** on the matched digraph — strongly connected components in reverse topological order,
which is the solve order. Each component of size one is a variable solvable on its own; anything
larger is a genuinely simultaneous block, and knowing which is which is the difference between
substitution and a nonlinear solve.

Relations are stored **undirected**. Baking in ``y = f(x)`` throws away exactly the flexibility
that makes the graph worth having — the same relation has to solve for ``x`` given ``y`` and for
``y`` given ``x``, and which one is wanted is a property of the query, not the equation.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class Matching:
    """A maximum matching between variables and relations."""

    variable_to_relation: dict[str, str] = field(default_factory=dict)
    relation_to_variable: dict[str, str] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.variable_to_relation)


def hopcroft_karp(variables: Iterable[str], relations: dict[str, set[str]]) -> Matching:
    """Maximum bipartite matching, variables ↔ relations.

    ``relations`` maps a relation id to the variables it mentions. The augmenting-path search is
    breadth-first by layer and then depth-first, which is what makes this O(E√V) rather than the
    O(VE) of naive augmentation — on a system with a few hundred relations the difference is
    the algorithm being usable inside an interactive call.
    """
    variable_list = sorted(variables)
    adjacency = {name: sorted(relations.get(name, set())) for name in relations}
    by_variable: dict[str, list[str]] = {name: [] for name in variable_list}
    for relation, mentioned in adjacency.items():
        for variable in mentioned:
            if variable in by_variable:
                by_variable[variable].append(relation)

    matching = Matching()
    INFINITY = float("inf")

    def augment(variable: str, distance: dict[str, float]) -> bool:
        for relation in by_variable[variable]:
            partner = matching.relation_to_variable.get(relation)
            if partner is None or (distance[partner] == distance[variable] + 1 and augment(partner, distance)):
                matching.variable_to_relation[variable] = relation
                matching.relation_to_variable[relation] = variable
                return True
        distance[variable] = INFINITY
        return False

    while True:
        # Layer the free variables, then look for augmenting paths of that length.
        distance: dict[str, float] = {}
        queue: deque[str] = deque()
        for variable in variable_list:
            if variable not in matching.variable_to_relation:
                distance[variable] = 0
                queue.append(variable)
            else:
                distance[variable] = INFINITY

        found = INFINITY
        while queue:
            variable = queue.popleft()
            if distance[variable] >= found:
                continue
            for relation in by_variable[variable]:
                partner = matching.relation_to_variable.get(relation)
                if partner is None:
                    found = distance[variable] + 1
                elif distance[partner] == INFINITY:
                    distance[partner] = distance[variable] + 1
                    queue.append(partner)

        if found == INFINITY:
            return matching

        for variable in variable_list:
            if variable not in matching.variable_to_relation:
                augment(variable, distance)


@dataclass
class Decomposition:
    """The Dulmage–Mendelsohn coarse decomposition."""

    underdetermined_variables: list[str] = field(default_factory=list)
    underdetermined_relations: list[str] = field(default_factory=list)
    wellconstrained_variables: list[str] = field(default_factory=list)
    wellconstrained_relations: list[str] = field(default_factory=list)
    overdetermined_variables: list[str] = field(default_factory=list)
    overdetermined_relations: list[str] = field(default_factory=list)

    @property
    def is_wellconstrained(self) -> bool:
        return not self.underdetermined_variables and not self.overdetermined_relations

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "underdetermined_variables": self.underdetermined_variables,
            "underdetermined_relations": self.underdetermined_relations,
            "wellconstrained_variables": self.wellconstrained_variables,
            "wellconstrained_relations": self.wellconstrained_relations,
            "overdetermined_variables": self.overdetermined_variables,
            "overdetermined_relations": self.overdetermined_relations,
        }


def dulmage_mendelsohn(variables: Iterable[str], relations: dict[str, set[str]]) -> Decomposition:
    """Split the system into under-, well- and over-determined parts.

    The under-determined part is everything reachable by alternating paths from an *unmatched
    variable*; the over-determined part is everything reachable from an unmatched *relation*.
    What is left is square and has a perfect matching.

    Reporting the subsets is the whole value: a singular system with one missing equation and a
    singular system with forty are the same verdict and completely different problems.
    """
    variable_list = sorted(variables)
    relation_list = sorted(relations)
    matching = hopcroft_karp(variable_list, relations)

    by_variable: dict[str, set[str]] = {name: set() for name in variable_list}
    for relation, mentioned in relations.items():
        for variable in mentioned:
            if variable in by_variable:
                by_variable[variable].add(relation)

    # From an unmatched variable: variable → any relation → its matched variable.
    under_variables: set[str] = set()
    under_relations: set[str] = set()
    queue = deque(name for name in variable_list if name not in matching.variable_to_relation)
    under_variables.update(queue)

    while queue:
        variable = queue.popleft()
        for relation in sorted(by_variable[variable]):
            if relation in under_relations:
                continue
            under_relations.add(relation)
            partner = matching.relation_to_variable.get(relation)
            if partner is not None and partner not in under_variables:
                under_variables.add(partner)
                queue.append(partner)

    # From an unmatched relation, the mirror image.
    over_relations: set[str] = set()
    over_variables: set[str] = set()
    relation_queue = deque(name for name in relation_list if name not in matching.relation_to_variable)
    over_relations.update(relation_queue)

    while relation_queue:
        relation = relation_queue.popleft()
        for variable in sorted(relations.get(relation, set())):
            if variable not in by_variable or variable in over_variables:
                continue
            over_variables.add(variable)
            partner = matching.variable_to_relation.get(variable)
            if partner is not None and partner not in over_relations:
                over_relations.add(partner)
                relation_queue.append(partner)

    well_variables = [v for v in variable_list if v not in under_variables and v not in over_variables]
    well_relations = [r for r in relation_list if r not in under_relations and r not in over_relations]

    return Decomposition(
        sorted(under_variables),
        sorted(under_relations),
        well_variables,
        well_relations,
        sorted(over_variables),
        sorted(over_relations),
    )


@dataclass
class Block:
    """One strongly connected component: variables that must be solved together."""

    variables: list[str]
    relations: list[str]

    @property
    def simultaneous(self) -> bool:
        return len(self.variables) > 1

    def to_dict(self) -> dict[str, object]:
        return {"variables": self.variables, "relations": self.relations, "simultaneous": self.simultaneous}


def tarjan_blocks(variables: Iterable[str], relations: dict[str, set[str]], matching: Matching | None = None) -> list[Block]:
    """Block lower-triangular decomposition, in solve order.

    Direction is imposed *here*, at analysis time, from the matching: each variable is assigned
    the relation that solves for it, and every other variable in that relation is an input.
    Tarjan's components on that digraph, in the order it produces them, are already reverse
    topological — so the list is a valid solve order with no separate sort.

    Iterative rather than recursive: a chain of two hundred substitutions is an ordinary
    engineering model and a perfectly ordinary way to exhaust the Python stack.
    """
    variable_list = sorted(variables)
    matching = matching or hopcroft_karp(variable_list, relations)

    # variable → the variables it depends on, via the relation that solves for it.
    dependencies: dict[str, list[str]] = {}
    for variable in variable_list:
        relation = matching.variable_to_relation.get(variable)
        if relation is None:
            dependencies[variable] = []
            continue
        dependencies[variable] = sorted(
            other for other in relations.get(relation, set()) if other != variable and other in matching.variable_to_relation
        )

    index_of: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    components: list[list[str]] = []
    counter = 0

    for root in variable_list:
        if root in index_of:
            continue

        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, child = work[-1]

            if child == 0:
                index_of[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on_stack.add(node)

            recursed = False
            neighbours = dependencies[node]
            while child < len(neighbours):
                neighbour = neighbours[child]
                child += 1
                if neighbour not in index_of:
                    work[-1] = (node, child)
                    work.append((neighbour, 0))
                    recursed = True
                    break
                if neighbour in on_stack:
                    low[node] = min(low[node], index_of[neighbour])
            if recursed:
                continue

            work[-1] = (node, child)
            work.pop()

            if low[node] == index_of[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(sorted(component))

            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])

    return [
        Block(
            component,
            sorted({matching.variable_to_relation[v] for v in component if v in matching.variable_to_relation}),
        )
        for component in components
    ]
