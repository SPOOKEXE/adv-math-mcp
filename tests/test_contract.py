"""Contract layer: the cross-formula errors the CAS structurally cannot see."""

from __future__ import annotations

import pytest

from math_mcp.contract.graph import dulmage_mendelsohn, hopcroft_karp, tarjan_blocks
from math_mcp.contract.model import (
    Assumption,
    ContractError,
    Formula,
    Scope,
    Variable,
    assumption_closure,
    audit,
    diff_contexts,
    find_orphans,
    impact,
    relax,
    resolve,
)
from math_mcp.contract.transformer import build_scope, seed_mup_conflict


class TestMatching:
    def test_a_square_system_matches_perfectly(self) -> None:
        relations = {"r1": {"x", "y"}, "r2": {"y", "z"}, "r3": {"z"}}
        matching = hopcroft_karp({"x", "y", "z"}, relations)
        assert matching.size == 3

    def test_a_structurally_singular_system_does_not(self) -> None:
        # Two relations, three unknowns: no numbers required to know this cannot be solved.
        matching = hopcroft_karp({"x", "y", "z"}, {"r1": {"x", "y"}, "r2": {"y", "z"}})
        assert matching.size == 2

    def test_a_variable_nothing_mentions_cannot_be_matched(self) -> None:
        matching = hopcroft_karp({"x", "orphan"}, {"r1": {"x"}})
        assert "orphan" not in matching.variable_to_relation

    def test_augmentation_finds_the_maximum_not_the_greedy_answer(self) -> None:
        # Greedy matching on this order stops at 2; the maximum is 3.
        relations = {"r1": {"a"}, "r2": {"a", "b"}, "r3": {"b", "c"}}
        assert hopcroft_karp({"a", "b", "c"}, relations).size == 3


class TestDulmageMendelsohn:
    def test_it_names_the_underdetermined_subset(self) -> None:
        # "Singular" is a dead end; "these three are constrained by two equations" is the fix.
        decomposition = dulmage_mendelsohn({"x", "y", "z"}, {"r1": {"x", "y"}, "r2": {"y", "z"}})
        assert not decomposition.is_wellconstrained
        assert set(decomposition.underdetermined_variables) == {"x", "y", "z"}

    def test_it_names_the_overdetermined_subset(self) -> None:
        decomposition = dulmage_mendelsohn({"x"}, {"r1": {"x"}, "r2": {"x"}})
        assert set(decomposition.overdetermined_relations) == {"r1", "r2"}

    def test_a_square_system_is_wellconstrained(self) -> None:
        decomposition = dulmage_mendelsohn({"x", "y"}, {"r1": {"x", "y"}, "r2": {"y"}})
        assert decomposition.is_wellconstrained
        assert set(decomposition.wellconstrained_variables) == {"x", "y"}

    def test_the_wellconstrained_part_is_isolated_from_the_rest(self) -> None:
        # One free-standing square block plus one under-determined pair.
        decomposition = dulmage_mendelsohn(
            {"a", "b", "p", "q"},
            {"r1": {"a"}, "r2": {"p", "q"}},
        )
        assert decomposition.wellconstrained_variables == ["a"]
        # `b` is under-determined too: declared and constrained by nothing at all.
        assert set(decomposition.underdetermined_variables) == {"b", "p", "q"}


class TestTarjan:
    def test_it_produces_a_valid_solve_order(self) -> None:
        # z from r3, then y from r2, then x from r1.
        relations = {"r1": {"x", "y"}, "r2": {"y", "z"}, "r3": {"z"}}
        blocks = tarjan_blocks({"x", "y", "z"}, relations)
        order = [block.variables[0] for block in blocks]

        assert order.index("z") < order.index("y") < order.index("x")
        assert all(not block.simultaneous for block in blocks)

    def test_a_cyclic_block_is_detected(self) -> None:
        # x and y appear in both relations: genuinely simultaneous, not substitutable.
        blocks = tarjan_blocks({"x", "y"}, {"r1": {"x", "y"}, "r2": {"x", "y"}})
        simultaneous = [block for block in blocks if block.simultaneous]
        assert len(simultaneous) == 1
        assert simultaneous[0].variables == ["x", "y"]

    def test_it_survives_a_long_chain(self) -> None:
        # A two-hundred-step substitution chain is an ordinary engineering model, and a
        # perfectly ordinary way to exhaust the Python stack with a recursive Tarjan.
        depth = 300
        relations = {f"r{i}": {f"v{i}", f"v{i + 1}"} for i in range(depth)}
        relations[f"r{depth}"] = {f"v{depth}"}
        blocks = tarjan_blocks({f"v{i}" for i in range(depth + 1)}, relations)
        assert len(blocks) == depth + 1


class TestDefine:
    def test_unknown_symbols_auto_register_as_provisional(self) -> None:
        # The registry rots if maintenance is tedious: a system that errors on an undeclared
        # symbol gets one formula entered and then abandoned.
        scope = Scope("s")
        _, provisional = scope.define_formula(Formula("f1", "definition", "energy = mass * c**2"))

        assert set(provisional) == {"energy", "mass", "c"}
        assert scope.variables["mass"].status == "provisional"
        assert "inferred from f1" in scope.variables["mass"].provenance

    def test_an_explicit_definition_promotes_a_provisional_one(self) -> None:
        scope = Scope("s")
        scope.define_formula(Formula("f1", "definition", "y = m * x"))
        scope.define_variable(Variable("m", "slope", status="hyperparameter"))

        assert scope.variables["m"].status == "hyperparameter"
        assert scope.variables["m"].semantics == "slope"

    def test_aliases_resolve_to_one_canonical_variable(self) -> None:
        scope = Scope("s")
        scope.define_variable(Variable("weight", "layer weight", aliases=("theta", "W"), status="free"))
        assert scope.canonical("theta") == "weight"
        assert scope.canonical("W") == "weight"

    def test_constraints_reach_the_cas_as_assumptions(self) -> None:
        scope = Scope("s")
        scope.define_variable(Variable("n", "count", domain="integer", constraints=("n > 0",), status="free"))
        assert scope.session.assumptions_of("n") == {"integer": True, "positive": True}


class TestUndirectedRelations:
    def test_one_relation_solves_in_both_directions(self) -> None:
        # Baking in `y = f(x)` throws away exactly the flexibility that makes the graph useful.
        scope = Scope("s")
        scope.define_formula(Formula("area", "definition", "a = w * h"))

        forward = resolve(scope, "a", {"w": 3, "h": 4})
        backward = resolve(scope, "w", {"a": 12, "h": 4})

        assert forward.value == "12"
        assert backward.value == "3"

    def test_the_relation_map_is_a_set_not_a_direction(self) -> None:
        scope = Scope("s")
        scope.define_formula(Formula("r", "definition", "a = w * h"))
        assert scope.relation_map()["r"] == {"a", "w", "h"}


class TestResolve:
    def test_it_returns_the_path_not_just_the_value(self) -> None:
        scope = build_scope()
        path = resolve(scope, "peak_memory", {
            "d_model": 512, "n_heads": 8, "n_layers": 6, "seq_len": 1024,
            "batch_seqs": 4, "bytes_per_elem": 2, "base_width": 256, "d_ff": 2048,
        })

        # A number with no derivation cannot be checked.
        assert path.solvable
        assert path.steps
        assert any("n_params" in step["variables"] for step in path.steps)
        assert path.value is not None

    def test_an_underdetermined_target_says_which_subset(self) -> None:
        scope = Scope("s")
        scope.define_formula(Formula("r1", "definition", "a = b + c"))

        path = resolve(scope, "a", {})
        assert path.solvable is False
        assert "under-determined" in path.reason
        assert "b" in path.reason and "c" in path.reason

    def test_a_target_nothing_mentions_is_named(self) -> None:
        scope = build_scope()
        path = resolve(scope, "loss", {})
        assert path.solvable is False
        assert "does not appear" in path.reason

    def test_a_chain_of_three_approximations_is_flagged(self) -> None:
        # Compounding approximation error silently is a classic way to be confidently wrong.
        # `step_time` rests on the parameter-count approximation, the FLOPs-per-token
        # approximation, and an empirically fitted MFU — each defensible, and stacked they are
        # a number nobody should plan a cluster booking around without being told.
        scope = build_scope()
        path = resolve(scope, "step_time", {
            "d_model": 512, "n_heads": 8, "n_layers": 6, "seq_len": 1024,
            "batch_seqs": 4, "device_flops": 9.89e14, "mfu": 0.4,
        })

        assert len(path.approximation_chain) >= 3
        assert any("compound" in warning for warning in path.warnings)

    def test_the_path_to_peak_memory_chains_only_two(self) -> None:
        # The flag has to be a real threshold, not something every path trips.
        scope = build_scope()
        path = resolve(scope, "peak_memory", {
            "d_model": 512, "n_heads": 8, "n_layers": 6, "seq_len": 1024,
            "batch_seqs": 4, "bytes_per_elem": 2,
        })
        assert sorted(path.approximation_chain) == ["activation-memory", "param-count"]
        assert path.warnings == []

    def test_a_short_path_is_not_flagged(self) -> None:
        scope = build_scope()
        path = resolve(scope, "d_head", {"d_model": 512, "n_heads": 8})
        assert path.value == "64"
        assert path.approximation_chain == []


class TestOrphans:
    def test_all_six_classes_are_distinguished(self) -> None:
        scope = Scope("s")
        scope.define_variable(Variable("declared_unused", "nobody references this", status="free"))
        scope.define_variable(
            Variable("tokens", "batch in tokens", aliases=("B",), status="free", units="tokens")
        )
        scope.define_variable(Variable("B", "batch in sequences", status="free", units="sequences"))
        scope.define_formula(Formula("f1", "definition", "y = provisional_thing * 2", assumes=("ghost",)))
        scope.define_formula(Formula("f2", "definition", "tokens = B * 512", status="verified"))

        report = find_orphans(scope, roots=("y",))

        assert "provisional_thing" in report.undefined
        assert "declared_unused" in report.unused
        assert "f1" in report.unverified
        assert "ghost" in report.dangling
        # One name, two meanings — invisible to every symbolic check.
        assert report.shadowed
        assert report.shadowed[0]["name"] == "B"

    def test_unreachable_needs_a_root_to_be_meaningful(self) -> None:
        scope = Scope("s")
        scope.define_formula(Formula("f1", "definition", "a = b"))
        scope.define_formula(Formula("f2", "definition", "p = q"))

        assert find_orphans(scope).unreachable == []
        assert set(find_orphans(scope, roots=("a",)).unreachable) == {"p", "q"}


class TestAudit:
    def test_it_never_reports_consistent(self) -> None:
        report = audit(build_scope())
        # "Consistent" is a claim nothing can support; "no contradiction found by X at depth N"
        # is one that can.
        assert "consistent" not in report.summary
        assert "no contradiction found by" in report.summary or "contradiction(s) found" in report.summary

    def test_it_names_the_methods_and_the_depth(self) -> None:
        report = audit(build_scope(), samples=7)
        assert report.depth == 7
        assert "numeric-probe" in report.methods

    def test_the_seeded_mup_conflict_is_caught_with_a_witness(self) -> None:
        # Both formulas are individually plausible and cite a real paper; the disagreement
        # exists only where they meet. Catching it costs a probe; not catching it costs a run.
        report = audit(seed_mup_conflict(build_scope()))

        assert report.witnesses
        witness = next(w for w in report.witnesses if w.tier == "numeric")
        assert set(witness.formulas) == {"init-variance", "mup-init"}
        # A witness, not a verdict: the assignment is reproducible.
        assert witness.assignment
        assert len(set(witness.values.values())) > 1

    def test_a_units_conflict_between_aliases_is_caught(self) -> None:
        scope = Scope("s")
        scope.define_variable(Variable("tokens", "batch", units="tokens", aliases=("B",), status="free"))
        scope.define_variable(Variable("B", "batch", units="sequences", status="free"))

        report = audit(scope)
        assert any(witness.tier == "units" for witness in report.witnesses)

    def test_the_baseline_scope_has_no_contradiction(self) -> None:
        assert audit(build_scope()).witnesses == []


class TestImpact:
    def test_the_closure_is_transitive(self) -> None:
        # A one-hop report leaves the third formula holding a verified stamp it no longer
        # deserves.
        scope = Scope("s")
        scope.define_formula(Formula("f1", "definition", "b = a * 2"))
        scope.define_formula(Formula("f2", "definition", "c = b + 1"))
        scope.define_formula(Formula("f3", "definition", "d = c * c"))
        scope.define_formula(Formula("unrelated", "definition", "z = w"))

        report = impact(scope, "a")
        assert report.downstream_formulas == ["f1", "f2", "f3"]
        assert set(report.downstream_variables) == {"b", "c", "d"}

    def test_staleness_propagates_and_only_to_verified_formulas(self) -> None:
        scope = Scope("s")
        scope.define_formula(Formula("f1", "definition", "b = a * 2", status="verified"))
        scope.define_formula(Formula("f2", "definition", "c = b + 1", status="verified"))
        scope.define_formula(Formula("f3", "definition", "d = c * c", status="unverified"))

        report = impact(scope, "a", mark_stale=True)
        assert report.newly_stale == ["f1", "f2"]
        assert scope.formulas["f1"].status == "stale"
        assert scope.formulas["f3"].status == "unverified"

    def test_staleness_clears_on_re_verification(self) -> None:
        from dataclasses import replace

        scope = Scope("s")
        scope.define_formula(Formula("f1", "definition", "b = a * 2", status="verified"))
        impact(scope, "a", mark_stale=True)
        assert scope.formulas["f1"].status == "stale"

        scope.formulas["f1"] = replace(scope.formulas["f1"], status="verified")
        assert find_orphans(scope).unverified == []

    def test_d_model_moves_everything_downstream(self) -> None:
        report = impact(build_scope(), "d_model")
        assert {"param-count", "flops-per-token", "peak-memory", "mup-lr", "init-variance"} <= set(
            report.downstream_formulas
        )


class TestAssumptions:
    def test_a_result_reports_what_it_rests_on(self) -> None:
        closure = assumption_closure(build_scope(), "peak-memory")
        assert "no-checkpointing" in closure["rests_on"]

    def test_relaxing_an_assumption_says_what_dies(self) -> None:
        # "If I relax iid, what dies?" is the useful direction.
        report = relax(build_scope(), "ffn-ratio")
        assert set(report["directly_invalidated"]) == {"param-count", "flops-per-token", "step-time"}
        assert "peak-memory" in report["transitively_affected"]

    def test_an_unknown_assumption_is_named(self) -> None:
        with pytest.raises(ContractError, match="unknown assumption"):
            relax(build_scope(), "nonexistent")


class TestFork:
    def test_a_fork_is_isolated_from_its_parent(self) -> None:
        baseline = build_scope("baseline")
        moe = baseline.fork("moe")
        moe.define_formula(Formula("param-count", "approximation", "n_params = 12 * n_layers * d_model**2 * 8"))

        # A fork sharing its parent's dictionaries corrupts the baseline the moment anyone
        # edits it, which is the exact failure fork exists to prevent.
        assert "8" not in baseline.formulas["param-count"].expression
        assert baseline.formulas["param-count"].expression != moe.formulas["param-count"].expression

    def test_a_fork_records_its_parent(self) -> None:
        assert build_scope("baseline").fork("v2").parent == "baseline"

    def test_diff_reports_added_removed_and_changed(self) -> None:
        baseline = build_scope("baseline")
        variant = baseline.fork("variant")
        variant.define_formula(Formula("moe-experts", "definition", "n_experts = 8"))
        variant.define_formula(Formula("mup-lr", "definition", "lr_scale = base_width / d_model**2"))
        del variant.formulas["kv-cache"]

        difference = diff_contexts(baseline, variant)
        assert "formula:moe-experts" in difference["added"]
        assert "formula:kv-cache" in difference["removed"]
        assert any(entry["id"] == "mup-lr" for entry in difference["changed"])


class TestTokensVersusSequences:
    def test_the_ambiguity_is_resolved_by_construction(self) -> None:
        # `B` meaning sequences in one equation and tokens in another is the canonical
        # cross-formula error, and the fix is two variables with a relation between them.
        scope = build_scope()
        assert scope.variables["batch_tokens"].units == "tokens"
        assert scope.variables["batch_seqs"].units == "sequences"

        path = resolve(scope, "batch_tokens", {"batch_seqs": 4, "seq_len": 1024})
        assert path.value == "4096"

    def test_activation_memory_uses_tokens_and_kv_cache_uses_sequences(self) -> None:
        scope = build_scope()
        relations = scope.relation_map()
        assert "batch_tokens" in relations["activation-memory"]
        assert "batch_seqs" in relations["kv-cache"]
