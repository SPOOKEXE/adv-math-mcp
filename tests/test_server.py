"""The MCP surface: schemas, dispatch, and the launch contract."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from math_mcp.contract.transformer import build_scope, seed_mup_conflict
from math_mcp.server import TOOL_SCHEMAS, MathServer

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def server() -> MathServer:
    return MathServer()


class TestSchemas:
    def test_every_schema_is_well_formed(self) -> None:
        for schema in TOOL_SCHEMAS:
            assert schema["name"]
            assert schema["description"]
            assert schema["inputSchema"]["type"] == "object"

    def test_every_declared_tool_has_a_handler(self, server: MathServer) -> None:
        assert {schema["name"] for schema in TOOL_SCHEMAS} == set(server.handlers)

    def test_the_contract_layer_is_six_verbs_not_a_verb_per_noun(self) -> None:
        # Tool count is a context cost paid on every turn. `define-variable` +
        # `define-formula` + `define-assumption` is three schemas describing one operation.
        names = {schema["name"] for schema in TOOL_SCHEMAS}
        assert {"define", "list", "audit", "resolve", "fork", "impact"} <= names
        assert not any(name.startswith("define-") for name in names)

    def test_the_launch_contract_resolves(self) -> None:
        contract = tomllib.loads((ROOT / "server.toml").read_text())["server"]
        assert contract["id"] == "math"
        assert contract["transport"] == "stdio"
        # The declared tool list must match what the server actually serves, or a health check
        # passes against a server missing half its surface.
        assert set(contract["tools"]) == {schema["name"] for schema in TOOL_SCHEMAS}


class TestDispatch:
    def test_an_unknown_tool_lists_what_exists(self, server: MathServer) -> None:
        result = server.call("integrate_everything", {})
        assert result["error"] == "unknown-tool"
        assert "parse" in result["available"]

    def test_a_parse_error_crosses_the_boundary_as_data(self, server: MathServer) -> None:
        # An exception becomes a transport error the model cannot act on; a column position is
        # something it can fix.
        result = server.call("parse", {"text": "x + * 2"})
        assert result["error"] == "parse"
        assert result["column"] >= 1

    def test_a_hostile_expression_is_refused_through_the_tool(self, server: MathServer) -> None:
        result = server.call("parse", {"text": "__import__('os').system('id')"})
        assert result["error"] == "parse"

    def test_a_missing_argument_is_named(self, server: MathServer) -> None:
        assert server.call("parse", {})["error"] == "missing-argument"

    def test_a_domain_error_is_structured(self, server: MathServer) -> None:
        result = server.call("matrix_grad", {"expr": "x", "wrt": ["x"], "layout": "sideways"})
        assert result["error"] == "MathError"
        assert "never safe to guess" in result["message"]


class TestCasTools:
    def test_parse_returns_a_handle_and_a_preview(self, server: MathServer) -> None:
        result = server.call("parse", {"text": "x**2 + 2*x + 1"})
        assert result["expr_id"].startswith("e:")
        assert result["free_symbols"] == ["x"]
        assert result["pretty"]

    def test_a_handle_is_accepted_wherever_an_expression_is(self, server: MathServer) -> None:
        handle = server.call("parse", {"text": "(x + 1)**2"})["expr_id"]
        result = server.call("check_equivalence", {"left": handle, "right": "x**2 + 2*x + 1"})
        assert result["verdict"] == "proved"

    def test_declare_then_check_changes_the_answer(self, server: MathServer) -> None:
        assert server.call("check_equivalence", {"left": "sqrt(x**2)", "right": "x"})["verdict"] == "disproved"

        server.call("declare", {"name": "x", "assumptions": {"nonnegative": True}})
        assert server.call("check_equivalence", {"left": "sqrt(x**2)", "right": "x"})["verdict"] == "proved"

    def test_a_batch_comes_back_indexed(self, server: MathServer) -> None:
        result = server.call("check_equivalence", {"pairs": [["(x+1)**2", "x**2+2*x+1"], ["x", "x+1"]]})
        assert [entry["index"] for entry in result["results"]] == [0, 1]
        assert [entry["verdict"] for entry in result["results"]] == ["proved", "disproved"]

    def test_check_derivation_reports_the_step_index(self, server: MathServer) -> None:
        result = server.call("check_derivation", {"steps": ["(x+1)**2", "x**2+2*x+1", "x**2+1"]})
        assert result["first_invalid_step"] == 2

    def test_matrix_grad_states_its_layout(self, server: MathServer) -> None:
        result = server.call("matrix_grad", {"expr": "Matrix([x*y, x+y])", "wrt": ["x", "y"], "layout": "numerator"})
        assert result["layout"] == "numerator"
        assert result["shape"] == [2, 2]

    def test_check_grad_rejects_a_wrong_gradient(self, server: MathServer) -> None:
        result = server.call("check_grad", {"expr": "x**2 * y", "claimed": "Matrix([2*x*y, -x**2])", "wrt": ["x", "y"]})
        assert result["ok"] is False

    def test_to_code_returns_source_and_the_op_counts(self, server: MathServer) -> None:
        result = server.call("to_code", {"expr": "exp(x*y) + log(x*y) + (x*y)**2", "target": "torch"})
        assert "torch.exp" in result["source"]
        assert result["operations_after"] < result["operations_before"]

    def test_shape_check_names_the_axis(self, server: MathServer) -> None:
        result = server.call("shape_check", {"spec": "bd,bd->b", "shapes": {"a": ["batch", "d"], "z": ["beams", "d"]}})
        assert result["ok"] is False
        assert result["axis"] == "b"

    def test_tensor_plan_crosses_the_server_boundary(self, server: MathServer) -> None:
        result = server.call(
            "tensor_plan",
            {
                "kind": "contraction",
                "spec": "mk,kn->mn",
                "inputs": ["a", "b"],
                "tensors": {"a": {"shape": [4, 8]}, "b": {"shape": [8, 2]}},
                "memory": {"fast_bytes": 1024},
            },
        )
        assert result["status"] == "feasible"
        assert result["operations"][0]["flops"] == 128

    def test_analyze_crosses_the_server_boundary(self, server: MathServer) -> None:
        result = server.call("analyze", {"op": "rigorous_bounds", "expr": "x**2", "box": {"x": [-1, 2]}})
        assert result["verdict"] == "proved"
        assert result["enclosure"]["upper"] == "4"


class TestContractTools:
    def _load(self, server: MathServer) -> MathServer:
        build_scope("default", server.scopes["default"])
        return server

    def test_define_auto_registers_and_reports_it(self, server: MathServer) -> None:
        result = server.call("define", {"noun": "formula", "body": {"id": "f1", "expression": "energy = mass * c**2"}})
        assert set(result["auto_registered"]) == {"energy", "mass", "c"}

    def test_define_covers_all_three_nouns(self, server: MathServer) -> None:
        assert server.call("define", {"noun": "variable", "body": {"name": "x"}})["defined"] == "variable"
        assert server.call("define", {"noun": "formula", "body": {"id": "f", "expression": "y = x"}})["defined"] == "formula"
        assert server.call("define", {"noun": "assumption", "body": {"id": "a", "statement": "x > 0"}})["defined"] == "assumption"

    def test_an_unknown_noun_is_named(self, server: MathServer) -> None:
        assert "unknown noun" in server.call("define", {"noun": "theorem", "body": {}})["message"]

    def test_list_defaults_to_a_summary(self, server: MathServer) -> None:
        self._load(server)
        summary = server.call("list", {"noun": "formula"})

        # Dumping 200 relations into context defeats the purpose of having a registry.
        assert "entries" not in summary
        assert summary["count"] == len(summary["ids"])
        assert "param-count" in summary["ids"]

    def test_list_full_mode_returns_the_entries(self, server: MathServer) -> None:
        self._load(server)
        full = server.call("list", {"noun": "formula", "mode": "full"})
        assert all("expression" in entry for entry in full["entries"])

    def test_list_filters(self, server: MathServer) -> None:
        self._load(server)
        approximations = server.call("list", {"noun": "formula", "filter": {"kind": "approximation"}})
        assert set(approximations["ids"]) == {"param-count", "flops-per-token", "activation-memory"}

    def test_resolve_returns_the_path(self, server: MathServer) -> None:
        self._load(server)
        result = server.call("resolve", {"target": "d_head", "given": {"d_model": 512, "n_heads": 8}})
        assert result["value"] == "64"
        assert result["steps"]

    def test_audit_never_says_consistent(self, server: MathServer) -> None:
        self._load(server)
        assert "no contradiction found by" in server.call("audit", {})["summary"]

    def test_audit_reports_the_seeded_conflict_as_a_witness(self, server: MathServer) -> None:
        seed_mup_conflict(build_scope("default", server.scopes["default"]))
        report = server.call("audit", {})

        assert report["witnesses"]
        assert "contradiction(s) found" in report["summary"]

    def test_impact_and_its_two_alternate_queries(self, server: MathServer) -> None:
        self._load(server)

        downstream = server.call("impact", {"symbol": "d_model"})
        assert "param-count" in downstream["downstream_formulas"]

        closure = server.call("impact", {"formula": "peak-memory"})
        assert "no-checkpointing" in closure["rests_on"]

        relaxed = server.call("impact", {"relax": "ffn-ratio"})
        assert "param-count" in relaxed["directly_invalidated"]

    def test_fork_isolates_and_diffs(self, server: MathServer) -> None:
        self._load(server)
        server.call("fork", {"name": "moe"})

        server.scopes["moe"].formulas.pop("kv-cache")
        difference = server.call("fork", {"scope": "default", "diff_against": "moe"})
        assert "formula:kv-cache" in difference["removed"]

        # The baseline is untouched.
        assert "kv-cache" in server.scopes["default"].formulas

    def test_forking_onto_an_existing_name_is_refused(self, server: MathServer) -> None:
        self._load(server)
        server.call("fork", {"name": "v2"})
        # Silently overwriting a scope loses whatever was in it, with no way to notice.
        assert "already exists" in server.call("fork", {"name": "v2"})["message"]

    def test_an_unknown_scope_is_named(self, server: MathServer) -> None:
        assert "unknown scope" in server.call("resolve", {"target": "x", "scope": "ghost"})["message"]


class TestEnvironment:
    """Putting one situation down and picking another up.

    The unit worth defending is *the whole environment*. A scope whose `x` is positive in one
    session and unconstrained in the next is not the same scope however identical its records
    read, so a save that carried the records and not the declarations would restore something
    that looks right and computes differently.
    """

    @pytest.fixture()
    def env(self, tmp_path: Path) -> MathServer:
        server = MathServer(tmp_path)
        server.declare({"name": "x", "assumptions": {"positive": True}})
        server.define({"noun": "variable", "body": {"name": "x", "semantics": "the input"}})
        server.define({"noun": "assumption", "body": {"id": "iid", "statement": "samples are iid"}})
        server.define({"noun": "formula", "body": {"id": "f1", "expression": "y = 2*x", "assumes": ["iid"]}})
        return server

    def test_a_saved_environment_comes_back_whole(self, env: MathServer) -> None:
        env.env({"action": "save", "name": "paper-v1"})
        env.env({"action": "clear"})
        env.env({"action": "load", "name": "paper-v1"})

        scope = env.scope()
        assert sorted(scope.variables) == ["x", "y"]
        assert list(scope.formulas) == ["f1"]
        assert list(scope.assumptions) == ["iid"]

    def test_declarations_come_back_too(self, env: MathServer) -> None:
        # The half a records-only save would lose. `x` positive and `x` unconstrained are
        # different symbols to SymPy, and an environment that restored one as the other would
        # compute different answers from identical-looking records.
        env.env({"action": "save", "name": "declared"})
        env.env({"action": "clear"})
        assert env.session.assumptions_of("x") == {}

        env.env({"action": "load", "name": "declared"})
        assert env.session.assumptions_of("x") == {"positive": True}

    def test_every_scope_is_saved_not_just_the_default(self, env: MathServer) -> None:
        # Forks are the reason scopes exist. Saving only `default` would silently drop the
        # branch someone was actually working in.
        env.fork({"name": "v2"})
        env.env({"action": "save", "name": "both"})
        env.env({"action": "clear"})

        assert sorted(env.env({"action": "load", "name": "both"})["scopes"]) == ["default", "v2"]

    def test_clear_keeps_default_rather_than_emptying_the_registry(self, env: MathServer) -> None:
        # A registry with no scopes is not a clean slate; it is a server that answers
        # `unknown scope` to everything until someone forks one.
        env.env({"action": "clear"})

        assert env.scope().name == "default"
        assert env.scope().variables == {}

    def test_load_replaces_rather_than_merges(self, env: MathServer) -> None:
        # Two half-environments sharing a symbol table is the failure this exists to prevent,
        # and there is no answer to "which `x` wins" that is right more than half the time.
        env.env({"action": "save", "name": "first"})
        env.env({"action": "clear"})
        env.define({"noun": "variable", "body": {"name": "z", "semantics": "unrelated"}})
        env.env({"action": "load", "name": "first"})

        assert "z" not in env.scope().variables

    def test_list_reports_what_is_there(self, env: MathServer) -> None:
        env.env({"action": "save", "name": "paper-v1", "note": "the paper formulation"})
        saved = env.env({"action": "list"})["saved"]

        assert saved[0]["name"] == "paper-v1"
        assert saved[0]["note"] == "the paper formulation"
        assert saved[0]["symbols"] == 1

    def test_an_unreadable_file_is_listed_and_marked(self, env: MathServer, tmp_path: Path) -> None:
        # Exactly what someone asking "what have I saved" needs to be told about. Omitting it
        # would answer the question wrongly rather than incompletely.
        (tmp_path / "corrupt.json").write_text("{not json", encoding="utf-8")

        assert env.env({"action": "list"})["saved"] == [{"name": "corrupt", "unreadable": True}]

    def test_deleting_one_leaves_the_others(self, env: MathServer) -> None:
        env.env({"action": "save", "name": "keep"})
        env.env({"action": "save", "name": "drop"})
        env.env({"action": "delete", "name": "drop"})

        assert [entry["name"] for entry in env.env({"action": "list"})["saved"]] == ["keep"]

    def test_loading_a_name_that_was_never_saved_says_where_to_look(self, env: MathServer) -> None:
        result = env.call("env", {"action": "load", "name": "absent"})

        assert "env list" in result["message"]

    def test_a_name_that_is_a_path_is_refused_rather_than_sanitised(self, env: MathServer) -> None:
        # A sanitised name is a different name, and the caller is never told which file it got.
        for name in ["../escape", "a/b", ".hidden", ""]:
            assert env.call("env", {"action": "save", "name": name})["error"]

    def test_an_unknown_action_lists_the_real_ones(self, env: MathServer) -> None:
        assert "save" in env.call("env", {"action": "teleport"})["message"]

    def test_a_failed_write_leaves_no_partial_file(self, env: MathServer, monkeypatch: pytest.MonkeyPatch) -> None:
        # A half-written environment that still parses is worse than none: it loads, and is
        # quietly missing half the work.
        import os as os_module

        monkeypatch.setattr(os_module, "replace", lambda *_: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError):
            env.env({"action": "save", "name": "doomed"})

        assert env.env({"action": "list"})["saved"] == []


class TestDomainTools:
    """The seven new doors dispatch, and the shared flags behave the same on each."""

    def test_solve_dispatches_and_verifies(self, server: MathServer) -> None:
        result = server.call("solve", {"kind": "equation", "exprs": ["x**2 = 4"], "wrt": ["x"]})
        assert {entry["x"] for entry in result["solutions"]} == {"-2", "2"}
        assert result["verified"] == ["proved", "proved"]

    def test_render_is_opt_in_and_returns_latex(self, server: MathServer) -> None:
        plain = server.call("simplify", {"expr": "x**2 + 2*x + 1"})
        assert "render" not in plain
        rendered = server.call("simplify", {"expr": "x**2 + 2*x + 1", "render": True})
        assert rendered["render"]["latex"] == r"\left(x + 1\right)^{2}"

    def test_calc_integrate_travels_with_its_verdict(self, server: MathServer) -> None:
        result = server.call("calc", {"op": "integrate", "expr": "cos(x)", "wrt": "x"})
        assert result["pretty"] == "sin(x)"
        assert result["verified"] == "proved"

    def test_matrix_grad_grows_a_hessian(self, server: MathServer) -> None:
        result = server.call("matrix_grad", {"expr": "x**2 + x*y", "wrt": ["x", "y"], "hessian": True})
        assert result["hessian"] is True
        assert result["shape"] == [2, 2]

    def test_the_jacobian_path_still_demands_a_layout(self, server: MathServer) -> None:
        # `hessian: true` relaxed the schema, not the rule: a first-order derivative without a
        # layout is still never safe to guess.
        result = server.call("matrix_grad", {"expr": "x**2", "wrt": ["x"]})
        assert result["error"] == "MathError"
        assert "never safe to guess" in result["message"]

    def test_linalg_numtheory_prob_eval_dispatch(self, server: MathServer) -> None:
        assert server.call("linalg", {"op": "det", "matrix": [["1", "2"], ["3", "4"]]})["det"] == "-2"
        assert server.call("numtheory", {"op": "is_prime", "values": ["97"]})["is_prime"] is True
        assert (
            server.call("prob", {"op": "probability", "family": "normal", "params": {"mean": "0", "std": "1"}, "condition": "X > 0"})[
                "pretty"
            ]
            == "1/2"
        )
        assert server.call("eval", {"op": "evalf", "expr": "E", "digits": 10})["value"].startswith("2.718281828")

    def test_solved_expressions_share_the_session_with_every_other_tool(self, server: MathServer) -> None:
        # One workspace: a handle minted by `calc` is a first-class citizen of `check_equivalence`.
        handle = server.call("calc", {"op": "integrate", "expr": "2*x", "wrt": "x"})["expr_id"]
        verdict = server.call("check_equivalence", {"left": handle, "right": "x**2"})["verdict"]
        assert verdict == "proved"
