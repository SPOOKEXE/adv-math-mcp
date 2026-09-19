"""Contract export, explain and rename coverage."""

from __future__ import annotations

import pytest

from math_mcp.contract.model import Assumption, Formula, Scope, Variable, Expression
from math_mcp.contract.export import export_scope, explain
from math_mcp.server import MathServer


class TestExport:
    def test_formula_and_expression_with_same_id_keep_separate_edges(self) -> None:
        scope = Scope("s")
        scope.define_formula(Formula("shared", "definition", "y = x"))
        scope.define_expression(Expression("shared", "z + 1"))
        graph = export_scope(scope, {"format": "json"})["graph"]
        edges = {(edge["from"], edge["to"]) for edge in graph["edges"]}
        assert ("formula:shared", "var:y") in edges
        assert ("expression:shared", "var:z") in edges
        assert ("formula:shared", "var:z") not in edges

    def test_item_export_respects_format(self) -> None:
        scope = Scope("s")
        scope.define_formula(Formula("law", "definition", "y = x"))
        latex = export_scope(scope, {"format": "latex", "noun": "formula", "id": "law"})
        dot = export_scope(scope, {"format": "dot", "noun": "formula", "id": "law"})
        assert latex["content"].startswith("\\documentclass")
        assert "x = y" not in latex["content"]
        assert "y = x" in latex["content"]
        assert '"formula:law" -> "var:x"' in dot["content"]

    def test_adhoc_expression_export_respects_format(self) -> None:
        scope = Scope("s")
        dot = export_scope(scope, {"format": "dot", "expr": "x + 1"})
        graph = export_scope(scope, {"format": "json", "expr": "x + 1"})["graph"]
        assert dot["content"].startswith("digraph math {")
        assert {node["id"] for node in graph["nodes"]} == {"expression:input", "var:x"}

    def test_export_dot_contains_formula_and_variable_edges(self) -> None:
        scope = Scope("s")
        scope.define_variable(Variable("x", symbol="x", units="m"))
        scope.define_variable(Variable("y", symbol="y", units="m/s"))
        scope.define_formula(Formula("law", "definition", "y = x * t"))
        result = export_scope(scope, {"format": "dot"})
        content = result["content"]
        assert content.startswith("digraph math {")
        assert '"formula:law" -> "var:x"' in content
        assert '"formula:law" -> "var:y"' in content
        assert '"formula:law" -> "var:t"' in content

    def test_export_dot_includes_notation_and_links(self) -> None:
        scope = Scope("s")
        scope.define_variable(Variable("x", symbol="x", subscript="i", links=("y",), units="m"))
        scope.define_variable(Variable("y", symbol="y", links=("x",), units="m/s"))
        scope.define_formula(Formula("law", "definition", "y = x * t"))
        result = export_scope(scope, {"format": "dot"})
        content = result["content"]
        assert "var:x" in content
        assert "var:y" in content
        assert "[style=dotted]" in content

    def test_export_json_returns_graph(self) -> None:
        scope = Scope("s")
        scope.define_variable(Variable("x", symbol="x", subscript="i", units="m"))
        scope.define_variable(Variable("y", symbol="y", units="m/s"))
        scope.define_formula(Formula("law", "definition", "y = x * t"))
        scope.define_expression(Expression("energy", "x**2 + y**2", description="combined"))
        result = export_scope(scope, {"format": "json"})
        graph = result["graph"]
        assert graph["directed"] is True
        assert len(graph["nodes"]) >= 4
        assert any(n["kind"] == "formula" for n in graph["nodes"])
        assert any(n["kind"] == "expression" for n in graph["nodes"])
        assert len(graph["edges"]) >= 3

    def test_export_latex_contains_sections_and_hyperlinks(self) -> None:
        scope = Scope("s")
        scope.define_variable(Variable("x", symbol="x", subscript="i", units="m"))
        scope.define_variable(Variable("y", symbol="y", units="m/s"))
        scope.define_formula(Formula("law", "definition", "y = x * t"))
        result = export_scope(scope, {"format": "latex"})
        content = result["content"]
        assert "\\documentclass" in content
        assert "\\section*{Variables}" in content
        assert "\\section*{Formulas}" in content
        assert "\\hypertarget{formula:law}{}" in content
        assert "\\hyperlink{formula:law}{law}" in content
        assert "\\hyperlink{var:x}{\\(x_{i}\\)}" in content

    def test_export_latex_formula_uses_definition_text_not_residual(self) -> None:
        scope = Scope("s")
        scope.define_variable(Variable("x", symbol="x"))
        scope.define_variable(Variable("y", symbol="y"))
        scope.define_formula(Formula("law", "definition", "y = x * t"))
        result = export_scope(scope, {"format": "latex"})
        content = result["content"]
        assert "\\text{definition: }" in content
        assert "residual form" not in content

    def test_export_raises_on_unknown_format(self) -> None:
        scope = Scope("s")
        with pytest.raises(Exception, match="unknown export format"):
            export_scope(scope, {"format": "svg"})

    def test_export_expr_raises_on_combined_noun_and_expr(self) -> None:
        scope = Scope("s")
        with pytest.raises(Exception, match="expr cannot be combined"):
            export_scope(scope, {"format": "latex", "expr": "x", "noun": "variable"})

    def test_export_latex_raises_on_unknown_noun(self) -> None:
        scope = Scope("s")
        with pytest.raises(Exception, match="unknown variable"):
            export_scope(scope, {"format": "latex", "noun": "variable", "id": "missing"})

    def test_export_json_raises_on_unknown_expression(self) -> None:
        scope = Scope("s")
        with pytest.raises(Exception, match="unknown expression"):
            export_scope(scope, {"format": "json", "noun": "expression", "id": "missing"})


class TestExplain:
    def test_explain_formula_returns_metadata(self) -> None:
        scope = Scope("s")
        scope.define_variable(Variable("x", symbol="x", subscript="i", units="m", semantics="position"))
        scope.define_variable(Variable("y", symbol="y", units="m/s", semantics="velocity"))
        scope.define_formula(Formula("law", "definition", "y = x * t"))
        result = explain(scope, {"noun": "formula", "id": "law"})
        assert result["kind"] == "formula"
        assert result["id"] == "law"
        assert result["kind_of"] == "definition"
        assert result["expression"] == "y = x * t"
        assert result["variables"]["x"]["symbol"] == "x"
        assert result["variables"]["x"]["subscript"] == "i"
        assert result["variables"]["x"]["units"] == "m"
        assert result["variables"]["x"]["links"] == []
        assert result["variables"]["y"]["semantics"] == "velocity"
        assert result["direct_assumptions"] == []
        assert result["missing_assumptions"] == []
        assert result["inactive_assumptions"] == []
        assert result["summary"]
        assert "no claim about truth" in result["summary"]

    def test_explain_formula_includes_missing_and_inactive_assumptions(self) -> None:
        scope = Scope("s")
        # ghost is never defined: it is missing
        scope.define_formula(Formula("law", "definition", "y = x", assumes=("ghost",)))
        result = explain(scope, {"noun": "formula", "id": "law"})
        assert "ghost" in result["missing_assumptions"]

    def test_explain_formula_includes_inactive_assumptions(self) -> None:
        scope = Scope("s")
        scope.define_assumption(Assumption("iid", "samples are iid", active=False))
        scope.define_formula(Formula("law", "definition", "y = x", assumes=("iid",)))
        result = explain(scope, {"noun": "formula", "id": "law"})
        assert "iid" in result["inactive_assumptions"]

    def test_explain_formula_includes_related_formulas(self) -> None:
        scope = Scope("s")
        scope.define_variable(Variable("x", symbol="x"))
        scope.define_variable(Variable("y", symbol="y"))
        scope.define_variable(Variable("z", symbol="z"))
        scope.define_formula(Formula("law1", "definition", "y = x * 2"))
        scope.define_formula(Formula("law2", "definition", "z = y + 1"))
        result = explain(scope, {"noun": "formula", "id": "law1"})
        assert "law2" in result["related_formulas"]

    def test_explain_expression_returns_metadata(self) -> None:
        scope = Scope("s")
        scope.define_variable(Variable("x", symbol="x", subscript="i", units="m"))
        scope.define_expression(Expression("energy", "x**2", description="kinetic"))
        result = explain(scope, {"noun": "expression", "id": "energy"})
        assert result["kind"] == "expression"
        assert result["id"] == "energy"
        assert result["description"] == "kinetic"
        assert result["variables"]["x"]["subscript"] == "i"
        assert result["variables"]["x"]["links"] == []
        assert "no claim about truth" in result["summary"]

    def test_explain_raises_on_unknown_formula(self) -> None:
        scope = Scope("s")
        with pytest.raises(Exception, match="unknown formula"):
            explain(scope, {"noun": "formula", "id": "missing"})

    def test_explain_raises_on_missing_id(self) -> None:
        scope = Scope("s")
        with pytest.raises(Exception, match="id is required"):
            explain(scope, {"noun": "formula"})


class TestRename:
    def test_tool_schema_and_expression_listing(self) -> None:
        from math_mcp.server import TOOL_SCHEMAS

        schemas = {tool["name"]: tool["inputSchema"] for tool in TOOL_SCHEMAS}
        assert "expression" in schemas["define"]["properties"]["noun"]["enum"]
        assert "expression" in schemas["list"]["properties"]["noun"]["enum"]
        server = MathServer()
        server.call("define", {"noun": "expression", "body": {"id": "e", "expression": "x + 1"}})
        assert server.call("list", {"noun": "expression"})["ids"] == ["e"]
        assert "x" in server.scopes["default"].variables

    def test_variable_notation_reaches_export(self) -> None:
        server = MathServer()
        server.call("define", {"noun": "variable", "body": {"name": "x", "symbol": "x", "subscript": "i"}})
        content = server.call("export", {"format": "latex"})["content"]
        assert "x_{i}" in content

    def test_rename_assumption_updates_formula_references(self) -> None:
        server = MathServer()
        server.call("define", {"noun": "assumption", "body": {"id": "old", "statement": "x > 0"}})
        server.call("define", {"noun": "formula", "body": {"id": "f", "expression": "y=x", "assumes": ["old"]}})
        server.call("rename", {"noun": "assumption", "old_id": "old", "new_id": "new"})
        assert server.scopes["default"].formulas["f"].assumes == ("new",)

    def test_rename_variable(self) -> None:
        server = MathServer()
        server.define({"noun": "variable", "body": {"name": "old", "semantics": "test"}})
        result = server.call("rename", {"noun": "variable", "old_id": "old", "new_id": "new_var"})
        assert result["renamed"]["new_id"] == "new_var"
        assert server.scopes["default"].variables["new_var"].name == "new_var"
        assert "old" not in server.scopes["default"].variables

    def test_rename_preserves_aliases(self) -> None:
        server = MathServer()
        server.define({"noun": "variable", "body": {"name": "old", "aliases": ("alias1",)}})
        server.call("rename", {"noun": "variable", "old_id": "old", "new_id": "new_var"})
        var = server.scopes["default"].variables["new_var"]
        assert "alias1" in var.aliases
        assert "old" in var.aliases

    def test_rename_variable_conflict_raises(self) -> None:
        server = MathServer()
        server.define({"noun": "variable", "body": {"name": "x"}})
        server.define({"noun": "variable", "body": {"name": "y"}})
        result = server.call("rename", {"noun": "variable", "old_id": "x", "new_id": "y"})
        assert result["error"] == "ContractError"

    def test_rename_formula(self) -> None:
        server = MathServer()
        server.define({"noun": "formula", "body": {"id": "old", "expression": "y = x"}})
        result = server.call("rename", {"noun": "formula", "old_id": "old", "new_id": "new_formula"})
        assert result["renamed"]["new_id"] == "new_formula"
        assert "new_formula" in server.scopes["default"].formulas
        assert "old" not in server.scopes["default"].formulas

    def test_rename_formula_conflict_raises(self) -> None:
        server = MathServer()
        server.define({"noun": "formula", "body": {"id": "f1", "expression": "y = x"}})
        server.define({"noun": "formula", "body": {"id": "f2", "expression": "z = x"}})
        result = server.call("rename", {"noun": "formula", "old_id": "f1", "new_id": "f2"})
        assert result["error"] == "ContractError"

    def test_rename_assumption(self) -> None:
        server = MathServer()
        server.define({"noun": "assumption", "body": {"id": "old", "statement": "x > 0"}})
        result = server.call("rename", {"noun": "assumption", "old_id": "old", "new_id": "new_assumption"})
        assert result["renamed"]["new_id"] == "new_assumption"
        assert "new_assumption" in server.scopes["default"].assumptions
        assert "old" not in server.scopes["default"].assumptions

    def test_rename_expression(self) -> None:
        server = MathServer()
        server.define({"noun": "expression", "body": {"id": "old", "expression": "x + y"}})
        result = server.call("rename", {"noun": "expression", "old_id": "old", "new_id": "new_expr"})
        assert result["renamed"]["new_id"] == "new_expr"
        assert "new_expr" in server.scopes["default"].expressions
        assert "old" not in server.scopes["default"].expressions

    def test_rename_expression_conflict_raises(self) -> None:
        server = MathServer()
        server.define({"noun": "expression", "body": {"id": "e1", "expression": "x + y"}})
        server.define({"noun": "expression", "body": {"id": "e2", "expression": "a + b"}})
        result = server.call("rename", {"noun": "expression", "old_id": "e1", "new_id": "e2"})
        assert result["error"] == "ContractError"

    def test_rename_raises_on_unknown_noun(self) -> None:
        server = MathServer()
        result = server.call("rename", {"noun": "theorem", "old_id": "x", "new_id": "y"})
        assert result["error"] == "ContractError"
        assert "unknown noun" in result["message"]

    def test_rename_raises_on_missing_old_id(self) -> None:
        server = MathServer()
        result = server.call("rename", {"noun": "variable", "old_id": "missing", "new_id": "new"})
        assert result["error"] == "ContractError"
        assert "unknown variable" in result["message"]

    def test_rename_preserves_scope_after_load(self, tmp_path) -> None:
        server = MathServer(root=tmp_path)
        server.define({"noun": "variable", "body": {"name": "x"}})
        server.call("rename", {"noun": "variable", "old_id": "x", "new_id": "y"})
        server.env({"action": "save", "name": "rename-test"})
        server.env({"action": "clear"})
        server.env({"action": "load", "name": "rename-test"})
        assert "y" in server.scopes["default"].variables
        assert "x" not in server.scopes["default"].variables
