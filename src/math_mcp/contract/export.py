"""LaTeX, DOT and structured exports for one scope."""

from __future__ import annotations

import json
import re
from typing import Any

import sympy

from .model import ContractError, Expression, Formula, Scope, Variable


def _safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", text)[:64]


def _tex_text(text: str) -> str:
    return "".join({"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}.get(char, char) for char in text)


def _math_symbol(var: Variable) -> str:
    parts: list[str] = []
    if var.symbol:
        parts.append(var.symbol)
    if var.subscript:
        parts.append(f"_{{{var.subscript}}}")
    if var.superscript:
        parts.append(f"^{{{var.superscript}}}")
    return "".join(parts) or var.name


def _annotation(var: Variable) -> str:
    meta = []
    if var.symbol or var.subscript or var.superscript:
        meta.append(f"notation: ${_math_symbol(var)}$")
    if var.links:
        meta.append("links: " + ", ".join(sorted(var.links)))
    if var.units:
        meta.append("units: " + var.units)
    if var.semantics:
        meta.append("semantics: " + var.semantics)
    return " \\\\ ".join(meta)


def _variable_nodes(scope: Scope) -> tuple[dict[str, str], dict[str, str]]:
    """Node ids and labels for variables."""
    nodes: dict[str, str] = {}
    labels: dict[str, str] = {}
    for variable in sorted(scope.variables.values(), key=lambda v: v.name):
        nid = f"var:{variable.name}"
        nodes[nid] = variable.name
        math = _math_symbol(variable)
        text = f"${math}$"
        if variable.units or variable.semantics:
            text += f" ({variable.units or '?'}, {variable.semantics or '?'})"
        labels[nid] = text.replace("&", "\\&")
    return nodes, labels


def _formula_nodes(scope: Scope) -> tuple[dict[str, str], dict[str, str]]:
    """Node ids and labels for formulas."""
    nodes: dict[str, str] = {}
    labels: dict[str, str] = {}
    for formula in sorted(scope.formulas.values(), key=lambda f: f.id):
        nid = f"formula:{formula.id}"
        nodes[nid] = formula.id
        labels[nid] = f"{formula.id}: ${_formula_math(scope, formula)}$"

    return nodes, labels


def _expression_nodes(scope: Scope) -> tuple[dict[str, str], dict[str, str]]:
    """Node ids and labels for expressions."""
    nodes: dict[str, str] = {}
    labels: dict[str, str] = {}
    for expression in sorted(scope.expressions.values(), key=lambda e: e.id):
        nid = f"expression:{expression.id}"
        nodes[nid] = expression.id
        labels[nid] = f"{expression.id}: ${sympy.latex(_parse_expr(scope, expression.expression))}$"
    return nodes, labels


def _parse_expr(scope: Scope, text: str) -> sympy.Expr:
    if "=" in text and "==" not in text and ">=" not in text and "<=" not in text:
        left, right = text.split("=", 1)
        text = f"({left}) - ({right})"
    return scope.session.parse(text)[1]


def _formula_math(scope: Scope, formula: Formula) -> str:
    if "=" in formula.expression and not any(op in formula.expression for op in ("==", ">=", "<=")):
        left, right = formula.expression.split("=", 1)
        return f"{sympy.latex(scope.session.parse(left)[1])} = {sympy.latex(scope.session.parse(right)[1])}"
    return sympy.latex(_parse_expr(scope, formula.expression))


def _formula_latex(scope: Scope, formula: Formula) -> str:
    return rf"\(\text{{{_tex_text(formula.kind)}: }} {_formula_math(scope, formula)}\)"


def _expression_latex(scope: Scope, expression: Expression) -> str:
    expr = _parse_expr(scope, expression.expression)
    return sympy.latex(expr)


def _variable_latex(scope: Scope, variable: Variable) -> str:
    notation = _math_symbol(variable)
    meta = []
    if variable.units:
        meta.append(f"units: {_tex_text(variable.units)}")
    if variable.semantics:
        meta.append(f"semantics: {_tex_text(variable.semantics)}")
    if variable.links:
        meta.append(f"links: {_tex_text(', '.join(sorted(variable.links)))}")
    header = rf"\({notation}\)"
    return f"{header}; {'; '.join(meta)}" if meta else header


def _formula_node_id(formula: Formula) -> str:
    return f"formula:{formula.id}"


def _variable_node_id(variable: Variable) -> str:
    return f"var:{variable.name}"


def _expression_node_id(expression: Expression) -> str:
    return f"expression:{expression.id}"


def _node_kind(nid: str) -> str:
    if nid.startswith("formula:"):
        return "formula"
    if nid.startswith("expression:"):
        return "expression"
    if nid.startswith("assumption:"):
        return "assumption"
    return "variable"


def export_scope(scope: Scope, args: dict[str, Any]) -> dict[str, Any]:
    """Export a scope as LaTeX, DOT or JSON.

    **LaTeX** produces a self-contained document with an internal legend: every
    variable name is a hyperlink target, and every formula points at the
    variable-link edges. **JSON** returns the same graph structure for callers
    that want to render it themselves.
    """
    fmt = str(args.get("format", "latex"))
    noun = args.get("noun")
    item_id = args.get("id")
    expr_arg = args.get("expr")
    if expr_arg is not None and (noun or item_id):
        raise ContractError("expr cannot be combined with noun/id")

    if fmt not in ("latex", "dot", "json"):
        raise ContractError(f"unknown export format `{fmt}`; use latex, dot or json")

    if expr_arg is not None:
        if expr_arg.startswith("e:"):
            expr = scope.session.get(expr_arg)
        else:
            _, expr = scope.session.parse(expr_arg)
        if fmt == "latex":
            return {"format": "latex", "content": sympy.latex(expr)}
        selected = Scope(scope.name, scope.session)
        selected.define_expression(Expression("input", sympy.sstr(expr)))
        return export_scope(selected, {"format": fmt})

    if noun is not None:
        if noun not in ("formula", "expression", "variable"):
            raise ContractError(f"unknown export noun `{noun}`")
        selected = _selected_scope(scope, noun, item_id)
        return export_scope(selected, {"format": fmt})
    if item_id is not None:
        raise ContractError("id requires noun")

    nodes: dict[str, str] = {}
    edges: list[dict[str, str]] = []
    node_labels: dict[str, str] = {}

    vnodes, vlabels = _variable_nodes(scope)
    nodes.update(vnodes)
    node_labels.update(vlabels)
    fnodes, flabels = _formula_nodes(scope)
    nodes.update(fnodes)
    node_labels.update(flabels)
    enodes, elabels = _expression_nodes(scope)
    nodes.update(enodes)
    node_labels.update(elabels)

    for formula in sorted(scope.formulas.values(), key=lambda f: f.id):
        for variable_name in sorted({scope.canonical(str(s)) for s in _parse_expr(scope, formula.expression).free_symbols}):
            edges.append({"from": _formula_node_id(formula), "to": f"var:{variable_name}"})
    for expression in sorted(scope.expressions.values(), key=lambda e: e.id):
        for variable_name in sorted({scope.canonical(str(s)) for s in _parse_expr(scope, expression.expression).free_symbols}):
            edges.append({"from": _expression_node_id(expression), "to": f"var:{variable_name}"})

    for variable in sorted(scope.variables.values(), key=lambda v: v.name):
        for link in variable.links:
            target = scope.canonical(link)
            if target in scope.variables and target != variable.name:
                edges.append({"from": _variable_node_id(variable), "to": _variable_node_id(Variable(target)), "style": "link"})

    for assumption in sorted(scope.assumptions.values(), key=lambda a: a.id):
        nid = f"assumption:{assumption.id}"
        nodes[nid] = assumption.id
        node_labels[nid] = assumption.statement
    for formula in scope.formulas.values():
        for assumption_id in formula.assumes:
            if assumption_id in scope.assumptions:
                edges.append({"from": _formula_node_id(formula), "to": f"assumption:{assumption_id}", "style": "dashed"})

    if fmt == "dot":
        lines = ["digraph math {", '  rankdir=LR;', '  node [shape=box, style=rounded];']
        for nid, label in node_labels.items():
            lines.append(f"  {json.dumps(nid)} [label={json.dumps(label)}];")
        for edge in edges:
            style = edge.get("style", "")
            attr = ' [style=dotted]' if style == "link" else (' [style=dashed]' if style else "")
            lines.append(f"  {json.dumps(edge['from'])} -> {json.dumps(edge['to'])}{attr};")
        lines.append("}")
        content = "\n".join(lines)
        return {"format": "dot", "content": content}

    graph: dict[str, Any] = {
        "directed": True,
        "nodes": [{"id": nid, "label": node_labels[nid], "kind": _node_kind(nid)} for nid in nodes],
        "edges": edges,
    }
    if fmt == "json":
        return {"format": "json", "content": "", "graph": graph, "nodes": nodes, "edges": edges}

    return _latex_document(scope, node_labels, edges)


def _selected_scope(scope: Scope, noun: str, item_id: str | None) -> Scope:
    """Keep requested records and the variables their expressions refer to."""
    selected = scope.fork(scope.name)
    selected.formulas = {}
    selected.expressions = {}
    selected.assumptions = {}
    if noun == "variable":
        items = [v for v in scope.variables.values() if not item_id or v.name == item_id or item_id in v.aliases]
        if not items:
            raise ContractError(f"unknown variable `{item_id}`")
        selected.variables = {v.name: v for v in items}
        return selected
    if noun == "formula":
        items = [f for f in scope.formulas.values() if not item_id or f.id == item_id]
        if not items:
            raise ContractError(f"unknown formula `{item_id}`")
        selected.formulas = {f.id: f for f in items}
        selected.assumptions = {
            name: scope.assumptions[name] for f in items for name in f.assumes if name in scope.assumptions
        }
        expressions = [f.expression for f in items]
    else:
        items = [e for e in scope.expressions.values() if not item_id or e.id == item_id]
        if not items:
            raise ContractError(f"unknown expression `{item_id}`")
        selected.expressions = {e.id: e for e in items}
        expressions = [e.expression for e in items]
    names = {scope.canonical(str(symbol)) for text in expressions for symbol in _parse_expr(scope, text).free_symbols}
    selected.variables = {name: scope.variables[name] for name in names if name in scope.variables}
    return selected


def _latex_document(scope: Scope, labels: dict[str, str], edges: list[dict[str, str]]) -> dict[str, str]:
    var_lines = []
    formula_lines = []
    expr_lines = []
    legend_lines = []

    for variable in sorted(scope.variables.values(), key=lambda v: v.name):
        nid = f"var:{variable.name}"
        var_lines.append(f"\\item \\hyperlink{{{_safe_label(nid)}}}{{\\({_math_symbol(variable)}\\)}}")
        legend_lines.append(
            f"\\par\\hypertarget{{{_safe_label(nid)}}}{{}}{_variable_latex(scope, variable)}"
        )

    for formula in sorted(scope.formulas.values(), key=lambda f: f.id):
        nid = f"formula:{formula.id}"
        legend_lines.append(f"\\par\\hypertarget{{{_safe_label(nid)}}}{{}}{_formula_latex(scope, formula)}")
        formula_lines.append(f"\\item \\hyperlink{{{_safe_label(nid)}}}{{{_tex_text(formula.id)}}}")

    for expression in sorted(scope.expressions.values(), key=lambda e: e.id):
        nid = f"expression:{expression.id}"
        expr_lines.append(f"\\item \\hyperlink{{{_safe_label(nid)}}}{{{_tex_text(expression.id)}}}")
        legend_lines.append(f"\\par\\hypertarget{{{_safe_label(nid)}}}{{}}\\({_expression_latex(scope, expression)}\\)")

    legend = "\n".join(legend_lines)
    sections = []
    if var_lines:
        sections.append("\\section*{Variables}\n\\begin{itemize}\n" + "\n".join(var_lines) + "\n\\end{itemize}")
    if formula_lines:
        sections.append("\\section*{Formulas}\n\\begin{itemize}\n" + "\n".join(formula_lines) + "\n\\end{itemize}")
    if expr_lines:
        sections.append("\\section*{Expressions}\n\\begin{itemize}\n" + "\n".join(expr_lines) + "\n\\end{itemize}")

    body = "\n".join(sections)
    latex = (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{amsmath}\n"
        "\\usepackage{hyperref}\n"
        "\\usepackage{geometry}\n"
        "\\geometry{margin=1in}\n"
        "\\hypersetup{colorlinks=true, linkcolor=blue, urlcolor=blue}\n"
        "\\title{Mathematical Scope Export}\n"
        "\\author{adv-math-mcp}\n"
        "\\begin{document}\n"
        "\\maketitle\n"
        "\\tableofcontents\n"
        "\\newpage\n"
        "\\section*{Legend}\n"
        + legend
        + "\n\\newpage\n"
        + body
        + "\n\\end{document}"
    )
    return {"format": "latex", "content": latex}


def explain(scope: Scope, args: dict[str, Any]) -> dict[str, Any]:
    """Explain a formula or expression with metadata and no verification claim."""
    noun = str(args.get("noun", "formula"))
    item_id = args.get("id")
    if not isinstance(item_id, str) or not item_id:
        raise ContractError("id is required for explain")

    if noun == "formula":
        item = scope.formulas.get(item_id)
        if item is None:
            raise ContractError(f"unknown formula `{item_id}`")
        expr = _parse_expr(scope, item.expression)
        rendered = _formula_math(scope, item)
        variables = {
            scope.canonical(str(symbol)): {
                "name": scope.canonical(str(symbol)),
                "semantics": scope.variables[scope.canonical(str(symbol))].semantics,
                "units": scope.variables[scope.canonical(str(symbol))].units,
                "symbol": scope.variables[scope.canonical(str(symbol))].symbol,
                "subscript": scope.variables[scope.canonical(str(symbol))].subscript,
                "superscript": scope.variables[scope.canonical(str(symbol))].superscript,
                "links": list(scope.variables[scope.canonical(str(symbol))].links),
            }
            for symbol in expr.free_symbols
        }
        direct = list(item.assumes)
        _ = scope.assumptions
        related = [
            other.id
            for other in scope.formulas.values()
            if other.id != item_id and set(scope.relation_map().get(other.id, set())) & set(variables)
        ]
        return {
            "kind": "formula",
            "id": item.id,
            "kind_of": item.kind,
            "expression": item.expression,
            "latex": rendered,
            "variables": variables,
            "direct_assumptions": direct,
            "missing_assumptions": [a for a in direct if a not in scope.assumptions],
            "inactive_assumptions": [a for a in direct if a in scope.assumptions and not scope.assumptions[a].active],
            "related_formulas": related,
            "summary": (
                f"`{item.id}` is a {item.kind}: {item.expression}. It uses variables "
                + ", ".join(sorted(variables))
                + " and directly assumes "
                + ", ".join(direct or ["none"])
                + ". This is descriptive metadata only; no claim about truth is made."
            ),
        }

    item = scope.expressions.get(item_id)
    if item is None:
        raise ContractError(f"unknown expression `{item_id}`")
    expr = _parse_expr(scope, item.expression)
    return {
        "kind": "expression",
        "id": item.id,
        "expression": item.expression,
        "description": item.description,
        "latex": sympy.latex(expr),
        "variables": {
            scope.canonical(str(symbol)): {
                "name": scope.canonical(str(symbol)),
                "semantics": scope.variables[scope.canonical(str(symbol))].semantics,
                "units": scope.variables[scope.canonical(str(symbol))].units,
                "symbol": scope.variables[scope.canonical(str(symbol))].symbol,
                "subscript": scope.variables[scope.canonical(str(symbol))].subscript,
                "superscript": scope.variables[scope.canonical(str(symbol))].superscript,
                "links": list(scope.variables[scope.canonical(str(symbol))].links),
            }
            for symbol in expr.free_symbols
        },
        "summary": (
            f"`{item.id}` is the expression {item.expression}. It mentions variables "
            + ", ".join(sorted({scope.canonical(str(s)) for s in expr.free_symbols}))
            + ". This is descriptive metadata only; no claim about truth is made."
        ),
    }
