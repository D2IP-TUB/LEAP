"""Stateless stdio MCP server for LEAP table transformations."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from leap.core import Action, Table

mcp = FastMCP(
    "leap-table-transformations",
    instructions="Apply one LEAP table transformation and return the resulting table state.",
)


def _apply(table_data: dict[str, Any], action: Action) -> dict[str, Any]:
    table = Table.from_dict(table_data)
    if not action.is_valid_for_table(table):
        raise ValueError(f"Invalid {action.name} arguments for the supplied table")
    result = action.apply_to_table(table)
    if result is None:
        raise ValueError(f"Failed to apply {action.name}")
    return {
        "action": action.to_string(),
        "table": result.to_dict(),
        "terminated": action.name == "end",
    }


@mcp.tool()
def select_row(table: dict[str, Any], rows: list[str]) -> dict[str, Any]:
    """Select table rows named as 'row N'."""
    indices: list[int] = []
    if rows == ["*"]:
        arguments: list[Any] = ["*"]
    else:
        for row in rows:
            prefix, separator, index = row.partition(" ")
            if prefix != "row" or separator != " " or not index.isdigit():
                raise ValueError(f"Invalid row label: {row!r}")
            indices.append(int(index))
        arguments = indices
    return _apply(table, Action("select_row", arguments))


@mcp.tool()
def select_column(table: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """Select a non-empty subset of table columns."""
    return _apply(table, Action("select_column", columns))


@mcp.tool()
def group_by(table: dict[str, Any], column: str) -> dict[str, Any]:
    """Group rows by one column and count occurrences."""
    return _apply(table, Action("group_by", [column]))


@mcp.tool()
def sort_by(table: dict[str, Any], column: str, order: str) -> dict[str, Any]:
    """Sort rows by one column in 'asc' or 'desc' order."""
    return _apply(table, Action("sort_by", [column, order]))


@mcp.tool()
def end(table: dict[str, Any]) -> dict[str, Any]:
    """Return the table unchanged and terminate the transformation chain."""
    return _apply(table, Action("end", []))


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
