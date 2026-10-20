"""Deterministic checks of temporal relations against the event intervals they relate.

The relation ontology (``RelationType``) has interval semantics; this module states them once and
checks every relation against the two events' timestamps, independently of the code that built
the relation. It also looks for cycles in the BEFORE graph, which the ontology forbids.
"""

from __future__ import annotations

from video_dataset.schemas.events import Event, RelationType, TemporalRelation

CAUSAL = {RelationType.CAUSES}


def relation_holds(rel: TemporalRelation, a: Event, b: Event, tol: float = 0.3) -> tuple[bool, str | None]:
    """Whether ``rel`` (a -> b) is consistent with the two events' intervals."""
    try:
        r = RelationType(str(rel.relation))
    except ValueError:
        return False, f"unknown relation type {rel.relation!r}"
    if r == RelationType.BEFORE:
        ok = a.end_time <= b.start_time + tol and a.start_time <= b.start_time + tol
        return ok, None if ok else f"BEFORE requires a.end ({a.end_time:.2f}) <= b.start ({b.start_time:.2f})"
    if r == RelationType.AFTER:
        ok = b.end_time <= a.start_time + tol
        return ok, None if ok else f"AFTER requires b.end ({b.end_time:.2f}) <= a.start ({a.start_time:.2f})"
    if r == RelationType.DURING:
        ok = a.start_time >= b.start_time - tol and a.end_time <= b.end_time + tol and (b.end_time - b.start_time) >= (a.end_time - a.start_time) - tol
        return ok, None if ok else "DURING requires a inside b"
    if r == RelationType.OVERLAPS:
        ok = a.start_time < b.start_time + tol and a.end_time > b.start_time - tol and a.end_time < b.end_time + tol
        return ok, None if ok else "OVERLAPS requires a.start < b.start < a.end < b.end"
    if r == RelationType.STARTS:
        ok = abs(a.start_time - b.start_time) <= tol and a.end_time <= b.end_time + tol
        return ok, None if ok else "STARTS requires equal starts and a shorter a"
    if r == RelationType.ENDS:
        ok = abs(a.end_time - b.end_time) <= tol and a.start_time >= b.start_time - tol
        return ok, None if ok else "ENDS requires equal ends and a shorter a"
    if r in (RelationType.CONTINUES, RelationType.CHANGES_TO, RelationType.CAUSES):
        ok = a.start_time <= b.start_time + tol
        return ok, None if ok else f"{r.value} requires a to start no later than b"
    if r == RelationType.INTERRUPTS:
        ok = a.start_time > b.start_time - tol and a.start_time < b.end_time + tol
        return ok, None if ok else "INTERRUPTS requires a to start inside b"
    return True, None


def before_cycles(relations: list[TemporalRelation]) -> list[list[str]]:
    """Cycles in the directed BEFORE graph (each as a list of event ids). An empty list = acyclic."""
    graph: dict[str, set[str]] = {}
    for r in relations:
        if r.relation == RelationType.BEFORE:
            graph.setdefault(r.event_a, set()).add(r.event_b)
    colour: dict[str, int] = {}
    stack: list[str] = []
    cycles: list[list[str]] = []

    def visit(n: str) -> None:
        colour[n] = 1
        stack.append(n)
        for m in graph.get(n, ()):
            c = colour.get(m, 0)
            if c == 0:
                visit(m)
            elif c == 1 and len(cycles) < 10:
                cycles.append(stack[stack.index(m):] + [m])
        stack.pop()
        colour[n] = 2

    for node in list(graph):
        if colour.get(node, 0) == 0:
            visit(node)
    return cycles
