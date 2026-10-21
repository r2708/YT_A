"""Leakage-safe, persisted splits: one split per source group (canonical video id), duplicate
groups pulled into one split, existing assignments kept across incremental runs, every change
reported."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dataset_factory.config import SplitRules


def _u(group: str, salt: str) -> float:
    return int(hashlib.sha1(f"{salt}|{group}".encode()).hexdigest()[:12], 16) / float(16**12)


def _clusters(groups: list[str], links: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Union-find over duplicate links: cluster representative (lexically smallest) -> members."""
    parent = {g: g for g in groups}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in links:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    out: dict[str, list[str]] = {}
    for g in groups:
        out.setdefault(find(g), []).append(g)
    return out


def assign(groups: list[str], rules: SplitRules, existing: dict[str, str] | None = None, links: list[tuple[str, str]] | None = None) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """(group -> split, change report). Existing assignments are kept; new groups get their hash
    bucket; groups linked by a duplicate relation share one split (an existing assignment in the
    cluster wins, else the representative's); eval splits are topped up to the minimums by moving
    whole *new, train-only* clusters with the lowest hash (train keeps at least one cluster)."""
    existing = dict(existing or {})
    groups = sorted(set(groups))
    out: dict[str, str] = {}
    changes: list[dict[str, Any]] = []
    test, val = max(0.0, rules.test_fraction), max(0.0, rules.validation_fraction)
    clusters = _clusters(groups, links or [])
    new_clusters: list[str] = []
    for rep, members in sorted(clusters.items()):
        known = sorted((m for m in members if m in existing), key=lambda m: (m != rep, m))
        if known:
            split = existing[known[0]]
        else:
            u = _u(rep, rules.salt)
            split = "test" if u < test else ("validation" if u < test + val else "train")
            new_clusters.append(rep)
        for m in members:
            out[m] = split
            if m not in existing:
                changes.append({"group": m, "from": None, "to": split, "why": "hash" if not known else f"duplicate_of:{known[0]}"})
            elif existing[m] != split:
                changes.append({"group": m, "from": existing[m], "to": split, "why": f"duplicate_of:{known[0]}"})
    counts = {s: sum(1 for rep, ms in clusters.items() if out[rep] == s) for s in ("train", "validation", "test")}
    want = {"test": max(0, int(rules.min_test_groups)), "validation": max(0, int(rules.min_validation_groups))}
    budget = len(clusters) - 1
    while want["test"] + want["validation"] > max(0, budget):
        if want["validation"] >= want["test"] and want["validation"] > 0:
            want["validation"] -= 1
        else:
            want["test"] -= 1
    movable = sorted((rep for rep in new_clusters if out[rep] == "train"), key=lambda rep: (_u(rep, rules.salt), rep))
    for sp in ("test", "validation"):
        while counts[sp] < want[sp] and counts["train"] > 1 and movable:
            rep = movable.pop(0)
            for m in clusters[rep]:
                out[m] = sp
                for c in changes:
                    if c["group"] == m:
                        c["to"], c["why"] = sp, f"top_up_min_{sp}"
            counts["train"] -= 1
            counts[sp] += 1
    return out, changes


def merge_duplicate_groups(assignment: dict[str, str], links: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Groups linked by a duplicate relation must share one split: the lexically smaller group's
    split wins; changes are reported."""
    changes = []
    for a, b in links:
        if a not in assignment or b not in assignment or assignment[a] == assignment[b]:
            continue
        keep, move = sorted((a, b))
        changes.append({"group": move, "from": assignment[move], "to": assignment[keep], "why": f"duplicate_of:{keep}"})
        assignment[move] = assignment[keep]
    return changes


def load_persisted(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return {str(k): str(v) for k, v in (json.loads(path.read_text()).get("groups") or {}).items()}
    except (ValueError, AttributeError):
        return {}


def persist(path: Path, assignment: dict[str, str], rules: SplitRules, changes: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {s: sum(1 for v in assignment.values() if v == s) for s in ("train", "validation", "test")}
    path.write_text(json.dumps({"method": "sha1(salt | group) buckets per source group, persisted; eval minimums topped up from new groups; duplicate groups merged", "salt": rules.salt, "fractions": {"validation": rules.validation_fraction, "test": rules.test_fraction}, "minimums": {"validation": rules.min_validation_groups, "test": rules.min_test_groups}, "counts": counts, "groups": dict(sorted(assignment.items()))}, indent=2))
    if changes:
        (path.parent / "split_changes.json").write_text(json.dumps(changes, indent=2))
