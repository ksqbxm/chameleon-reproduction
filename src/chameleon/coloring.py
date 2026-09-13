"""Model-unit communication conflicts and deterministic greedy DSATUR rounds."""

from itertools import combinations


def conflict_graph(unit_devices: dict[str, tuple[str, ...]]) -> dict[str, frozenset[str]]:
    """Include endpoints; a device may participate in only one unit per round."""
    devices = {}
    for unit, owners in unit_devices.items():
        if (not isinstance(unit, str) or not unit.strip() or not owners
                or any(not isinstance(owner, str) or not owner.strip() for owner in owners)
                or len(set(owners)) != len(owners)):
            raise ValueError("units require nonempty unique device IDs")
        devices[unit] = set(owners)
    graph = {unit: set() for unit in sorted(devices)}
    for left, right in combinations(graph, 2):
        if devices[left] & devices[right]:
            graph[left].add(right)
            graph[right].add(left)
    return {unit: frozenset(peers) for unit, peers in graph.items()}


def dsatur(graph: dict[str, frozenset[str]]) -> tuple[tuple[str, ...], ...]:
    """Saturation, degree, then stable ID; no general minimum-color guarantee."""
    if any(not isinstance(unit, str) or not unit.strip() for unit in graph):
        raise ValueError("graph IDs must be nonempty strings")
    for unit, peers in graph.items():
        if unit in peers or any(peer not in graph or unit not in graph[peer] for peer in peers):
            raise ValueError("conflict graph must be undirected without self edges")
    colors = {}
    while len(colors) < len(graph):
        uncolored = set(graph) - colors.keys()
        unit = min(uncolored, key=lambda u: (-len({colors[p] for p in graph[u] if p in colors}),
                                             -len(graph[u]), u))
        forbidden = {colors[peer] for peer in graph[unit] if peer in colors}
        color = 0
        while color in forbidden:
            color += 1
        colors[unit] = color
    return tuple(tuple(sorted(unit for unit in graph if colors[unit] == color))
                 for color in range(max(colors.values(), default=-1) + 1))
