from itertools import combinations

import pytest

from chameleon.coloring import conflict_graph, dsatur


def graph_of_edges(units, edges):
    graph = {unit: set() for unit in units}
    for left, right in edges:
        graph[left].add(right)
        graph[right].add(left)
    return graph


@pytest.mark.parametrize("kind,n,expected", (("path", 6, 2), ("even_cycle", 6, 2),
                                            ("odd_cycle", 5, 3), ("complete", 5, 5),
                                            ("isolated", 5, 1)))
def test_standard_graphs(kind, n, expected):
    units = tuple(str(i) for i in range(n))
    edges = list(zip(units, units[1:])) if kind in ("path", "even_cycle", "odd_cycle") else []
    if "cycle" in kind:
        edges.append((units[-1], units[0]))
    if kind == "complete":
        edges = list(combinations(units, 2))
    graph = graph_of_edges(units, edges)
    rounds = dsatur(graph)
    assert len(rounds) == expected
    colors = {unit: color for color, row in enumerate(rounds) for unit in row}
    assert set(colors) == set(units)
    assert all(colors[left] != colors[right] for left, right in edges)
    assert rounds == dsatur({unit: graph[unit] for unit in reversed(units)})


def test_saturation_precedes_degree_and_id():
    graph = graph_of_edges("abcde", (("a", "b"), ("a", "c"), ("b", "c"), ("b", "d"), ("d", "e")))
    assert dsatur(graph) == (("b", "e"), ("a", "d"), ("c",))


def test_asymmetric_partitions_include_endpoints_and_have_no_device_conflicts():
    owners = {"embedding": ("p0s0", "p1s0"), "blocks.0": ("p0s0", "p1s1"),
              "blocks.1": ("p0s1", "p1s1"), "final_norm": ("p0s1", "p1s2"),
              "lm_head": ("p0s1", "p1s2"), "extra": ("p0s1", "p1s2")}
    graph = conflict_graph(owners)
    for left, right in combinations(owners, 2):
        assert (right in graph[left]) == bool(set(owners[left]) & set(owners[right]))
    for row in dsatur(graph):
        devices = [device for unit in row for device in owners[unit]]
        assert len(devices) == len(set(devices))
    assert conflict_graph(dict(reversed(list(owners.items())))) == graph


def test_empty_graph():
    assert dsatur(conflict_graph({})) == ()


@pytest.mark.parametrize("graph", ({"a": {"a"}}, {"a": {"b"}}, {"a": {"b"}, "b": set()}, {"": set()}))
def test_invalid_graph(graph):
    with pytest.raises(ValueError):
        dsatur(graph)


@pytest.mark.parametrize("owners", ({"a": ()}, {"a": ("d", "d")}, {"": ("d",)}, {"a": ("",)}))
def test_invalid_unit_devices(owners):
    with pytest.raises(ValueError):
        conflict_graph(owners)


def test_device_id_cannot_be_interpreted_as_a_sequence_of_character_ids():
    with pytest.raises(ValueError):
        conflict_graph({"embedding": "gpu0", "lm_head": "gpu1"})
