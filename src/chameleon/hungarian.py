"""Deterministic integer Hungarian assignment, preserving exact byte costs."""

from dataclasses import dataclass

from .contracts import _integer


@dataclass(frozen=True)
class Assignment:
    columns: tuple[int, ...]
    total_cost: int


def hungarian(costs) -> Assignment:
    """Minimize a square cost matrix; ascending indices resolve algorithm ties."""
    matrix = tuple(tuple(row) for row in costs)
    n = len(matrix)
    if not n or any(len(row) != n for row in matrix):
        raise ValueError("cost matrix must be nonempty and square")
    for row in matrix:
        for cost in row:
            _integer("cost", cost, 0)
    u, v, owner, previous = ([0] * (n + 1) for _ in range(4))
    for row in range(1, n + 1):
        owner[0], column = row, 0
        minimum, visited = [None] * (n + 1), [False] * (n + 1)
        while True:
            visited[column] = True
            current, delta, next_column = owner[column], None, 0
            for target in range(1, n + 1):
                if visited[target]:
                    continue
                reduced = matrix[current - 1][target - 1] - u[current] - v[target]
                if minimum[target] is None or reduced < minimum[target]:
                    minimum[target], previous[target] = reduced, column
                if delta is None or minimum[target] < delta:
                    delta, next_column = minimum[target], target
            for target in range(n + 1):
                if visited[target]:
                    u[owner[target]] += delta
                    v[target] -= delta
                elif minimum[target] is not None:
                    minimum[target] -= delta
            column = next_column
            if owner[column] == 0:
                break
        while column:
            parent = previous[column]
            owner[column] = owner[parent]
            column = parent
    columns = [0] * n
    for target in range(1, n + 1):
        columns[owner[target] - 1] = target - 1
    return Assignment(tuple(columns), sum(matrix[i][j] for i, j in enumerate(columns)))
