from itertools import permutations
import random

import pytest

from chameleon.hungarian import hungarian


def oracle(matrix):
    return min(sum(matrix[i][j] for i, j in enumerate(p)) for p in permutations(range(len(matrix))))


@pytest.mark.parametrize("n", range(1, 7))
def test_fixed_seed_matrices_against_permutation_oracle(n):
    rng = random.Random(700 + n)
    for _ in range(25):
        matrix = tuple(tuple(rng.randrange(20) for _ in range(n)) for _ in range(n))
        result = hungarian(matrix)
        assert sorted(result.columns) == list(range(n))
        assert result.total_cost == oracle(matrix)
        assert result.total_cost == sum(matrix[i][j] for i, j in enumerate(result.columns))
        assert result == hungarian(matrix)


def test_paper_figure3_layer_count_matrix():
    # Figure 3, DP2 -> DP2': rows (1,2,3), (4,5,6), (7,8,9), and an incoming node.
    matrix = ((0, 1, 2, 3), (2, 1, 0, 3), (2, 2, 2, 0), (2, 2, 2, 3))
    result = hungarian(matrix)
    assert result.total_cost == oracle(matrix) == 2


@pytest.mark.parametrize("n", range(1, 9))
def test_exact_ties_are_stable(n):
    assert hungarian([[0] * n for _ in range(n)]).columns == tuple(range(n))


def test_costs_above_float_precision_remain_exact():
    n = 2 ** 60
    matrix = ((n + 1, n), (n, n + 1))
    assert hungarian(matrix).columns == (1, 0)
    assert hungarian(matrix).total_cost == 2 * n


@pytest.mark.parametrize("matrix", ((), ((0, 1),), ((0,), (1,)), ((-1,),), ((True,),), ((1.5,),),
                                    ((float("nan"),),), ((float("inf"),),)))
def test_invalid_matrices(matrix):
    with pytest.raises(ValueError):
        hungarian(matrix)
