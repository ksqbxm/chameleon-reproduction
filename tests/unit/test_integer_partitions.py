from itertools import product

import pytest

from chameleon.planner import integer_partitions


@pytest.mark.parametrize("allowed", [(1, 2, 3, 4), (1, 3), (2, 4), (4, 2, 2, 1)])
@pytest.mark.parametrize("survivors", range(1, 9))
@pytest.mark.parametrize("dp", range(1, 5))
def test_partitions_against_exhaustive_product(survivors, dp, allowed):
    expected = tuple(sorted({lengths for lengths in product(allowed, repeat=dp)
                             if sum(lengths) == survivors and tuple(sorted(lengths)) == lengths}))
    assert integer_partitions(survivors, dp, allowed) == expected


def test_asymmetric_partitions_use_all_survivors():
    assert integer_partitions(7, 2, range(1, 6)) == ((2, 5), (3, 4))
    assert integer_partitions(3, 2, (2, 3)) == ()


@pytest.mark.parametrize("args", [(0, 1, (1,)), (True, 1, (1,)), (2, 0, (1,)),
                                  (2, 1.5, (1,)), (2, 1, ()), (2, 1, (0, 2)), (2, 1, (True,))])
def test_invalid_partition_inputs(args):
    with pytest.raises(ValueError):
        integer_partitions(*args)
